"""Flatten raw FAERS JSON (bronze) into three clean Parquet tables (silver).

Step 1 (per file): extract the fields we need from each bronze file into
        small staging Parquet files. Re-running skips finished files.
Step 2 (all files): use DuckDB to remove duplicate reports and write
        data/silver/reports.parquet, report_drugs.parquet, report_reactions.parquet
"""
import gzip
from datetime import date, datetime
from pathlib import Path

import duckdb
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from medsignal.config import BRONZE_DIR, SILVER_DIR

FAERS_DIR = BRONZE_DIR / "faers"
STAGING_DIR = SILVER_DIR / "_staging"

# FAERS age unit codes -> multiply by this to get years
AGE_UNIT_TO_YEARS = {
    "800": 10.0,         # decade
    "801": 1.0,          # year
    "802": 1 / 12,       # month
    "803": 1 / 52.1775,  # week
    "804": 1 / 365.25,   # day
    "805": 1 / 8766,     # hour
}

REPORT_SCHEMA = pa.schema([
    ("safetyreportid", pa.string()),
    ("safetyreportversion", pa.int32()),
    ("receivedate", pa.date32()),
    ("receiptdate", pa.date32()),
    ("is_serious", pa.bool_()),
    ("died", pa.bool_()),
    ("hospitalized", pa.bool_()),
    ("life_threatening", pa.bool_()),
    ("disabled", pa.bool_()),
    ("congenital_anomaly", pa.bool_()),
    ("other_serious", pa.bool_()),
    ("reporter_type", pa.int32()),
    ("country", pa.string()),
    ("age_years", pa.float64()),
    ("sex", pa.int32()),
    ("weight_kg", pa.float64()),
])

DRUG_SCHEMA = pa.schema([
    ("safetyreportid", pa.string()),
    ("safetyreportversion", pa.int32()),
    ("medicinal_product", pa.string()),
    ("generic_name", pa.string()),
    ("brand_name", pa.string()),
    ("drug_role", pa.int32()),      # 1 suspect, 2 concomitant, 3 interacting
    ("indication", pa.string()),
])

REACTION_SCHEMA = pa.schema([
    ("safetyreportid", pa.string()),
    ("safetyreportversion", pa.int32()),
    ("reaction_pt", pa.string()),
    ("reaction_outcome", pa.int32()),
])


# ---------- small cleaning helpers ----------

def to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_date(value) -> date | None:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except (TypeError, ValueError):
        return None


def flag(value) -> bool:
    return str(value) == "1"


def clean_text(value) -> str | None:
    if not value:
        return None
    text = " ".join(str(value).split()).upper()
    return text or None


def first(values):
    return values[0] if isinstance(values, list) and values else None


def age_in_years(age, unit) -> float | None:
    age = to_float(age)
    factor = AGE_UNIT_TO_YEARS.get(str(unit))
    if age is None or factor is None:
        return None
    years = round(age * factor, 2)
    return years if 0 <= years <= 120 else None


# ---------- one report -> three kinds of rows ----------

def extract(report: dict) -> tuple[dict, list[dict], list[dict]]:
    report_id = report.get("safetyreportid")
    version = to_int(report.get("safetyreportversion")) or 1
    patient = report.get("patient") or {}
    source = report.get("primarysource") or {}

    report_row = {
        "safetyreportid": report_id,
        "safetyreportversion": version,
        "receivedate": parse_date(report.get("receivedate")),
        "receiptdate": parse_date(report.get("receiptdate")),
        "is_serious": flag(report.get("serious")),
        "died": flag(report.get("seriousnessdeath")),
        "hospitalized": flag(report.get("seriousnesshospitalization")),
        "life_threatening": flag(report.get("seriousnesslifethreatening")),
        "disabled": flag(report.get("seriousnessdisabling")),
        "congenital_anomaly": flag(report.get("seriousnesscongenitalanomali")),
        "other_serious": flag(report.get("seriousnessother")),
        "reporter_type": to_int(source.get("qualification")),
        "country": clean_text(report.get("occurcountry") or report.get("primarysourcecountry")),
        "age_years": age_in_years(patient.get("patientonsetage"),
                                  patient.get("patientonsetageunit")),
        "sex": to_int(patient.get("patientsex")),
        "weight_kg": to_float(patient.get("patientweight")),
    }

    drug_rows = []
    for drug in patient.get("drug") or []:
        openfda = drug.get("openfda") or {}
        drug_rows.append({
            "safetyreportid": report_id,
            "safetyreportversion": version,
            "medicinal_product": clean_text(drug.get("medicinalproduct")),
            "generic_name": clean_text(first(openfda.get("generic_name"))),
            "brand_name": clean_text(first(openfda.get("brand_name"))),
            "drug_role": to_int(drug.get("drugcharacterization")),
            "indication": clean_text(drug.get("drugindication")),
        })

    reaction_rows = []
    for reaction in patient.get("reaction") or []:
        reaction_rows.append({
            "safetyreportid": report_id,
            "safetyreportversion": version,
            "reaction_pt": clean_text(reaction.get("reactionmeddrapt")),
            "reaction_outcome": to_int(reaction.get("reactionoutcome")),
        })

    return report_row, drug_rows, reaction_rows


# ---------- step 1: one bronze file -> three staging files ----------

def stage_file(path: Path) -> bool:
    """Returns True if the file was processed, False if it was already done."""
    part_name = f"{path.parent.name}__{path.name.removesuffix('.json.gz')}.parquet"
    targets = {name: STAGING_DIR / name / part_name
               for name in ("reports", "drugs", "reactions")}
    if all(target.exists() for target in targets.values()):
        return False

    with gzip.open(path, "rb") as f:
        records = orjson.loads(f.read())

    reports, drugs, reactions = [], [], []
    for record in records:
        report_row, drug_rows, reaction_rows = extract(record)
        reports.append(report_row)
        drugs.extend(drug_rows)
        reactions.extend(reaction_rows)

    for name, rows, schema in (("reports", reports, REPORT_SCHEMA),
                               ("drugs", drugs, DRUG_SCHEMA),
                               ("reactions", reactions, REACTION_SCHEMA)):
        target = targets[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(target.name + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp_path)
        tmp_path.rename(target)
    return True


# ---------- step 2: remove duplicates across all files ----------

def consolidate() -> dict[str, int]:
    staging = STAGING_DIR.as_posix()
    silver = SILVER_DIR.as_posix()
    con = duckdb.connect()

    # Keep one row per report: the latest version.
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{staging}/reports/*.parquet')
            QUALIFY row_number() OVER (
                PARTITION BY safetyreportid
                ORDER BY safetyreportversion DESC, receiptdate DESC NULLS LAST
            ) = 1
        ) TO '{silver}/reports.parquet' (FORMAT parquet)
    """)

    # Keep drug and reaction rows only for that kept version, without duplicates.
    child_tables = {
        "drugs": "d.medicinal_product, d.generic_name, d.brand_name, d.drug_role, d.indication",
        "reactions": "d.reaction_pt, d.reaction_outcome",
    }
    for name, columns in child_tables.items():
        con.execute(f"""
            COPY (
                SELECT DISTINCT d.safetyreportid, {columns}
                FROM read_parquet('{staging}/{name}/*.parquet') AS d
                JOIN read_parquet('{silver}/reports.parquet') AS r
                  ON d.safetyreportid = r.safetyreportid
                 AND d.safetyreportversion = r.safetyreportversion
            ) TO '{silver}/report_{name}.parquet' (FORMAT parquet)
        """)

    def count(sql_path: str) -> int:
        return con.execute(f"SELECT count(*) FROM read_parquet('{sql_path}')").fetchone()[0]

    return {
        "raw report rows (with duplicates)": count(f"{staging}/reports/*.parquet"),
        "unique reports": count(f"{silver}/reports.parquet"),
        "report-drug rows": count(f"{silver}/report_drugs.parquet"),
        "report-reaction rows": count(f"{silver}/report_reactions.parquet"),
    }


def run() -> None:
    files = sorted(FAERS_DIR.glob("*/*.json.gz"))
    print(f"Found {len(files)} bronze files.")
    for i, path in enumerate(files, start=1):
        if stage_file(path):
            print(f"[{i}/{len(files)}] staged {path.parent.name}/{path.name}")

    print("\nRemoving duplicates and writing silver tables...")
    for name, value in consolidate().items():
        print(f"  {name:<36} {value:>12,}")


if __name__ == "__main__":
    run()