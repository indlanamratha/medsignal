"""Build one row of model features per report.

Target: is_serious (the report was serious: death, hospitalization, life-threatening,
disability, congenital anomaly, or other serious outcome).

Leakage rules: the model must only see what is known when a report arrives, and
nothing that directly encodes the outcome. So it never sees the seriousness flags,
the reaction outcome field, or reaction terms that describe the outcome itself
(death, hospitalisation, fatal events).
"""
import re

import duckdb
import pandas as pd

TOP_REACTIONS = 100

LEAKAGE_FREE_REACTIONS_SQL = """
    reaction_pt not like '%DEATH%'
    and reaction_pt not like '%FATAL%'
    and reaction_pt not like '%HOSPITAL%'
    and reaction_pt not like '%DISABILITY%'
    and reaction_pt <> 'COMPLETED SUICIDE'
"""


def _clean(name: str) -> str:
    """Make a feature name safe for XGBoost and LightGBM."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()


def _flags(long: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Turn (report_id, value) rows into one 0/1 column per value."""
    wide = pd.crosstab(long["report_id"], long[column]).clip(upper=1)
    wide.columns = [f"{prefix}{_clean(c)}" for c in wide.columns]
    return wide


def build_features(con: duckdb.DuckDBPyConnection, train_end_year: int) -> pd.DataFrame:
    """Return report_id, received_date, is_serious, and all feature columns.

    The top reaction terms are chosen from the training years only, so the
    feature set never peeks at validation or test data.
    """
    reports = con.sql("""
        select r.report_id, r.received_date, r.is_serious,
               r.age_years, r.weight_kg, r.sex, r.reporter_type_label,
               coalesce(r.country = 'US', false)::int as is_us,
               (r.country is null)::int as country_missing
        from stg_reports r
        where r.report_id in (select report_id from fct_report_drug)
    """).df()

    drugs = con.sql("select distinct report_id, drug_group from fct_report_drug").df()
    drug_counts = con.sql("""
        select report_id,
               count(*) filter (where drug_role = 'Suspect') as n_suspect_drugs,
               count(*) filter (where drug_role = 'Concomitant') as n_concomitant_drugs
        from stg_report_drugs
        where report_id in (select report_id from fct_report_drug)
        group by report_id
    """).df()
    reactions = con.sql(f"""
        select distinct report_id, reaction_pt
        from stg_report_reactions
        where report_id in (select report_id from fct_report_drug)
          and {LEAKAGE_FREE_REACTIONS_SQL}
    """).df()

    train_ids = reports.loc[reports["received_date"].dt.year <= train_end_year, "report_id"]
    top_terms = (reactions[reactions["report_id"].isin(train_ids)]["reaction_pt"]
                 .value_counts().head(TOP_REACTIONS).index)

    n_reactions = reactions.groupby("report_id").size().rename("n_reactions")
    reaction_flags = _flags(reactions[reactions["reaction_pt"].isin(top_terms)], "reaction_pt", "rx_")
    drug_flags = _flags(drugs, "drug_group", "drug_")

    df = (reports.set_index("report_id")
          .join(drug_counts.set_index("report_id"))
          .join(n_reactions)
          .join(drug_flags)
          .join(reaction_flags))

    df["age_missing"] = df["age_years"].isna().astype(int)
    df["weight_missing"] = df["weight_kg"].isna().astype(int)
    df = pd.get_dummies(df, columns=["sex", "reporter_type_label"], prefix=["sex", "reporter"], dtype=int)
    df.columns = [c if c in ("received_date", "is_serious") else _clean(c) for c in df.columns]

    count_cols = [c for c in df.columns if c.startswith(("rx_", "drug_", "n_"))]
    df[count_cols] = df[count_cols].fillna(0).astype(int)
    df["is_serious"] = df["is_serious"].astype(int)
    return df.reset_index()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("report_id", "received_date", "is_serious")]