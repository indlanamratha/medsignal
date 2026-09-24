"""Disproportionality analysis: find drug-reaction pairs reported more often than expected.

For each suspect drug D and reaction R, reports are split into a 2x2 table:

                     reaction R    all other reactions
    drug D               a                 b
    other study drugs    c                 d

Comparator: reports for the other 12 study drug groups (an active-comparator
design). Because every comparator is also an obesity or diabetes drug, this
reduces confounding by indication: differences are less likely to reflect the
underlying disease and more likely to reflect the drug.

Signal rule (Evans criteria): PRR >= 2, chi-square >= 4, and at least 3 reports.
"""
import duckdb
import numpy as np
import pandas as pd

from medsignal.config import DATA_DIR, PROJECT_ROOT

DATABASE = DATA_DIR / "medsignal.duckdb"
RESULTS = PROJECT_ROOT / "analytics" / "results"
MIN_REPORTS = 3

# Terms that are not medical reactions: medication errors, product issues, and
# outcomes such as death, which need separate handling.
EXCLUDED_TERMS_SQL = """
    reaction_pt not like '%DOSE%'
    and reaction_pt not like '%ADMINIST%'
    and reaction_pt not like '%TECHNIQUE%'
    and reaction_pt not like 'PRODUCT %'
    and reaction_pt not like '%MEDICATION ERROR%'
    and reaction_pt not like '%DEVICE%'
    and reaction_pt not like '%ERROR%'
    and reaction_pt not like 'THERAPEUTIC %'
    and reaction_pt not in ('MEDICAL PROCEDURE', 'COLONOSCOPY', 'EMOTIONAL DISTRESS')
    and reaction_pt not in ('DEATH', 'OFF LABEL USE', 'NO ADVERSE EVENT', 'DRUG INEFFECTIVE')
"""

COUNTS_SQL = f"""
with reactions as (
    select distinct report_id, drug_group, reaction_pt
    from fct_drug_reaction
    where {EXCLUDED_TERMS_SQL}
),
drug_n as (
    select drug_group, count(distinct report_id) as n_drug
    from fct_report_drug group by drug_group
),
reaction_n as (
    select reaction_pt, count(distinct report_id) as n_reaction
    from reactions group by reaction_pt
),
total as (
    select count(distinct report_id) as n_total from fct_report_drug
),
pairs as (
    select drug_group, reaction_pt, count(distinct report_id) as a
    from reactions group by drug_group, reaction_pt
)
select p.drug_group, p.reaction_pt, p.a, d.n_drug, r.n_reaction, t.n_total
from pairs p
join drug_n d using (drug_group)
join reaction_n r using (reaction_pt)
cross join total t
where p.a >= {MIN_REPORTS}
"""

# Reactions listed on GLP-1 labels (or reviewed by regulators), used to sanity-check the method.
KNOWN_GLP1_REACTIONS = [
    "NAUSEA", "VOMITING", "DIARRHOEA", "CONSTIPATION", "ABDOMINAL PAIN",
    "PANCREATITIS", "CHOLELITHIASIS", "IMPAIRED GASTRIC EMPTYING", "OPTIC ISCHAEMIC NEUROPATHY",
]


def compute_signals(counts: pd.DataFrame) -> pd.DataFrame:
    """Add 2x2 cells, PRR, ROR with 95% CI, chi-square, and signal flags.

    `counts` needs columns: a, n_drug, n_reaction, n_total.
    If any cell is zero, 0.5 is added to every cell (Haldane-Anscombe correction)
    so the statistics stay finite.
    """
    df = counts.copy()
    df["b"] = df["n_drug"] - df["a"]
    df["c"] = df["n_reaction"] - df["a"]
    df["d"] = df["n_total"] - df["n_drug"] - df["c"]

    a, b, c, d = (df[col].astype(float) for col in ("a", "b", "c", "d"))
    zero = ((a == 0) | (b == 0) | (c == 0) | (d == 0)).astype(float) * 0.5
    a, b, c, d = a + zero, b + zero, c + zero, d + zero
    n = a + b + c + d

    df["prr"] = (a / (a + b)) / (c / (c + d))
    df["ror"] = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    df["ror_ci_low"] = np.exp(np.log(df["ror"]) - 1.96 * se)
    df["ror_ci_high"] = np.exp(np.log(df["ror"]) + 1.96 * se)
    # Chi-square with Yates continuity correction.
    df["chi2"] = (n * np.maximum(np.abs(a * d - b * c) - n / 2, 0) ** 2
                  / ((a + b) * (c + d) * (a + c) * (b + d)))

    df["is_signal"] = (df["prr"] >= 2) & (df["chi2"] >= 4) & (df["a"] >= MIN_REPORTS)
    df["ror_signal"] = df["ror_ci_low"] > 1
    df["high_confidence"] = df["is_signal"] & (df["a"] >= 20) & (df["ror_ci_low"] >= 2)
    return df


def run() -> None:
    con = duckdb.connect(str(DATABASE))
    counts = con.sql(COUNTS_SQL).df()
    signals = compute_signals(counts)
    con.register("signals_df", signals)
    con.execute("create or replace table mart_signals as select * from signals_df")
    con.close()

    RESULTS.mkdir(parents=True, exist_ok=True)
    flagged = signals[signals["is_signal"]].sort_values(["drug_group", "ror_ci_low"], ascending=[True, False])
    flagged.to_csv(RESULTS / "signals.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"Tested {len(signals):,} drug-reaction pairs (at least {MIN_REPORTS} reports each).")
    print(f"Signals found: {len(flagged):,}\n")

    summary = signals.groupby("drug_group").agg(pairs_tested=("a", "size"), signals=("is_signal", "sum"), high_confidence=("high_confidence", "sum"))
    print("===== Signals per drug =====")
    print(summary.sort_values("signals", ascending=False).to_string())

    cols = ["reaction_pt", "a", "prr", "ror", "ror_ci_low", "chi2"]
    for drug in ["semaglutide", "tirzepatide"]:
        top = flagged[(flagged["drug_group"] == drug) & (flagged["a"] >= 20)].head(10)
        print(f"\n===== Strongest signals: {drug} (at least 20 reports, ranked by ROR lower CI) =====")
        print(top[cols].round(2).to_string(index=False))

    check = signals[(signals["drug_group"] == "semaglutide")
                    & signals["reaction_pt"].isin(KNOWN_GLP1_REACTIONS)]
    print("\n===== Sanity check: known GLP-1 reactions for semaglutide =====")
    print(check[cols + ["is_signal"]].round(2).sort_values("prr", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run()