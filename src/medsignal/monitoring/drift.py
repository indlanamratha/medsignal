"""Data drift monitoring: does new data still look like the data the model learned from?

Compares feature distributions (and the model's predicted probabilities) for a
recent year against the training years, using Evidently. Saves an HTML report
and a CSV summary, and recommends retraining when many features have drifted.

Retraining is recommended, not automatic: a person reviews the new model in
MLflow before it replaces the current champion.
"""
import re
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from medsignal.config import DATA_DIR, PROJECT_ROOT
from medsignal.ml.features import build_features

DATABASE = DATA_DIR / "medsignal.duckdb"
MODEL_PATH = PROJECT_ROOT / "models" / "serious_model.joblib"
DRIFT_DIR = PROJECT_ROOT / "reports" / "drift"
TRAIN_END = 2023
SAMPLE_SIZE = 20_000
RETRAIN_IF_DRIFT_SHARE_ABOVE = 0.30

BASE_COLUMNS = ["age_years", "weight_kg", "n_reactions", "n_suspect_drugs", "n_concomitant_drugs",
                "is_us", "reporter_consumer", "reporter_physician", "reporter_lawyer"]
TOP_REACTION_COLUMNS = 15

DRIFT_METRIC = re.compile(
    r"ValueDrift\(column=(?P<column>.+?),method=(?P<method>.+?),threshold=(?P<threshold>[\d.]+)\)"
)


def drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, Report]:
    """Run Evidently's data drift preset and return one row per column."""
    result = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    rows = []
    for metric in result.dict()["metrics"]:
        match = DRIFT_METRIC.match(metric.get("metric_name", ""))
        if not match:
            continue
        score, threshold = float(metric["value"]), float(match["threshold"])
        uses_p_value = "p_value" in match["method"]
        rows.append({
            "column": match["column"],
            "method": match["method"],
            "score": score,
            "threshold": threshold,
            # p-value tests: drift when p is small. Distance methods: drift when distance is large.
            "drifted": score < threshold if uses_p_value else score >= threshold,
        })
    table = pd.DataFrame(rows).sort_values(["drifted", "column"], ascending=[False, True])
    return table.reset_index(drop=True), result


def monitored_columns(df: pd.DataFrame) -> list[str]:
    drugs = [c for c in df.columns if c.startswith("drug_")]
    reactions = df[[c for c in df.columns if c.startswith("rx_")]].mean().nlargest(TOP_REACTION_COLUMNS).index
    return [c for c in BASE_COLUMNS if c in df.columns] + drugs + list(reactions)


def run(current_year: int = 2025) -> dict:
    con = duckdb.connect(str(DATABASE), read_only=True)
    df = build_features(con, TRAIN_END)
    con.close()

    columns = monitored_columns(df)
    if Path(MODEL_PATH).exists():
        bundle = joblib.load(MODEL_PATH)
        X = df.reindex(columns=bundle["features"], fill_value=0)
        df["prediction"] = bundle["model"].predict_proba(X)[:, 1]
        columns.append("prediction")

    year = df["received_date"].dt.year
    reference = df.loc[year <= TRAIN_END, columns]
    current = df.loc[year == current_year, columns]
    reference = reference.sample(min(SAMPLE_SIZE, len(reference)), random_state=42)
    current = current.sample(min(SAMPLE_SIZE, len(current)), random_state=42)
    # Evidently cannot compare a column that is completely empty in either period.
    usable = [c for c in columns if reference[c].notna().any() and current[c].notna().any()]
    reference, current = reference[usable], current[usable]

    table, result = drift_table(reference, current)
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = DRIFT_DIR / f"drift_{current_year}_vs_2020-{TRAIN_END}.html"
    result.save_html(str(html_path))
    table.to_csv(DRIFT_DIR / f"drift_{current_year}_summary.csv", index=False)

    share = float(table["drifted"].mean())
    retrain = share > RETRAIN_IF_DRIFT_SHARE_ABOVE
    pd.set_option("display.width", 200)
    print(f"Reference: 2020-{TRAIN_END} ({len(reference):,} sampled reports). "
          f"Current: {current_year} ({len(current):,} sampled reports).")
    print(f"Drifted columns: {int(table['drifted'].sum())} of {len(table)} ({100 * share:.0f}%)\n")
    print(table.round(4).to_string(index=False))
    print(f"\nReport saved to {html_path.relative_to(PROJECT_ROOT)}")
    if retrain:
        print(f"More than {RETRAIN_IF_DRIFT_SHARE_ABOVE:.0%} of columns drifted: RETRAINING RECOMMENDED. "
              "Run `uv run python -m medsignal.ml.train` and review the new model in MLflow.")
    else:
        print("Drift is within the acceptable range. No retraining needed.")
    return {"drift_share": share, "retrain_recommended": retrain, "report": str(html_path)}


if __name__ == "__main__":
    run()
