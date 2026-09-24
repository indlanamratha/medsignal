"""Run every SQL file in analytics/queries against the warehouse.

Prints each result and saves it as a CSV in analytics/results/.
"""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "analytics" / "queries"
RESULTS = ROOT / "analytics" / "results"
DATABASE = ROOT / "data" / "medsignal.duckdb"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DATABASE), read_only=True)
    for path in sorted(QUERIES.glob("*.sql")):
        df = con.sql(path.read_text()).df()
        df.to_csv(RESULTS / f"{path.stem}.csv", index=False)
        print(f"\n===== {path.stem}  ({len(df)} rows) =====")
        print(df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
