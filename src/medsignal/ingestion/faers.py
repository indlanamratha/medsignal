"""Download FAERS adverse event reports from openFDA into the bronze layer.

Each drug-month is saved as its own gzipped JSON file:
    data/bronze/faers/<drug>/<YYYY-MM>.json.gz

Re-running is safe: months that already exist are skipped.
"""
import argparse
import calendar
import gzip
import json
from pathlib import Path

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from medsignal.config import BRONZE_DIR, DRUG_QUERIES, END_YEAR, OPENFDA_KEY, START_YEAR

URL = "https://api.fda.gov/drug/event.json"
PAGE_SIZE = 1000        # max records per request
MAX_RECORDS = 26_000    # API can't page past skip=25,000
FAERS_DIR = BRONZE_DIR / "faers"


class TooManyResults(Exception):
    """A single window has more records than the API can page through."""


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
)
def _get(params: dict) -> dict:
    response = requests.get(URL, params=params, timeout=120)
    if response.status_code == 404:  # openFDA's way of saying "no results"
        return {"meta": {"results": {"total": 0}}, "results": []}
    response.raise_for_status()
    return response.json()


def fetch_window(search: str, start: str, end: str) -> list[dict]:
    """Fetch every report matching `search` received between start and end."""
    query = f"({search}) AND receivedate:[{start} TO {end}]"
    records: list[dict] = []
    skip = 0
    while True:
        data = _get({"search": query, "limit": PAGE_SIZE, "skip": skip,
                     "api_key": OPENFDA_KEY})
        total = data["meta"]["results"]["total"]
        if total > MAX_RECORDS:
            raise TooManyResults(f"{total:,} records between {start} and {end}")
        records.extend(data["results"])
        skip += PAGE_SIZE
        if skip >= total:
            return records


def month_bounds(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}{month:02d}01", f"{year}{month:02d}{last_day}"


def output_path(drug: str, year: int, month: int) -> Path:
    return FAERS_DIR / drug.replace(" ", "_") / f"{year}-{month:02d}.json.gz"


def ingest_month(drug: str, search: str, year: int, month: int) -> int | None:
    """Download one drug-month. Returns record count, or None if already done."""
    path = output_path(drug, year, month)
    if path.exists():
        return None

    start, end = month_bounds(year, month)
    records = fetch_window(search, start, end)

    # Write to a temp file first, then rename. If the script is interrupted,
    # we never leave a half-written file that looks complete.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(records, f)
    tmp_path.rename(path)
    return len(records)


def run(drugs: list[str] | None, start_year: int, end_year: int) -> None:
    drugs = drugs or list(DRUG_QUERIES)
    grand_total = 0
    for drug in drugs:
        search = DRUG_QUERIES[drug]
        drug_total = 0
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                count = ingest_month(drug, search, year, month)
                if count is None:
                    continue
                drug_total += count
                print(f"{drug:<22} {year}-{month:02d}  {count:>7,} reports")
        print(f"== {drug}: {drug_total:,} new reports\n")
        grand_total += drug_total
    print(f"Done. {grand_total:,} new reports downloaded in total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FAERS reports from openFDA.")
    parser.add_argument("--drug", action="append", choices=list(DRUG_QUERIES),
                        help="Only this drug (repeatable). Default: all drugs.")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    args = parser.parse_args()
    run(args.drug, args.start_year, args.end_year)