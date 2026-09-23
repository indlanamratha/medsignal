"""Explore the openFDA adverse event API before building the pipeline."""
import json

import requests

from medsignal.config import DATA_DIR, OPENFDA_KEY

URL = "https://api.fda.gov/drug/event.json"

DRUGS = [
    "semaglutide", "tirzepatide", "liraglutide", "dulaglutide", "exenatide",
    "metformin", "empagliflozin", "dapagliflozin", "sitagliptin", "insulin glargine",
    "phentermine", "orlistat", "naltrexone",
]


def count_reports(drug: str) -> int:
    params = {
        "search": f'patient.drug.openfda.generic_name:"{drug}"',
        "limit": 1,
        "api_key": OPENFDA_KEY,
    }
    response = requests.get(URL, params=params, timeout=60)
    if response.status_code == 404:
        return 0
    response.raise_for_status()
    return response.json()["meta"]["results"]["total"]


def show_sample_report() -> None:
    params = {
        "search": 'patient.drug.openfda.generic_name:"semaglutide"',
        "limit": 1,
        "api_key": OPENFDA_KEY,
    }
    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    report = response.json()["results"][0]

    print("\n--- One semaglutide report ---")
    print("Report ID:  ", report["safetyreportid"])
    print("Received:   ", report["receivedate"])
    print("Serious:    ", report.get("serious"), "(1 = yes, 2 = no)")
    print("Drugs:      ", [d.get("medicinalproduct") for d in report["patient"]["drug"]])
    print("Reactions:  ", [rx.get("reactionmeddrapt") for rx in report["patient"]["reaction"]])

    DATA_DIR.mkdir(exist_ok=True)
    sample_path = DATA_DIR / "sample_report.json"
    sample_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved to {sample_path}")


if __name__ == "__main__":
    print("Total reports in FAERS per drug:")
    for drug in DRUGS:
        print(f"  {drug:<18} {count_reports(drug):>10,}")
    show_sample_report()