import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"

OPENFDA_KEY = os.getenv("OPENFDA_KEY")
if not OPENFDA_KEY:
    raise RuntimeError("OPENFDA_KEY is missing. Check your .env file.")


    # ---- Ingestion scope ----
START_YEAR = 2020
END_YEAR = 2025


def _generic(name: str) -> str:
    return f'patient.drug.openfda.generic_name:"{name}"'


DRUG_QUERIES = {
    # GLP-1 / GIP (obesity focus)
    "semaglutide": _generic("semaglutide"),
    "tirzepatide": _generic("tirzepatide"),
    "liraglutide": _generic("liraglutide"),
    "dulaglutide": _generic("dulaglutide"),
    "exenatide": _generic("exenatide"),
    # Other diabetes drugs (comparators)
    "metformin": _generic("metformin"),
    "empagliflozin": _generic("empagliflozin"),
    "dapagliflozin": _generic("dapagliflozin"),
    "sitagliptin": _generic("sitagliptin"),
    "insulin glargine": _generic("insulin glargine"),
    # Other weight-loss drugs
    "phentermine": _generic("phentermine"),
    "orlistat": _generic("orlistat"),
    # Contrave = naltrexone + bupropion in the same report
    "naltrexone_bupropion": f'{_generic("naltrexone")} AND {_generic("bupropion")}',
}