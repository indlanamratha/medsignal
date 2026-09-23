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