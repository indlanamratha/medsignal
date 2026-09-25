"""Download FDA drug labels from openFDA and split them into retrievable chunks.

Labels are the official prescribing information for each medicine. Each label is
already divided into sections (Boxed Warning, Warnings and Precautions, Adverse
Reactions, ...), so chunks never mix two sections, and every chunk keeps its
drug, brand, section, and label ID so answers can cite exactly where they came from.
"""
import json
import re

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from medsignal.config import BRONZE_DIR, DRUG_QUERIES, OPENFDA_KEY, SILVER_DIR

LABEL_URL = "https://api.fda.gov/drug/label.json"
LABELS_DIR = BRONZE_DIR / "labels"
CHUNKS_PATH = SILVER_DIR / "label_chunks.parquet"
MAX_LABELS_PER_DRUG = 4
CHUNK_WORDS = 300
OVERLAP_WORDS = 50

# Warehouse names for each ingestion query key.
DRUG_GROUPS = {key: ("naltrexone/bupropion" if key == "naltrexone_bupropion" else key) for key in DRUG_QUERIES}

SECTIONS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "overdosage": "Overdosage",
    "mechanism_of_action": "Mechanism of Action",
    "patient_counseling_information": "Patient Counseling Information",
}


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
def _get(params: dict) -> list[dict]:
    response = requests.get(LABEL_URL, params=params, timeout=60)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return response.json()["results"]


def select_labels(labels: list[dict], allow_combinations: bool) -> list[dict]:
    """Keep the most recent label per brand, skipping combination products unless wanted."""
    newest_by_brand: dict[str, dict] = {}
    for label in sorted(labels, key=lambda x: x.get("effective_time", ""), reverse=True):
        openfda = label.get("openfda") or {}
        generic = (openfda.get("generic_name") or [""])[0].upper()
        brand = (openfda.get("brand_name") or [generic])[0].upper()
        if not brand or (" AND " in generic and not allow_combinations):
            continue
        newest_by_brand.setdefault(brand, label)
    return list(newest_by_brand.values())[:MAX_LABELS_PER_DRUG]


def download_labels() -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for key, search in DRUG_QUERIES.items():
        label_search = search.replace("patient.drug.", "")
        labels = _get({"search": label_search, "limit": 100, "sort": "effective_time:desc",
                       "api_key": OPENFDA_KEY})
        chosen = select_labels(labels, allow_combinations=(key == "naltrexone_bupropion"))
        (LABELS_DIR / f"{key.replace(' ', '_')}.json").write_text(json.dumps(chosen))
        brands = [(x.get("openfda") or {}).get("brand_name", ["?"])[0] for x in chosen]
        print(f"{DRUG_GROUPS[key]:<22} {len(chosen)} labels: {', '.join(brands)}")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    step = size - overlap
    return [" ".join(words[start:start + size]) for start in range(0, len(words) - overlap, step)]


def chunk_label(label: dict, drug_group: str) -> list[dict]:
    openfda = label.get("openfda") or {}
    generic = (openfda.get("generic_name") or [""])[0].title()
    brand = (openfda.get("brand_name") or [generic])[0].title()
    set_id = label.get("set_id", label.get("id", "unknown"))
    effective = label.get("effective_time", "")
    rows = []
    for field, title in SECTIONS.items():
        text = clean(" ".join(label.get(field) or []))
        for i, piece in enumerate(split_words(text)):
            rows.append({
                "chunk_id": f"{set_id}:{field}:{i}",
                "drug_group": drug_group,
                "brand_name": brand,
                "generic_name": generic,
                "section": title,
                "set_id": set_id,
                "effective_date": effective,
                "text": piece,
                # A short header gives every chunk context, which improves retrieval.
                "context_text": f"{brand} ({generic}) - {title}: {piece}",
            })
    return rows


def build_chunks() -> pd.DataFrame:
    rows = []
    for key in DRUG_QUERIES:
        path = LABELS_DIR / f"{key.replace(' ', '_')}.json"
        for label in json.loads(path.read_text()):
            rows.extend(chunk_label(label, DRUG_GROUPS[key]))
    chunks = pd.DataFrame(rows).drop_duplicates("chunk_id")
    chunks.to_parquet(CHUNKS_PATH, index=False)
    return chunks


if __name__ == "__main__":
    download_labels()
    chunks = build_chunks()
    print(f"\n{len(chunks):,} chunks from {chunks['set_id'].nunique()} labels saved to {CHUNKS_PATH.name}")
    print(chunks.groupby("section").size().sort_values(ascending=False).to_string())
