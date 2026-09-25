"""Build a small, self-contained copy of the MedSignal app for Streamlit Community Cloud (free).

Creates deploy/demo/ with only what the live demo needs:
  - the dashboard pages, with the medsignal package placed next to them so Streamlit can import it
  - a slim copy of the warehouse: just the 5 tables the app reads
  - the FDA label chunks for label search
  - requirements.txt pinned to the same versions as uv.lock

The free host has too little memory for PyTorch, so the live demo's label search uses keyword
search (BM25) instead of hybrid + reranking. Everything else is the same code as the main repo.

Run:  uv run python scripts/deploy_demo.py
Then push deploy/demo to its own GitHub repo (medsignal-demo) and deploy it on share.streamlit.io.
Running it again rebuilds the files and keeps deploy/demo/.git, so you can commit and push updates.
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "demo"
TABLES = ["fct_report_drug", "fct_drug_reaction", "dim_drug", "mart_monthly_reports", "mart_signals"]
PACKAGES = ["streamlit", "duckdb", "pandas", "pyarrow", "plotly", "python-dotenv", "groq", "pydantic",
            "sqlglot", "rank-bm25", "requests", "tenacity", "numpy"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")

DEMO_README = """# MedSignal: live demo

Live demo of **[MedSignal](https://github.com/indlanamratha/medsignal)**: FDA adverse event reports (FAERS)
for 13 obesity and diabetes drug groups, 2020–2025, plus an AI agent ("Ask MedSignal") that answers drug
safety questions from FDA labels, report data, and signal statistics.

This repo is generated from the main project by `scripts/deploy_demo.py`. The full source code, pipeline,
models, and evaluation live in the main repo.

Note: to fit the free hosting tier, label search in this demo uses keyword search (BM25).
The main project uses hybrid search with a cross-encoder reranker.

FAERS reports show that a reaction was reported after a drug was taken. They do not prove the drug
caused it. This is a portfolio project, not a medical tool.
"""

STREAMLIT_CONFIG = """[server]
headless = true
fileWatcherType = "none"

[browser]
gatherUsageStats = false
"""


def clean_output() -> None:
    """Empty deploy/demo but keep its .git folder, so updates can be pushed."""
    OUT.mkdir(parents=True, exist_ok=True)
    for item in OUT.iterdir():
        if item.name == ".git":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def build_slim_database(source: Path, target: Path) -> None:
    """Copy only the tables the app reads into a new, smaller DuckDB file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with duckdb.connect(str(source), read_only=True) as src:
            for table in TABLES:
                src.execute(f"COPY (SELECT * FROM {table}) TO '{tmp}/{table}.parquet' (FORMAT parquet)")
        with duckdb.connect(str(target)) as dst:
            for table in TABLES:
                dst.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{tmp}/{table}.parquet')")
                rows = dst.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                print(f"  {table}: {rows:,} rows")
            dst.execute("CHECKPOINT")


def use_keyword_search(package_dir: Path) -> None:
    """Make BM25 the default search mode in the demo copy (no PyTorch needed)."""
    path = package_dir / "rag" / "retrieve.py"
    text = path.read_text()
    old = 'mode: str = "hybrid_rerank"'
    if text.count(old) != 1:
        raise SystemExit("Could not find the default search mode in retrieve.py; the demo was not built.")
    path.write_text(text.replace(old, 'mode: str = "bm25"'))


def pinned_requirements() -> str:
    """Versions of the demo's packages, taken from uv.lock so the demo matches the main project."""
    exported = subprocess.run(["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project"],
                              cwd=ROOT, check=True, capture_output=True, text=True).stdout
    wanted = {re.sub(r"[-_.]+", "-", p).lower() for p in PACKAGES}
    lines = []
    for line in exported.splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==", line)
        if match and re.sub(r"[-_.]+", "-", match.group(1)).lower() in wanted:
            lines.append(line.strip())
    found = {re.sub(r"[-_.]+", "-", re.match(r"^([A-Za-z0-9_.\-]+)", x).group(1)).lower() for x in lines}
    lines += sorted(wanted - found)  # anything not in the lock file: latest version
    return "\n".join(lines) + "\n"


def build() -> None:
    main_page = ROOT / "dashboard" / "Dashboard.py"
    if not main_page.exists():
        raise SystemExit("dashboard/Dashboard.py not found. Rename it first: git mv dashboard/app.py dashboard/Dashboard.py")

    clean_output()
    print("Copying the dashboard and code...")
    shutil.copytree(ROOT / "dashboard", OUT / "dashboard", ignore=IGNORE)
    # Streamlit adds the main page's folder to the import path, so the package goes next to it.
    shutil.copytree(ROOT / "src" / "medsignal", OUT / "dashboard" / "medsignal", ignore=IGNORE)
    use_keyword_search(OUT / "dashboard" / "medsignal")

    print("Copying FDA label chunks...")
    (OUT / "data" / "silver").mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "silver" / "label_chunks.parquet", OUT / "data" / "silver" / "label_chunks.parquet")

    print("Building a slim warehouse with only the tables the app uses...")
    build_slim_database(ROOT / "data" / "medsignal.duckdb", OUT / "data" / "medsignal.duckdb")

    print("Writing requirements and settings...")
    (OUT / "requirements.txt").write_text(pinned_requirements())
    (OUT / "README.md").write_text(DEMO_README)
    (OUT / ".gitignore").write_text(".env\n__pycache__/\n.streamlit/secrets.toml\n")
    (OUT / ".streamlit").mkdir()
    (OUT / ".streamlit" / "config.toml").write_text(STREAMLIT_CONFIG)

    size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file() and ".git" not in f.parts) / 1e6
    print(f"\nBuilt {OUT.relative_to(ROOT)} ({size_mb:.0f} MB). Next: push it to GitHub as medsignal-demo.")


if __name__ == "__main__":
    build()
