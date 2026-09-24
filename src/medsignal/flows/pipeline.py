"""End-to-end MedSignal pipeline: ingest (bronze) -> flatten (silver) -> dbt build (gold).

Every step is safe to re-run: finished downloads and staged files are skipped,
and dbt rebuilds the warehouse tables and re-runs all data quality tests.
"""
import subprocess
import sys
from pathlib import Path

from prefect import flow, task

from medsignal.config import END_YEAR, PROJECT_ROOT, START_YEAR
from medsignal.ingestion import faers, flatten

DBT_DIR = PROJECT_ROOT / "dbt"
DBT_EXECUTABLE = str(Path(sys.executable).parent / "dbt")


@task(retries=2, retry_delay_seconds=60)
def ingest_bronze(start_year: int, end_year: int) -> None:
    faers.run(None, start_year, end_year)


@task
def build_silver() -> None:
    flatten.run()


@task
def build_gold() -> None:
    subprocess.run([DBT_EXECUTABLE, "build"], cwd=DBT_DIR, check=True)


@flow(name="medsignal-pipeline", log_prints=True)
def pipeline(start_year: int = START_YEAR, end_year: int = END_YEAR) -> None:
    ingest_bronze(start_year, end_year)
    build_silver()
    build_gold()


if __name__ == "__main__":
    pipeline()
