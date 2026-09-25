"""Tools the MedSignal agent can call.

    search_labels  RAG over FDA drug labels (hybrid retrieval + reranking)
    query_faers    read-only SQL over the FAERS warehouse, with strict safety checks
    get_signal     PRR/ROR disproportionality statistics for a drug-reaction pair

SQL safety (defense in depth):
    1. The query must parse as exactly one SELECT statement.
    2. It may only read approved tables (CTEs are allowed).
    3. Table functions like read_csv or read_parquet are rejected.
    4. The database is opened read-only with file-system access disabled.
    5. Results are capped at MAX_ROWS rows.
"""
import duckdb
import sqlglot
from sqlglot import exp

from medsignal.config import DATA_DIR

DATABASE = DATA_DIR / "medsignal.duckdb"
MAX_ROWS = 50
ALLOWED_TABLES = {"fct_report_drug", "fct_drug_reaction", "mart_monthly_reports", "dim_drug", "mart_signals"}

SCHEMA = """Tables (DuckDB SQL). Count reports with count(DISTINCT report_id).
- fct_report_drug: one row per report x suspect study drug.
    report_id, drug_group, drug_class, received_date (DATE), is_serious (BOOL), died (BOOL),
    hospitalized (BOOL), age_group ('0-17','18-44','45-64','65+','Unknown'), sex ('M','F','U'),
    reporter_type_label ('Consumer','Physician','Pharmacist','Other health professional','Lawyer','Unknown'),
    country (2-letter code, e.g. 'US')
- fct_drug_reaction: one row per report x suspect study drug x reaction.
    report_id, drug_group, drug_class, reaction_pt (UPPERCASE MedDRA term, e.g. 'PANCREATITIS'),
    received_date, is_serious
- mart_monthly_reports: drug_group, drug_class, report_month (DATE), n_reports, n_serious, pct_serious
- mart_signals: drug_group, reaction_pt, a (reports with drug and reaction), prr, ror, ror_ci_low,
    ror_ci_high, chi2, is_signal (BOOL), high_confidence (BOOL)
drug_group values: semaglutide, tirzepatide, liraglutide, dulaglutide, exenatide, metformin,
    empagliflozin, dapagliflozin, sitagliptin, insulin glargine, phentermine, orlistat, naltrexone/bupropion.
Data covers reports received 2020-01-01 to 2025-12-31."""


class UnsafeQueryError(ValueError):
    pass


def check_sql(sql: str) -> str:
    """Validate a query and return it with a row limit. Raises UnsafeQueryError."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as error:
        raise UnsafeQueryError(f"Could not parse SQL: {error}") from error
    if len(statements) != 1:
        raise UnsafeQueryError("Exactly one SQL statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise UnsafeQueryError("Only SELECT queries are allowed.")

    cte_names = {cte.alias for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not table.name:
            raise UnsafeQueryError("Table functions such as read_csv or read_parquet are not allowed.")
        if table.name not in ALLOWED_TABLES | cte_names:
            raise UnsafeQueryError(f"Table '{table.name}' is not allowed. Use: {sorted(ALLOWED_TABLES)}")

    if not tree.args.get("limit"):
        tree = tree.limit(MAX_ROWS)
    return tree.sql(dialect="duckdb")


def connect(database=DATABASE) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(database), read_only=True, config={"enable_external_access": False})


def query_faers(sql: str, database=DATABASE) -> dict:
    try:
        safe_sql = check_sql(sql)
        with connect(database) as con:
            df = con.execute(safe_sql).df().head(MAX_ROWS)
    except UnsafeQueryError as error:
        return {"error": str(error)}
    except duckdb.Error as error:
        return {"error": f"SQL error: {error}"}
    return {"sql": safe_sql, "rows": df.astype(str).to_dict("records"), "row_count": len(df)}


def get_signal(drug_group: str, reaction: str, database=DATABASE) -> dict:
    with connect(database) as con:
        row = con.execute("""
            select drug_group, reaction_pt, a as reports, round(prr, 2) as prr, round(ror, 2) as ror,
                   round(ror_ci_low, 2) as ror_ci_low, round(ror_ci_high, 2) as ror_ci_high,
                   round(chi2, 1) as chi2, is_signal, high_confidence
            from mart_signals
            where drug_group = ? and reaction_pt = ?
        """, [drug_group.strip().lower(), reaction.strip().upper()]).df()
    if row.empty:
        return {"drug_group": drug_group, "reaction_pt": reaction.upper(),
                "result": "Not tested: fewer than 3 reports, or the term was excluded as non-medical."}
    return row.iloc[0].to_dict()


def search_labels(retriever, question: str, drug_group: str | None = None, k: int = 4) -> dict:
    chunks = retriever.retrieve(question, drug_group, k=k)
    return {"passages": [{"id": f"L{i}", "brand": c["brand_name"], "section": c["section"],
                          "text": c["text"][:1200]} for i, c in enumerate(chunks, start=1)]}


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_labels",
        "description": "Search official FDA drug labels (warnings, adverse reactions, dosing, "
                       "contraindications). Use for what the label says about a drug.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "What to look up"},
            "drug_group": {"type": "string", "description": "Optional drug_group, e.g. semaglutide"}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "query_faers",
        "description": "Run one read-only SELECT query on the FAERS adverse event warehouse. "
                       "Use for counts, trends, and breakdowns of reports.\n" + SCHEMA,
        "parameters": {"type": "object", "properties": {
            "sql": {"type": "string", "description": "A single DuckDB SELECT statement"}},
            "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "get_signal",
        "description": "Get disproportionality statistics (PRR, ROR with 95% CI, chi-square) for one "
                       "drug and one reaction, and whether it is flagged as a safety signal.",
        "parameters": {"type": "object", "properties": {
            "drug_group": {"type": "string"},
            "reaction": {"type": "string", "description": "MedDRA preferred term, e.g. PANCREATITIS"}},
            "required": ["drug_group", "reaction"]}}},
]
