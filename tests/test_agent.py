"""Agent and SQL-safety tests. They use a tiny test database and a fake LLM (no API key needed)."""
import json
from types import SimpleNamespace

import duckdb
import pytest

from medsignal.agent import tools
from medsignal.agent.agent import MedSignalAgent


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(path))
    con.execute("""create table fct_report_drug as select * from (values
        ('r1', 'semaglutide', date '2025-01-05', true), ('r2', 'semaglutide', date '2025-02-01', false),
        ('r3', 'tirzepatide', date '2025-03-01', false)) t(report_id, drug_group, received_date, is_serious)""")
    con.execute("""create table mart_signals as select * from (values
        ('semaglutide', 'PANCREATITIS', 1052, 2.85, 2.90, 2.70, 3.10, 919.8, true, true))
        t(drug_group, reaction_pt, a, prr, ror, ror_ci_low, ror_ci_high, chi2, is_signal, high_confidence)""")
    con.execute("create table secret_table as select 1 as x")
    con.close()
    return path


# ---------- SQL safety ----------

def test_select_on_allowed_table_runs_with_row_limit(db):
    out = tools.query_faers("select drug_group, count(distinct report_id) as n from fct_report_drug "
                            "group by drug_group order by n desc", db)
    assert out["rows"][0] == {"drug_group": "semaglutide", "n": "2"}
    assert "LIMIT 50" in out["sql"].upper()


def test_cte_is_allowed(db):
    sql = "with s as (select * from fct_report_drug where is_serious) select count(*) as n from s"
    out = tools.query_faers(sql, db)
    assert out["rows"][0]["n"] == "1"


@pytest.mark.parametrize("sql", [
    "delete from fct_report_drug",
    "drop table fct_report_drug",
    "select * from secret_table",
    "select * from read_csv('/etc/passwd')",
    "select 1; select 2",
    "attach 'other.db'",
])
def test_unsafe_queries_are_rejected(db, sql):
    assert "error" in tools.query_faers(sql, db)


def test_get_signal_found_and_not_found(db):
    assert tools.get_signal("Semaglutide", "pancreatitis", db)["is_signal"]
    assert "Not tested" in tools.get_signal("semaglutide", "HICCUPS", db)["result"]


# ---------- Agent loop ----------

def tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeLLM:
    def __init__(self, *messages):
        self.messages, self.requests = list(messages), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.messages.pop(0))])


class FakeRetriever:
    def retrieve(self, question, drug=None, k=4, mode="hybrid_rerank"):
        return [{"brand_name": "Ozempic", "section": "Warnings and Precautions",
                 "text": "Acute pancreatitis has been observed."}]


def test_agent_calls_tools_then_answers(db):
    llm = FakeLLM(
        SimpleNamespace(content="", tool_calls=[
            tool_call("c1", "search_labels", {"question": "pancreatitis", "drug_group": "semaglutide"}),
            tool_call("c2", "get_signal", {"drug_group": "semaglutide", "reaction": "PANCREATITIS"})]),
        SimpleNamespace(content="Yes. The label warns about pancreatitis [L1], and it is a FAERS signal "
                                "(PRR 2.85). Reports do not prove causation.", tool_calls=None),
    )
    agent = MedSignalAgent(client=llm, retriever=FakeRetriever(), model="test", database=db)
    result = agent.ask("Is pancreatitis a concern with semaglutide?")

    assert [s["tool"] for s in result.steps] == ["search_labels", "get_signal"]
    assert result.steps[0]["output"]["passages"][0]["id"] == "L1"
    assert result.steps[1]["output"]["prr"] == 2.85
    assert "[L1]" in result.answer
    tool_messages = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2          # both results were sent back to the model


def test_agent_reports_bad_sql_back_to_the_model(db):
    llm = FakeLLM(
        SimpleNamespace(content="", tool_calls=[tool_call("c1", "query_faers", {"sql": "delete from dim_drug"})]),
        SimpleNamespace(content="I can only run read-only queries.", tool_calls=None),
    )
    result = MedSignalAgent(client=llm, retriever=FakeRetriever(), model="test", database=db).ask("Delete data")
    assert "error" in result.steps[0]["output"]
