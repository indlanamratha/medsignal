"""Evaluate MedSignal's AI layer on a hand-written 50-question test set (eval/questions.jsonl).

Three parts:
  retrieval  32 label questions x 4 search modes, each with and without query expansion
             and near-duplicate removal: does the top 5 contain the right drug and
             label section? Reports hit rate and MRR. No LLM calls.
  qa         Grounded answering (hybrid + rerank):
               - fact recall: required facts present in the answer
               - faithfulness: share of the answer's claims supported by the passages,
                 judged by a second LLM (LLM-as-judge)
               - refusals: 10 questions the labels can't answer must be declined
  agent      8 data questions: the agent's answer must contain the number (or drug) that
             a reference SQL query returns.

Results are saved after every question, so if the free API tier hits a rate limit you can
simply run the command again and it continues where it stopped.

Run:  uv run python -m medsignal.eval.run_eval            (everything)
      uv run python -m medsignal.eval.run_eval --part retrieval
      uv run python -m medsignal.eval.run_eval --fresh    (ignore saved results)
"""
import argparse
import json
import os
import re
from types import SimpleNamespace

import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from medsignal.config import PROJECT_ROOT

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")
K = 5

JUDGE_PROMPT = """You check whether an answer is supported by source passages.

Split the ANSWER into its individual factual claims. Ignore disclaimers, advice to consult a doctor,
and statements that the sources don't cover something. For each claim, decide whether the PASSAGES
directly support it. A claim with a number or detail that does not appear in the passages is NOT supported.

Reply with JSON only: {"claims": [{"claim": "<text>", "supported": true or false}]}"""


# ---------------- small, testable helpers ----------------

def normalize(text: str) -> str:
    """Lowercase and turn unicode dashes and spaces into plain ones, so 'C‑cell' matches 'c-cell'."""
    text = re.sub(r"[‐-―−]", "-", text or "")
    return re.sub(r"[  ]", " ", text).lower()


def fact_recall(answer: str, facts: list[str]) -> float:
    """Share of required facts found. Each fact may list alternatives separated by '|'."""
    if not facts:
        return 1.0
    text = normalize(answer)
    return sum(any(alt in text for alt in fact.split("|")) for fact in facts) / len(facts)


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")]


def matches_truth(answer: str, truth) -> bool:
    """True if the answer contains the reference value: counts exactly, decimals within rounding."""
    if isinstance(truth, str):
        return normalize(truth) in normalize(answer)
    if isinstance(truth, int):
        return truth in numbers_in(answer)
    return any(abs(n - float(truth)) <= 0.051 for n in numbers_in(answer))


def hit_rank(chunks: list[dict], drug: str, sections: list[str]) -> int | None:
    """1-based rank of the first chunk from the right drug and an expected section, else None."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["drug_group"] == drug and (not sections or chunk["section"] in sections):
            return rank
    return None


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS_PATH.read_text().splitlines() if line.strip()]


# ---------------- API client with patient retries ----------------

def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return "RateLimit" in type(error).__name__ or status in (429, 500, 502, 503)


class RetryingClient:
    """Wraps the Groq client so rate-limit errors wait and retry instead of failing the run."""

    def __init__(self, client):
        self._client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(8),
           wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True)
    def _create(self, **kwargs):
        return self._client.chat.completions.create(**kwargs)


def judge_faithfulness(client, answer: str, passages: list[str]) -> float | None:
    content = "PASSAGES:\n" + "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1)) + f"\n\nANSWER:\n{answer}"
    response = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": content}])
    claims = json.loads(response.choices[0].message.content).get("claims", [])
    return sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else None


# ---------------- results cache ----------------

def load_done(name: str) -> dict[str, dict]:
    path = RESULTS_DIR / f"{name}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def save_row(name: str, row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / f"{name}.jsonl").open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


# ---------------- the three evaluations ----------------

def eval_retrieval(questions: list[dict], retriever) -> pd.DataFrame:
    """Each search mode, first plain, then with query expansion and near-duplicate removal."""
    configs = [(mode, False) for mode in MODES] + [(mode, True) for mode in MODES]
    rows = []
    for q in (q for q in questions if q["type"] == "label"):
        for mode, improved in configs:
            chunks = retriever.retrieve(q["question"], k=K, mode=mode, expand=improved, dedupe=improved)
            name = f"{mode} + expansion + dedupe" if improved else mode
            rows.append({"id": q["id"], "mode": name, "rank": hit_rank(chunks, q["drug"], q["sections"])})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "retrieval.csv", index=False)
    order = list(dict.fromkeys(df["mode"]))
    return (df.assign(hit=df["rank"].notna(), rr=1 / df["rank"])
              .groupby("mode").agg(hit_rate_at_5=("hit", "mean"), mrr_at_5=("rr", lambda s: s.fillna(0).mean()))
              .reindex(order))


def eval_qa(questions: list[dict], qa, client) -> pd.DataFrame:
    done = load_done("qa")
    for q in (q for q in questions if q["type"] in ("label", "unanswerable")):
        if q["id"] in done:
            continue
        result = qa.answer(q["question"])
        row = {"id": q["id"], "type": q["type"], "answered": result.found_in_sources,
               "n_citations": len(result.citations), "answer": result.answer}
        if q["type"] == "label":
            row["fact_recall"] = fact_recall(result.answer, q["facts"]) if result.found_in_sources else 0.0
            row["faithfulness"] = (judge_faithfulness(client, result.answer, result.passages)
                                   if result.found_in_sources else None)
        save_row("qa", row)
        done[q["id"]] = row
        print(f"  {q['id']}: answered={row['answered']} "
              + (f"facts={row['fact_recall']:.2f} faithful={row['faithfulness']}" if q["type"] == "label" else ""))
    return pd.DataFrame(done.values())


def eval_agent(questions: list[dict], agent, con) -> pd.DataFrame:
    done = load_done("agent")
    for q in (q for q in questions if q["type"] == "data"):
        if q["id"] in done:
            continue
        truth = con.execute(q["truth_sql"]).fetchone()[0]
        result = agent.ask(q["question"])
        row = {"id": q["id"], "truth": truth, "correct": matches_truth(result.answer, truth),
               "tools": [s["tool"] for s in result.steps], "answer": result.answer}
        save_row("agent", row)
        done[q["id"]] = row
        print(f"  {q['id']}: truth={truth} correct={row['correct']} tools={row['tools']}")
    return pd.DataFrame(done.values())


def summarize(retrieval: pd.DataFrame | None, qa: pd.DataFrame | None, agent: pd.DataFrame | None) -> str:
    lines = ["# MedSignal AI Evaluation\n"]
    if retrieval is not None:
        lines += ["## Retrieval (32 label questions, top 5)\n", retrieval.round(3).to_markdown(), ""]
    if qa is not None and not qa.empty:
        label, unans = qa[qa["type"] == "label"], qa[qa["type"] == "unanswerable"]
        metrics = {
            "Label questions answered": f"{label['answered'].mean():.0%} of {len(label)}",
            "Fact recall (answered questions)": f"{label.loc[label['answered'], 'fact_recall'].mean():.2f}",
            "Faithfulness (LLM judge)": f"{label['faithfulness'].dropna().astype(float).mean():.2f}",
            "Unanswerable questions correctly declined": f"{(~unans['answered']).mean():.0%} of {len(unans)}",
        }
        lines += ["## Grounded answering (hybrid + rerank)\n",
                  pd.Series(metrics, name="value").to_frame().to_markdown(), ""]
    if agent is not None and not agent.empty:
        lines += ["## Agent data questions\n",
                  f"Correct: {agent['correct'].sum()} of {len(agent)} ({agent['correct'].mean():.0%})\n"]
    return "\n".join(lines)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate retrieval, grounded answering, and the agent.")
    parser.add_argument("--part", choices=["retrieval", "qa", "agent", "all"], default="all")
    parser.add_argument("--fresh", action="store_true", help="Delete saved results and start over")
    args = parser.parse_args()

    from medsignal.agent import tools
    from medsignal.agent.agent import MedSignalAgent
    from medsignal.rag.answer import LabelQA, groq_client
    from medsignal.rag.retrieve import LabelRetriever

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for name in ("qa", "agent"):
            (RESULTS_DIR / f"{name}.jsonl").unlink(missing_ok=True)

    questions = load_questions()
    retriever = LabelRetriever()
    retrieval = qa = agent = None

    if args.part in ("retrieval", "all"):
        print("Evaluating retrieval...")
        retrieval = eval_retrieval(questions, retriever)
    if args.part in ("qa", "agent", "all"):
        client = RetryingClient(groq_client())
        if args.part in ("qa", "all"):
            print("Evaluating grounded answering...")
            qa = eval_qa(questions, LabelQA(retriever=retriever, client=client), client)
        if args.part in ("agent", "all"):
            print("Evaluating the agent...")
            with tools.connect() as con:
                agent = eval_agent(questions, MedSignalAgent(client=client, retriever=retriever), con)

    report = summarize(retrieval, qa, agent)
    (RESULTS_DIR / "summary.md").write_text(report)
    print("\n" + report + f"\nSaved to {RESULTS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
