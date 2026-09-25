"""MedSignal agent: answers drug safety questions by combining three tools.

The LLM decides which tools to call (label search, SQL over FAERS, signal statistics),
reads the results, and writes one answer that cites label passages and quotes real
numbers. The loop is written by hand (no framework) so every step is visible.

Try it:  uv run python -m medsignal.agent.agent "Is pancreatitis a real concern with semaglutide?"
"""
import argparse
import json
import time
from dataclasses import dataclass, field

from medsignal.agent import tools
from medsignal.rag.answer import DISCLAIMER, groq_client

MAX_STEPS = 6

SYSTEM_PROMPT = """You are MedSignal, an assistant for drug safety questions about 13 obesity and diabetes drugs.

You have three tools:
- search_labels: what the official FDA label says.
- query_faers: counts and trends from FDA adverse event reports (FAERS), 2020-2025.
- get_signal: PRR/ROR statistics showing whether a reaction is reported disproportionately.

Rules:
- Use tools to get facts. Never invent numbers, label text, or statistics.
- Cite label passages by their id in square brackets, e.g. [L1].
- When you use FAERS numbers or signal statistics, say where they came from.
- FAERS reports show that a reaction was reported after taking a drug; they do not prove
  the drug caused it. Say this whenever you discuss report counts or signals.
- If the tools cannot answer the question, say so plainly.
- Do not give personal medical advice.
- Keep the final answer under 150 words."""


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[dict] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


class MedSignalAgent:
    def __init__(self, client=None, retriever=None, model: str | None = None, database=tools.DATABASE):
        self._client, self._retriever, self.model_override, self.database = client, retriever, model, database

    @property
    def client(self):
        if self._client is None:
            self._client = groq_client()
        return self._client

    @property
    def retriever(self):
        if self._retriever is None:
            from medsignal.rag.retrieve import LabelRetriever
            self._retriever = LabelRetriever()
        return self._retriever

    @property
    def model(self) -> str:
        import os

        from medsignal.rag.answer import DEFAULT_MODEL
        return self.model_override or os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    def run_tool(self, name: str, args: dict) -> dict:
        if name == "search_labels":
            return tools.search_labels(self.retriever, args["question"], args.get("drug_group"))
        if name == "query_faers":
            return tools.query_faers(args["sql"], self.database)
        if name == "get_signal":
            return tools.get_signal(args["drug_group"], args["reaction"], self.database)
        return {"error": f"Unknown tool {name}"}

    def ask(self, question: str) -> AgentResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        result = AgentResult(question=question, answer="")
        for _ in range(MAX_STEPS):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools.TOOL_SPECS,
                tool_choice="auto", temperature=0)
            message = response.choices[0].message
            calls = message.tool_calls or []
            if not calls:
                result.answer = message.content or ""
                return result

            messages.append({"role": "assistant", "content": message.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name,
                                                          "arguments": c.function.arguments}} for c in calls]})
            for call in calls:
                start = time.perf_counter()
                try:
                    args = json.loads(call.function.arguments or "{}")
                    output = self.run_tool(call.function.name, args)
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    args, output = {}, {"error": f"Bad tool arguments: {error}"}
                result.steps.append({"tool": call.function.name, "args": args, "output": output,
                                     "ms": round((time.perf_counter() - start) * 1000)})
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(output, default=str)[:6000]})
        result.answer = "I couldn't finish within the step limit. Try a more specific question."
        return result


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask the MedSignal agent a drug safety question.")
    parser.add_argument("question")
    args = parser.parse_args()

    result = MedSignalAgent().ask(args.question)
    print(f"\nQ: {result.question}\n")
    for i, step in enumerate(result.steps, start=1):
        print(f"Step {i}: {step['tool']}({json.dumps(step['args'])[:200]})  [{step['ms']} ms]")
        if "sql" in step["output"]:
            print(f"        SQL run: {step['output']['sql'][:200]}")
        if "error" in step["output"]:
            print(f"        error: {step['output']['error']}")
    print(f"\nA: {result.answer}\n\n{result.disclaimer}")


if __name__ == "__main__":
    main()
