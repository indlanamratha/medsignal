"""Grounded question answering over FDA drug labels.

Retrieves label passages, then asks an LLM (via Groq) to answer ONLY from those
passages, cite them, and say so when the labels don't contain the answer.

Guardrails:
  1. If retrieval finds nothing relevant, the system declines WITHOUT calling the LLM.
  2. The LLM must return JSON that matches a schema; invalid output is retried once.
  3. Citations must point at passages that were actually retrieved.

Try it:  uv run python -m medsignal.rag.answer "What is the boxed warning for Wegovy?"
"""
import argparse
import json
import os

from pydantic import BaseModel, Field, ValidationError

from medsignal.rag.retrieve import LabelRetriever

DEFAULT_MODEL = "openai/gpt-oss-120b"   # override with GROQ_MODEL in .env
MIN_RERANK_SCORE = 0.0        # cross-encoder scores below this mean "not relevant"
NOT_FOUND = "The FDA labels in this project don't contain information to answer this question."
DISCLAIMER = "This is information from FDA drug labels, not medical advice. Ask a doctor or pharmacist."

SYSTEM_PROMPT = """You answer questions about medicines using ONLY the numbered FDA label passages provided.

Rules:
- Use only facts stated in the passages. Never use outside knowledge.
- Cite every claim with the passage number in square brackets, like [1] or [2][3].
- If the passages do not contain the answer, set found_in_sources to false and say the labels don't cover it.
- Do not give personal medical advice.
- Be concise: 2 to 5 sentences.

Reply with JSON only, in exactly this shape:
{"answer": "<answer with [n] citations>", "citations": [<passage numbers used>], "found_in_sources": <true or false>}"""


class LLMAnswer(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)
    found_in_sources: bool


class Citation(BaseModel):
    number: int
    brand_name: str
    section: str
    set_id: str
    chunk_id: str


class GroundedAnswer(BaseModel):
    question: str
    drug: str | None
    answer: str
    found_in_sources: bool
    citations: list[Citation]
    disclaimer: str = DISCLAIMER


def format_passages(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {c['brand_name']} label, {c['section']}:\n{c['text']}"
                       for i, c in enumerate(chunks, start=1))


def groq_client():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return Groq(api_key=api_key)


class LabelQA:
    def __init__(self, retriever: LabelRetriever | None = None, client=None, model: str | None = None):
        self.retriever = retriever or LabelRetriever()
        self._client = client
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    @property
    def client(self):
        if self._client is None:
            self._client = groq_client()
        return self._client

    def _ask_llm(self, question: str, passages: str) -> LLMAnswer:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question}"}]
        last_error = None
        for _ in range(2):  # one retry if the JSON is invalid
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0,
                response_format={"type": "json_object"})
            try:
                return LLMAnswer.model_validate(json.loads(response.choices[0].message.content))
            except (json.JSONDecodeError, ValidationError) as error:
                last_error = error
        raise RuntimeError(f"LLM returned invalid JSON twice: {last_error}")

    def answer(self, question: str, drug: str | None = None, k: int = 5,
               mode: str = "hybrid_rerank") -> GroundedAnswer:
        drug = drug or self.retriever.detect(question)
        chunks = self.retriever.retrieve(question, drug, k=k, mode=mode)
        if mode == "hybrid_rerank":
            chunks = [c for c in chunks if c.get("rerank_score", 0) >= MIN_RERANK_SCORE]
        if not chunks:
            return GroundedAnswer(question=question, drug=drug, answer=NOT_FOUND,
                                  found_in_sources=False, citations=[])

        result = self._ask_llm(question, format_passages(chunks))
        valid = sorted({n for n in result.citations if 1 <= n <= len(chunks)})
        citations = [Citation(number=n, **{f: chunks[n - 1][f] for f in
                     ("brand_name", "section", "set_id", "chunk_id")}) for n in valid]
        found = result.found_in_sources and bool(citations)
        return GroundedAnswer(question=question, drug=drug,
                              answer=result.answer if found else NOT_FOUND,
                              found_in_sources=found, citations=citations)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask a question about FDA drug labels.")
    parser.add_argument("question")
    parser.add_argument("--drug")
    args = parser.parse_args()

    result = LabelQA().answer(args.question, args.drug)
    print(f"\nQ: {result.question}   (drug: {result.drug or 'not detected'})\n")
    print(f"A: {result.answer}\n")
    for c in result.citations:
        print(f"  [{c.number}] {c.brand_name} label, {c.section}")
    print(f"\n{result.disclaimer}")


if __name__ == "__main__":
    main()
