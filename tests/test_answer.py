"""Tests for grounded answering, using fake retrieval and a fake LLM (no API key needed)."""
import json
from types import SimpleNamespace

from medsignal.rag.answer import NOT_FOUND, LabelQA

CHUNKS = [
    {"brand_name": "Wegovy", "section": "Boxed Warning", "set_id": "s1", "chunk_id": "s1:boxed:0",
     "text": "WARNING: RISK OF THYROID C-CELL TUMORS", "rerank_score": 6.5},
    {"brand_name": "Wegovy", "section": "Warnings and Precautions", "set_id": "s1", "chunk_id": "s1:warn:0",
     "text": "Acute pancreatitis has been observed.", "rerank_score": 2.1},
]


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def detect(self, question):
        return "semaglutide"

    def retrieve(self, question, drug=None, k=5, mode="hybrid_rerank"):
        return self.chunks[:k]


class FakeLLM:
    """Mimics the Groq client: returns the given replies in order and counts calls."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def reply(answer, citations, found=True):
    return json.dumps({"answer": answer, "citations": citations, "found_in_sources": found})


def test_answer_includes_citation_metadata():
    llm = FakeLLM(reply("Wegovy has a boxed warning for thyroid C-cell tumors [1].", [1]))
    result = LabelQA(FakeRetriever(CHUNKS), llm).answer("What is the boxed warning for Wegovy?")
    assert result.found_in_sources
    assert result.citations[0].section == "Boxed Warning"
    assert result.drug == "semaglutide"


def test_declines_without_calling_llm_when_nothing_relevant():
    irrelevant = [{**CHUNKS[0], "rerank_score": -8.0}]
    llm = FakeLLM()
    result = LabelQA(FakeRetriever(irrelevant), llm).answer("What does Wegovy cost?")
    assert result.answer == NOT_FOUND and not result.found_in_sources
    assert llm.calls == 0


def test_invalid_json_is_retried_once():
    llm = FakeLLM("not json", reply("Pancreatitis has been observed [2].", [2]))
    result = LabelQA(FakeRetriever(CHUNKS), llm).answer("Does Wegovy cause pancreatitis?")
    assert llm.calls == 2 and result.citations[0].number == 2


def test_citations_to_passages_that_do_not_exist_are_dropped():
    llm = FakeLLM(reply("Made-up claim [7].", [7]))
    result = LabelQA(FakeRetriever(CHUNKS), llm).answer("Anything?")
    assert not result.found_in_sources and result.answer == NOT_FOUND


def test_model_can_say_the_answer_is_not_in_the_labels():
    llm = FakeLLM(reply("The labels don't say.", [], found=False))
    result = LabelQA(FakeRetriever(CHUNKS), llm).answer("What is the price of Wegovy?")
    assert not result.found_in_sources and result.citations == []
