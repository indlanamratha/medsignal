"""Hybrid retrieval over FDA label chunks: keyword (BM25) + vector search, fused and reranked.

Modes (compared in the evaluation):
    dense          vector search only
    bm25           keyword search only
    hybrid         both, merged with reciprocal rank fusion
    hybrid_rerank  hybrid, then a cross-encoder reorders the candidates (default)

Build the index first:  uv run python -m medsignal.rag.retrieve --build
Try a question:         uv run python -m medsignal.rag.retrieve "What is the boxed warning for Wegovy?"
"""
import argparse
import re

import pandas as pd
from rank_bm25 import BM25Okapi

from medsignal.config import DATA_DIR
from medsignal.rag.labels import CHUNKS_PATH

CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "fda_labels"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CANDIDATES = 20
MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge ranked lists: each item scores 1 / (k + rank) in every list it appears in."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def drug_aliases(chunks: pd.DataFrame) -> dict[str, str]:
    """Map every brand and generic name (lowercase) to its drug group."""
    aliases = {}
    for column in ("drug_group", "brand_name", "generic_name"):
        for name, group in zip(chunks[column].str.lower(), chunks["drug_group"]):
            aliases.setdefault(name, group)
    return aliases


def detect_drug(question: str, aliases: dict[str, str]) -> str | None:
    """Find which study drug a question is about, by generic or brand name."""
    text = question.lower()
    for name in sorted(aliases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", text):
            return aliases[name]
    return None


def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def _collection(create: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if create:
        if COLLECTION in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION)
        return client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}},
                                        embedding_function=None)
    return client.get_collection(COLLECTION, embedding_function=None)


def build_index(batch_size: int = 256) -> None:
    chunks = pd.read_parquet(CHUNKS_PATH)
    embeddings = _embedder().encode(chunks["context_text"].tolist(), batch_size=32,
                                    normalize_embeddings=True, show_progress_bar=True)
    collection = _collection(create=True)
    meta_cols = ["drug_group", "brand_name", "generic_name", "section", "set_id", "effective_date"]
    for start in range(0, len(chunks), batch_size):
        part = chunks.iloc[start:start + batch_size]
        collection.add(ids=part["chunk_id"].tolist(),
                       embeddings=embeddings[start:start + batch_size].tolist(),
                       metadatas=part[meta_cols].to_dict("records"),
                       documents=part["text"].tolist())
    print(f"Indexed {collection.count():,} chunks in ChromaDB at {CHROMA_DIR}")


class LabelRetriever:
    def __init__(self, embedder=None, reranker=None, collection=None):
        self.chunks = pd.read_parquet(CHUNKS_PATH).set_index("chunk_id", drop=False)
        self.ids = self.chunks["chunk_id"].tolist()
        self.bm25 = BM25Okapi([tokenize(t) for t in self.chunks["context_text"]])
        self.aliases = drug_aliases(self.chunks)
        self._embedder, self._reranker, self._collection = embedder, reranker, collection

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = _embedder()
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    @property
    def collection(self):
        if self._collection is None:
            self._collection = _collection()
        return self._collection

    def _dense(self, question: str, drug: str | None) -> list[str]:
        query = self.embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True)[0].tolist()
        result = self.collection.query(query_embeddings=[query], n_results=CANDIDATES,
                                       where={"drug_group": drug} if drug else None)
        return result["ids"][0]

    def _bm25(self, question: str, drug: str | None) -> list[str]:
        scores = pd.Series(self.bm25.get_scores(tokenize(question)), index=self.ids)
        if drug:
            scores = scores[(self.chunks["drug_group"] == drug).to_numpy()]
        return scores.nlargest(CANDIDATES).index.tolist()

    def retrieve(self, question: str, drug: str | None = None, k: int = 5,
                 mode: str = "hybrid_rerank") -> list[dict]:
        """Return the top-k chunks for a question, each with its citation metadata."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        drug = drug or detect_drug(question, self.aliases)
        if mode == "dense":
            ranked = self._dense(question, drug)
        elif mode == "bm25":
            ranked = self._bm25(question, drug)
        else:
            ranked = reciprocal_rank_fusion([self._dense(question, drug), self._bm25(question, drug)])

        candidates = self.chunks.loc[ranked].to_dict("records")
        if mode == "hybrid_rerank" and candidates:
            scores = self.reranker.predict([(question, c["context_text"]) for c in candidates])
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search FDA label chunks.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--drug", help="Restrict to one drug group, e.g. semaglutide")
    parser.add_argument("--mode", default="hybrid_rerank", choices=MODES)
    parser.add_argument("--build", action="store_true", help="Embed all chunks into ChromaDB")
    args = parser.parse_args()

    if args.build:
        build_index()
    if args.question:
        retriever = LabelRetriever()
        drug = args.drug or detect_drug(args.question, retriever.aliases)
        print(f"Question: {args.question}\nDrug filter: {drug or 'none'}   Mode: {args.mode}\n")
        for i, chunk in enumerate(retriever.retrieve(args.question, drug, mode=args.mode), start=1):
            score = f"  rerank={chunk['rerank_score']:.2f}" if "rerank_score" in chunk else ""
            print(f"[{i}] {chunk['brand_name']} | {chunk['section']}{score}")
            print(f"    {chunk['text'][:300]}...\n")


if __name__ == "__main__":
    main()
