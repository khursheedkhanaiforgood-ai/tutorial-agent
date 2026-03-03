from __future__ import annotations

from models import SearchResult
from rag.vector_store import VectorStore

_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder


class Retriever:
    def __init__(self, collection: VectorStore):
        self._collection = collection

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        embedder = _get_embedder()
        query_embedding = embedder.encode([query], show_progress_bar=False)[0].tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self._collection.count(), 1)),
        )

        search_results: list[SearchResult] = []
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            # ChromaDB returns L2 distances; convert to a 0-1 relevance score
            relevance = 1.0 / (1.0 + dist)
            search_results.append(SearchResult(
                source=meta.get("source", "unknown"),
                snippet=doc[:500],
                relevance_score=round(relevance, 4),
            ))

        return search_results
