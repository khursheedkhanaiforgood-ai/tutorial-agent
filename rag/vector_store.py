"""Lightweight numpy+pickle vector store — drop-in replacement for chromadb.Collection."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, persist_dir: Path, name: str = "default"):
        self._path = persist_dir / f"{name}.pkl"
        self._ids: list[str] = []
        self._embeddings: list[list[float]] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = pickle.loads(self._path.read_bytes())
            self._ids = data["ids"]
            self._embeddings = data["embeddings"]
            self._documents = data["documents"]
            self._metadatas = data["metadatas"]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(pickle.dumps({
            "ids": self._ids,
            "embeddings": self._embeddings,
            "documents": self._documents,
            "metadatas": self._metadatas,
        }))

    def count(self) -> int:
        return len(self._ids)

    def get(self, where: dict | None = None, limit: int | None = None) -> dict:
        if where is None:
            ids = self._ids[:limit] if limit else self._ids[:]
            metas = self._metadatas[:len(ids)]
            return {"ids": ids, "metadatas": metas}

        matching_ids, matching_metas = [], []
        for id_, meta in zip(self._ids, self._metadatas):
            if all(meta.get(k) == v for k, v in where.items()):
                matching_ids.append(id_)
                matching_metas.append(meta)
                if limit and len(matching_ids) >= limit:
                    break
        return {"ids": matching_ids, "metadatas": matching_metas}

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            if id_ in self._ids:
                idx = self._ids.index(id_)
                self._embeddings[idx] = emb
                self._documents[idx] = doc
                self._metadatas[idx] = meta
            else:
                self._ids.append(id_)
                self._embeddings.append(emb)
                self._documents.append(doc)
                self._metadatas.append(meta)
        self._save()

    def query(self, query_embeddings: list[list[float]], n_results: int = 5) -> dict:
        if not self._embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.array(query_embeddings[0], dtype=np.float32)
        store = np.array(self._embeddings, dtype=np.float32)

        q_norm = q / (np.linalg.norm(q) + 1e-10)
        s_norms = store / (np.linalg.norm(store, axis=1, keepdims=True) + 1e-10)
        distances = (1.0 - s_norms @ q_norm).tolist()

        top_k = min(n_results, len(self._ids))
        top_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:top_k]

        return {
            "documents": [[self._documents[i] for i in top_indices]],
            "metadatas": [[self._metadatas[i] for i in top_indices]],
            "distances": [[distances[i] for i in top_indices]],
        }
