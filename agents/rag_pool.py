import asyncio
from concurrent.futures import ThreadPoolExecutor

from models import SearchResult
from rag.retriever import Retriever


class RagPool:
    def __init__(self, num_agents: int, retriever: Retriever):
        self._semaphore = asyncio.Semaphore(num_agents)
        self._retriever = retriever
        self._executor = ThreadPoolExecutor(max_workers=num_agents)

    async def _query_one(self, query: str) -> list[SearchResult]:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self._retriever.retrieve,
                query,
            )

    async def query_all(self, queries: list[str]) -> list[SearchResult]:
        tasks = [self._query_one(q) for q in queries]
        nested = await asyncio.gather(*tasks)
        merged: dict[str, SearchResult] = {}
        for batch in nested:
            for r in batch:
                key = f"{r.source}::{r.snippet[:50]}"
                if key not in merged or r.relevance_score > merged[key].relevance_score:
                    merged[key] = r
        return sorted(merged.values(), key=lambda x: x.relevance_score, reverse=True)
