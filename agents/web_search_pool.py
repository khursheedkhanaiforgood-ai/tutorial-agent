import asyncio
from tavily import AsyncTavilyClient

from models import SearchResult


class WebSearchPool:
    def __init__(self, num_agents: int, tavily_api_key: str):
        self._semaphore = asyncio.Semaphore(num_agents)
        self._client = AsyncTavilyClient(api_key=tavily_api_key)

    async def _search_one(self, query: str) -> list[SearchResult]:
        async with self._semaphore:
            try:
                response = await self._client.search(
                    query=query,
                    search_depth="basic",
                    max_results=5,
                )
                results: list[SearchResult] = []
                for item in response.get("results", []):
                    results.append(SearchResult(
                        source=item.get("url", ""),
                        snippet=item.get("content", "")[:500],
                        relevance_score=round(item.get("score", 0.0), 4),
                    ))
                return results
            except Exception as exc:
                return [SearchResult(source="error", snippet=str(exc), relevance_score=0.0)]

    async def search_all(self, queries: list[str]) -> list[SearchResult]:
        tasks = [self._search_one(q) for q in queries]
        nested = await asyncio.gather(*tasks)
        merged: dict[str, SearchResult] = {}
        for batch in nested:
            for r in batch:
                if r.source not in merged or r.relevance_score > merged[r.source].relevance_score:
                    merged[r.source] = r
        return sorted(merged.values(), key=lambda x: x.relevance_score, reverse=True)
