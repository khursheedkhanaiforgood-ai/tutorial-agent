from pydantic import BaseModel


class TutorialStep(BaseModel):
    index: int
    title: str
    content: str
    completed: bool = False


class TutorialDocument(BaseModel):
    title: str
    steps: list[TutorialStep]


class SearchResult(BaseModel):
    source: str
    snippet: str
    relevance_score: float


class StepContext(BaseModel):
    step: TutorialStep
    web_results: list[SearchResult]
    rag_results: list[SearchResult]
