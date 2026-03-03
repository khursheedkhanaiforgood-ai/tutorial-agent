from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    tavily_api_key: str = ""
    tutorial_doc_path: Path = Path("")
    internal_docs_path: Path = Path("")
    num_web_agents: int = Field(2, ge=1, le=5)
    num_rag_agents: int = Field(2, ge=1, le=5)
    chroma_persist_dir: Path = Path(".chroma_db")
