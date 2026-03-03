from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import Orchestrator
from agents.rag_pool import RagPool
from agents.web_search_pool import WebSearchPool
from config import AppConfig
from document_reader import read_document
from rag.indexer import get_collection, index_documents
from rag.retriever import Retriever
from ui.cli import TutorialCLI


async def run(config: AppConfig, cli: TutorialCLI) -> None:
    # Parse tutorial document
    try:
        doc = read_document(config.tutorial_doc_path)
    except Exception as exc:
        cli.show_error(f"Could not read document: {exc}")
        return

    cli._console.print(
        f"[green]✓[/green] Loaded [bold]{doc.title}[/bold] "
        f"({len(doc.steps)} step(s))."
    )

    # Initialise vector store and index internal docs
    collection = get_collection(config.chroma_persist_dir)

    cli.show_indexing_start()
    with cli.indexing_progress() as progress:
        task_id = progress.add_task("Indexing documents…", total=None)

        def _progress_cb(done: int, total: int) -> None:
            progress.update(task_id, completed=done, total=total)

        n_indexed = index_documents(config.internal_docs_path, collection, _progress_cb)

    cli.show_indexed(n_indexed)

    # Build agent pools
    web_pool = WebSearchPool(config.num_web_agents, config.tavily_api_key)
    rag_pool = RagPool(config.num_rag_agents, Retriever(collection))

    # Run tutorial
    orchestrator = Orchestrator(config, web_pool, rag_pool, doc)
    await orchestrator.run_tutorial(cli)


def main() -> None:
    cli = TutorialCLI()
    # greet_and_configure uses questionary which calls asyncio.run() internally,
    # so it must run BEFORE we start our own event loop.
    config = cli.greet_and_configure()
    asyncio.run(run(config, cli))


if __name__ == "__main__":
    main()
