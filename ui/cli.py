from __future__ import annotations

import asyncio
from pathlib import Path

import questionary
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.text import Text

from config import AppConfig

console = Console()


def _validate_path(value: str, must_exist: bool = True, is_dir: bool = False) -> bool | str:
    p = Path(value.strip())
    if must_exist and not p.exists():
        return f"Path does not exist: {p}"
    if is_dir and p.exists() and not p.is_dir():
        return f"Expected a directory: {p}"
    return True


class TutorialCLI:
    def __init__(self):
        self._console = console

    # ------------------------------------------------------------------ setup

    def greet_and_configure(self) -> AppConfig:
        self._console.print(
            Panel.fit(
                "[bold cyan]Interactive Multi-Agent Tutorial System[/bold cyan]\n"
                "[dim]Powered by Claude + Tavily + ChromaDB[/dim]",
                border_style="cyan",
            )
        )
        self._console.print()

        doc_path = questionary.path(
            "Path to tutorial document (.docx or .pdf):",
            validate=lambda v: _validate_path(v, must_exist=True),
        ).ask()

        internal_path = questionary.path(
            "Path to internal docs folder (for RAG):",
            validate=lambda v: _validate_path(v, must_exist=True, is_dir=True),
            default=".",
        ).ask()

        num_web = questionary.text(
            "Number of web search agents (1-5):",
            default="2",
            validate=lambda v: v.isdigit() and 1 <= int(v) <= 5 or "Enter a number 1-5",
        ).ask()

        num_rag = questionary.text(
            "Number of RAG agents (1-5):",
            default="2",
            validate=lambda v: v.isdigit() and 1 <= int(v) <= 5 or "Enter a number 1-5",
        ).ask()

        # Load base config from .env and override with user choices
        base = AppConfig()
        return AppConfig(
            anthropic_api_key=base.anthropic_api_key,
            tavily_api_key=base.tavily_api_key,
            tutorial_doc_path=Path(doc_path.strip()),
            internal_docs_path=Path(internal_path.strip()),
            num_web_agents=int(num_web),
            num_rag_agents=int(num_rag),
            chroma_persist_dir=base.chroma_persist_dir,
        )

    # --------------------------------------------------------------- indexing

    def show_indexing_start(self) -> None:
        self._console.print("\n[dim]Indexing internal documents...[/dim]")

    def show_indexed(self, n_docs: int) -> None:
        self._console.print(
            f"[green]✓[/green] Indexed [bold]{n_docs}[/bold] document(s) into ChromaDB.\n"
        )

    def indexing_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=self._console,
            transient=True,
        )

    # ------------------------------------------------------------------ steps

    def show_step_header(self, step_index: int, total: int, title: str) -> None:
        self._console.print()
        self._console.print(Rule(
            f"[bold]Step {step_index + 1} / {total}[/bold] — {title}",
            style="blue",
        ))
        self._console.print()

    def show_verifying(self) -> None:
        self._console.print("[dim]🔍 Fact-checking in background...[/dim]")

    def show_verification_note(self, note: str) -> None:
        self._console.print(
            Panel(
                note,
                title="[bold]Verification[/bold]",
                border_style="green" if note.startswith("✅") else "yellow",
                expand=False,
            )
        )

    def show_lesson(self, text: str) -> None:
        self._console.print(Markdown(text))
        self._console.print()

    def show_summary(self, text: str) -> None:
        self._console.print()
        self._console.print(Rule("[bold green]Session Complete[/bold green]", style="green"))
        self._console.print(Markdown(text))
        self._console.print()

    # ------------------------------------------------------------------ input

    def show_qa_prompt(self) -> None:
        self._console.print(
            "[dim]──────────────────────────────────────────[/dim]\n"
            "  [bold]Ask a question[/bold] about this step, or type [bold cyan]next[/bold cyan] to continue.\n"
            "[dim]──────────────────────────────────────────[/dim]"
        )

    def ask_user(self, prompt: str = "You: ") -> str:
        try:
            return input(f"\n{prompt}").strip() or ""
        except (EOFError, KeyboardInterrupt):
            return ""

    async def ask_user_async(self, prompt: str = "You: ") -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.ask_user, prompt)

    def show_error(self, message: str) -> None:
        self._console.print(f"[bold red]Error:[/bold red] {message}")
