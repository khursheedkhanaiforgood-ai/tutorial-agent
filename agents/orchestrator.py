from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import anthropic

from agents.rag_pool import RagPool
from agents.web_search_pool import WebSearchPool
from config import AppConfig
from models import SearchResult, TutorialDocument, TutorialStep

if TYPE_CHECKING:
    from ui.cli import TutorialCLI

_TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information relevant to the current tutorial step. "
            "Use this to find examples, documentation, or background context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries (1-3 queries).",
                }
            },
            "required": ["queries"],
        },
    },
    {
        "name": "rag_search",
        "description": (
            "Search internal proprietary documents for relevant context. "
            "Use this when the tutorial references internal procedures, code, or company-specific details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries (1-3 queries).",
                }
            },
            "required": ["queries"],
        },
    },
]

_QUICK_VERIFY_SYSTEM = """\
You are a concise fact-checker. Given an ANSWER and its SOURCE MATERIAL, identify any claims \
in the answer that are NOT supported by the sources.

Output rules (be brief — max 5 lines):
- If every claim is supported: output exactly one line: "✅ All claims verified against sources."
- If some claims are unsupported: output a short bullet list, one bullet per unsupported claim, \
  each starting with "⚠️". Do NOT rewrite or repeat the answer.
"""


def _format_results(results: list[SearchResult], label: str) -> str:
    if not results:
        return f"[{label}: no results found]"
    lines = [f"### {label} Results"]
    for i, r in enumerate(results[:5], 1):
        lines.append(f"{i}. **{r.source}** (score: {r.relevance_score})\n   {r.snippet}")
    return "\n".join(lines)


def _build_system_prompt(doc: TutorialDocument) -> str:
    toc = "\n".join(f"  {s.index + 1}. {s.title}" for s in doc.steps)
    return (
        f"You are an expert interactive tutor guiding a learner through: **{doc.title}**.\n\n"
        f"Table of contents:\n{toc}\n\n"
        "## Tool use rules (mandatory)\n"
        "- For EVERY lesson and EVERY question, call BOTH `web_search` AND `rag_search` first.\n"
        "- Generate 2-3 targeted queries per tool.\n"
        "- Cite every fact inline as `[Source: <url or filename>]`.\n"
        "- End every response with a `## Sources` section listing all cited sources.\n\n"
        "## Response rules\n"
        "- Use markdown. Be concise but thorough.\n"
        "- Synthesise web and internal doc results together.\n"
    )


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        web_pool: WebSearchPool,
        rag_pool: RagPool,
        document: TutorialDocument,
    ):
        self._config = config
        self._web_pool = web_pool
        self._rag_pool = rag_pool
        self._document = document
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def _dispatch_tool(self, tool_name: str, tool_input: dict) -> str:
        queries: list[str] = tool_input.get("queries", [])
        if not queries:
            return "No queries provided."
        if tool_name == "web_search":
            results = await self._web_pool.search_all(queries)
            return _format_results(results, "Web Search")
        elif tool_name == "rag_search":
            results = await self._rag_pool.query_all(queries)
            return _format_results(results, "Internal Docs")
        return "Unknown tool."

    async def _call_until_text(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> tuple[str, list[str]]:
        """
        Drive the tool-use loop until Claude produces a text response.
        Returns (answer_text, collected_source_blocks).
        """
        collected_sources: list[str] = []

        while True:
            async with self._client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=_build_system_prompt(self._document),
                tools=_TOOLS,
                messages=messages,
            ) as stream:
                response = await stream.get_final_message()

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if tool_calls:
                messages.append({"role": "assistant", "content": response.content})
                tool_results = await asyncio.gather(
                    *[self._dispatch_tool(tc.name, tc.input) for tc in tool_calls]
                )
                collected_sources.extend(tool_results)
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": result,
                        }
                        for tc, result in zip(tool_calls, tool_results)
                    ],
                })
                continue

            messages.append({"role": "assistant", "content": response.content})
            return "\n".join(b.text for b in text_blocks), collected_sources

    async def _quick_verify(self, answer: str, sources: list[str]) -> str:
        """
        Fast hallucination check using Haiku. Runs AFTER the answer is already
        shown to the user, so it never blocks reading.
        Returns a compact one-liner or short bullet list.
        """
        if not sources or not answer.strip():
            return ""

        # Cap source size to keep the Haiku call fast
        source_block = "\n\n---\n\n".join(sources)
        if len(source_block) > 6000:
            source_block = source_block[:6000] + "\n...[truncated for brevity]"

        prompt = f"## ANSWER\n\n{answer}\n\n## SOURCE MATERIAL\n\n{source_block}"

        response = await self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=_QUICK_VERIFY_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(b.text for b in response.content if b.type == "text")

    async def _answer_and_verify(
        self,
        messages: list[dict],
        cli: TutorialCLI,
        max_tokens: int = 2048,
    ) -> None:
        """
        1. Fetch answer (with tool calls) — show it immediately so the user
           can start reading.
        2. Fire off a fast Haiku verification concurrently with the user reading.
        3. Append a compact verification note below the answer.
        """
        answer, sources = await self._call_until_text(messages, max_tokens)

        # Show answer right away — user does not wait for verification
        cli.show_lesson(answer)

        # Verify concurrently while user reads
        if sources:
            cli.show_verifying()
            note = await self._quick_verify(answer, sources)
            if note:
                cli.show_verification_note(note)

    async def _run_step_session(self, step: TutorialStep, cli: TutorialCLI) -> None:
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Please teach me **Step {step.index + 1}: {step.title}**.\n\n"
                    f"Document content:\n{step.content}\n\n"
                    "Call web_search AND rag_search first, then deliver the enriched lesson "
                    "with every fact cited as [Source: <url or filename>] and a "
                    "'## Sources' section at the end."
                ),
            }
        ]

        await self._answer_and_verify(messages, cli)

        # Interactive Q&A — loops until user explicitly types "next"
        while True:
            cli.show_qa_prompt()
            user_input = await cli.ask_user_async("You: ")
            command = user_input.strip().lower()

            if not command:
                continue

            if command in {"next", "n", "continue"}:
                return

            wrapped = (
                f"{user_input}\n\n"
                "(Call web_search AND rag_search before answering. "
                "Cite every fact as [Source: <url or filename>] and include a '## Sources' section.)"
            )
            messages.append({"role": "user", "content": wrapped})
            try:
                await self._answer_and_verify(messages, cli)
            except Exception as exc:
                cli.show_error(f"Error getting response: {exc}")
                messages.pop()

    async def _generate_summary(self, cli: TutorialCLI) -> None:
        titles = "\n".join(f"{s.index + 1}. {s.title}" for s in self._document.steps)
        messages = [{
            "role": "user",
            "content": (
                f"The learner completed the tutorial: **{self._document.title}**.\n"
                f"Steps covered:\n{titles}\n\n"
                "Write a brief, encouraging summary of what they learned."
            ),
        }]
        async with self._client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system="You are a friendly tutor summarizing a completed tutorial.",
            messages=messages,
        ) as stream:
            response = await stream.get_final_message()

        text = "\n".join(b.text for b in response.content if b.type == "text")
        cli.show_summary(text)

    async def run_tutorial(self, cli: TutorialCLI) -> None:
        for step in self._document.steps:
            cli.show_step_header(step.index, len(self._document.steps), step.title)
            await self._run_step_session(step, cli)
            step.completed = True
        await self._generate_summary(cli)
