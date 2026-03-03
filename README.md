# Interactive Multi-Agent Tutorial System

A CLI application that reads a Word or PDF tutorial document and guides users through it interactively using a three-tier AI agent architecture:

- **Orchestrator** — Claude claude-sonnet-4-6 drives the session, calls tools, synthesises answers
- **Web Search Agents** — Tavily searches the internet for up-to-date context
- **RAG Agents** — A local vector store retrieves relevant internal documents
- **Hallucination Check** — Every response is fact-checked against retrieved sources using Claude Haiku

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [API Keys](#2-api-keys)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Running the App](#5-running-the-app)
6. [Project Structure](#6-project-structure)
7. [How It Works](#7-how-it-works)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Minimum version | How to check |
|-------------|----------------|--------------|
| Python | 3.10+ | `python3 --version` |
| pip | 23+ | `pip3 --version` |

> **macOS users:** Python 3.14 is supported. Python can be installed via [python.org](https://www.python.org/downloads/) or `brew install python`.

---

## 2. API Keys

You need **two** API keys before running the app.

### Anthropic (Claude)
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign in → **API Keys** in the left sidebar
3. Click **Create Key** — copy the value (starts with `sk-ant-`)

### Tavily (Web Search)
1. Go to [app.tavily.com](https://app.tavily.com)
2. Sign in — your API key is shown on the dashboard (starts with `tvly-`)
3. Free tier: 1,000 searches/month

---

## 3. Installation

### Option A — Run directly from source (recommended for most users)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tutorial-agent.git
cd tutorial-agent

# Install dependencies
pip3 install -r requirements.txt
```

### Option B — Install as a package

```bash
git clone https://github.com/YOUR_USERNAME/tutorial-agent.git
cd tutorial-agent

pip3 install .
```

This installs a `tutorial-agent` command you can run from anywhere.

### Option C — Install directly from GitHub (no clone needed)

```bash
pip3 install git+https://github.com/YOUR_USERNAME/tutorial-agent.git
```

---

## 4. Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Open `.env` in any text editor:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
TAVILY_API_KEY=tvly-your-key-here
```

> ⚠️ **Never commit `.env` to Git.** It is listed in `.gitignore` and will not be tracked.

### Optional settings (add to `.env` to override defaults)

```
NUM_WEB_AGENTS=2        # Concurrent web search agents (1–5)
NUM_RAG_AGENTS=2        # Concurrent RAG agents (1–5)
CHROMA_PERSIST_DIR=.chroma_db   # Where the vector store is saved
```

---

## 5. Running the App

### From source

```bash
cd tutorial-agent
python3 main.py
```

### If installed as a package

```bash
tutorial-agent
```

The app will prompt you for:

| Prompt | What to enter |
|--------|--------------|
| Path to tutorial document | Full path to a `.docx` or `.pdf` file |
| Path to internal docs folder | A folder containing `.pdf`, `.docx`, `.txt`, or `.md` files for RAG context. Enter `.` to use the current directory. |
| Number of web search agents | 1–5 (default: 2) |
| Number of RAG agents | 1–5 (default: 2) |

### In-session commands

| You type | Effect |
|----------|--------|
| Any question | Claude searches the web + internal docs and answers |
| `next` | Advance to the next step |
| `n` | Same as `next` |
| Ctrl+C | Exit the session |

---

## 6. Project Structure

```
tutorial_agent/
├── main.py                 # Entry point
├── config.py               # Pydantic-Settings config (reads .env)
├── models.py               # Shared data models (TutorialStep, SearchResult, …)
├── document_reader.py      # Parses .docx and .pdf into TutorialDocument
│
├── agents/
│   ├── orchestrator.py     # Claude tool-use loop + hallucination verification
│   ├── web_search_pool.py  # Async Tavily agent pool
│   └── rag_pool.py         # Async vector-store query pool
│
├── rag/
│   ├── indexer.py          # Chunks, embeds, and upserts docs to the vector store
│   ├── retriever.py        # Semantic query wrapper
│   └── vector_store.py     # Lightweight numpy+pickle vector store (no external DB)
│
├── ui/
│   └── cli.py              # Rich console UI + questionary prompts
│
├── requirements.txt        # pip-installable dependencies
├── pyproject.toml          # Package metadata (for pip install .)
└── .env.example            # Template — copy to .env and fill in keys
```

---

## 7. How It Works

```
User input / step advance
        │
        ▼
  Orchestrator (Claude claude-sonnet-4-6)
  ┌──────────────────────────────────────────┐
  │  Builds prompt → calls tools             │
  │                                          │
  │  tool: web_search(queries)  ──►  Tavily  │
  │  tool: rag_search(queries)  ──►  Vector  │
  │                                   Store  │
  │  asyncio.gather(both calls)              │
  │  ◄── merged SearchResult lists          │
  │                                          │
  │  Synthesises → answer with citations     │
  └──────────────────────────────────────────┘
        │ answer shown immediately
        ▼
  Hallucination Check (Claude Haiku — fast)
  ┌──────────────────────────────────────────┐
  │  Compares answer vs. raw source text     │
  │  ✅ All verified  OR  ⚠️ flagged claims  │
  └──────────────────────────────────────────┘
        │
        ▼
    Rich CLI (formatted markdown output)
```

**RAG indexing** happens once on first run. The vector store is persisted to `.chroma_db/` so subsequent runs skip re-indexing.

---

## 8. Troubleshooting

### `RuntimeError: asyncio.run() cannot be called from a running event loop`
This is a known conflict between `questionary` and asyncio on Python 3.12+. It is already handled in this version — if you see it, make sure you are running the latest code from `main` branch.

### `WARNING: You are sending unauthenticated requests to the HF Hub`
Harmless. The sentence-transformers model (`all-MiniLM-L6-v2`) downloads without authentication. Set `HF_TOKEN` in your environment if you hit rate limits.

### `ModuleNotFoundError`
Run `pip3 install -r requirements.txt` again from the project root.

### ChromaDB errors on Python 3.14
ChromaDB ≥ 1.x uses Pydantic v1 internally which is incompatible with Python 3.14. This project replaces ChromaDB with a built-in numpy+pickle vector store — no external DB required.

### Tavily returns no results
Check your `TAVILY_API_KEY` in `.env`. Free-tier keys expire after 1,000 searches/month.

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss significant changes.

---

## License

MIT
