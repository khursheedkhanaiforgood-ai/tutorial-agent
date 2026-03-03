from pathlib import Path
from typing import Callable

from rag.vector_store import VectorStore

_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50
_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif suffix == ".docx":
        import docx
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def index_documents(
    folder: Path,
    collection: VectorStore,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    supported = {".pdf", ".docx", ".txt", ".md"}
    files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in supported]

    if not files:
        return 0

    # Check if already indexed by looking for docs with folder metadata
    existing = collection.get(where={"folder": str(folder)}, limit=1)
    if existing and existing["ids"]:
        # Already indexed — return approximate count
        all_docs = collection.get(where={"folder": str(folder)})
        return len(set(m.get("source", "") for m in (all_docs["metadatas"] or [])))

    embedder = _get_embedder()
    doc_count = 0

    for file_idx, file_path in enumerate(files):
        text = _extract_text(file_path)
        if not text.strip():
            continue

        chunks = _chunk_text(text)
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

        ids = [f"{file_path.name}::chunk{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path.name, "folder": str(folder), "chunk": i} for i in range(len(chunks))]

        collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        doc_count += 1

        if progress_callback:
            progress_callback(file_idx + 1, len(files))

    return doc_count


def get_collection(persist_dir: Path) -> VectorStore:
    return VectorStore(persist_dir, name="internal_docs")
