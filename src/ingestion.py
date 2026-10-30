from pathlib import Path
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(path)
        pages = [{"text": page.get_text("text"), "page": i + 1} for i, page in enumerate(doc)]
        doc.close()
        return pages
    if suffix == ".txt":
        return [{"text": path.read_text(encoding="utf-8", errors="ignore"), "page": 1}]
    raise ValueError("Unsupported file type: " + suffix + " (supported: .pdf, .txt)")


def ingest_file(vectorstore, path, settings) -> int:
    pages = load_file(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks, metas = [], []
    for page in pages:
        for chunk in splitter.split_text(page["text"]):
            chunks.append(chunk)
            metas.append({"source": Path(path).name, "page": page["page"]})
    # simple de-duplication
    seen, uniq_chunks, uniq_metas = set(), [], []
    for c, m in zip(chunks, metas):
        key = c.strip()
        if key and key not in seen:
            seen.add(key)
            uniq_chunks.append(c)
            uniq_metas.append(m)
    if uniq_chunks:
        vectorstore.add_documents(uniq_chunks, uniq_metas)
    return len(uniq_chunks)


def ingest_dir(vectorstore, data_dir, settings) -> dict:
    results = {}
    for path in sorted(Path(data_dir).rglob("*")):
        if path.suffix.lower() in (".pdf", ".txt") and path.is_file():
            results[path.name] = ingest_file(vectorstore, path, settings)
    return results
