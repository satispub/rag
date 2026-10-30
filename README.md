# Local RAG — LangChain · Ollama · Qwen3/Llama3 · BGE · Qdrant · Redis · Streamlit

A fully local, runnable Retrieval-Augmented Generation repository. No API keys, no cloud.

## Features
- **LLM**: Ollama-hosted `qwen3:8b` (default) or `llama3.1:8b` via `langchain-ollama`
- **Embeddings**: BGE dense (`BAAI/bge-base-en-v1.5`, fastembed/ONNX — no torch needed for retrieval)
- **Hybrid search**: dense vectors + BM25 sparse vectors in Qdrant, fused with weighted RRF
- **Reranker**: `BAAI/bge-reranker-base` cross-encoder, toggleable
- **Cache**: two-tier Redis cache (exact hash + semantic cosine similarity)
- **Ingestion**: PDF (PyMuPDF) + TXT, recursive chunking, de-duplication
- **UI**: Streamlit app — ask, ingest files, view sources, run evaluation, inspect traces
- **Evaluation**: Ragas metrics (faithfulness, answer relevancy, context precision/recall)
- **Observability**: JSONL trace logs of every LLM/retrieval/chain event + optional LangSmith

## Quickstart

```bash
# 1. Prereqs: Docker, Python 3.10+, Ollama (https://ollama.com)
git clone <your-repo> && cd local-rag-repo
cp .env.example .env

# 2. Start infra + pull models + install deps
chmod +x scripts/setup.sh && ./scripts/setup.sh

# 3. Ingest documents
python scripts/ingest.py data

# 4. Launch UI
streamlit run app.py
```

Then open http://localhost:8501, ask questions, upload PDFs/TXTs in the sidebar, or run the Ragas eval in the **Evaluate** tab.

## CLI

```bash
python scripts/ingest.py data/my_doc.pdf   # ingest a file or folder
python scripts/evaluate.py eval/qa_pairs.json
```

## Configuration

All via `.env` (see `.env.example`): model, top-k, hybrid alpha, cache threshold/TTL, chunking, reranker toggle. Switch LLM with `LLM_MODEL=llama3.1:8b`.

## Architecture

```
PDF/TXT → chunk → BGE dense + BM25 sparse → Qdrant (hybrid RRF) → reranker → prompt → Ollama LLM
                                       ↘ Redis semantic cache (exact + cosine) ↗
Observability: JSONL traces per request; Ragas for offline quality metrics.
```

## Notes & limits
- Semantic cache uses a linear scan over cached embeddings — perfectly fine for thousands of entries; swap for a vector index if you cache at scale.
- First embed runs download ONNX models (~1 GB total); first LLM answer warms up Ollama.
- Ragas metrics call the local LLM as judge, so evals take a few minutes.
