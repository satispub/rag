#!/usr/bin/env python
"""CLI ingestion: python scripts/ingest.py data"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import settings
from src.rag import RAG


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else settings.data_dir
    rag = RAG(settings)
    n = rag.ingest_path(path)
    print(f"Ingested {n} chunks. Collection '{settings.collection_name}' now has {rag.collection_count()} points.")


if __name__ == "__main__":
    main()
