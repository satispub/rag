#!/usr/bin/env python
"""Run Ragas evaluation: python scripts/evaluate.py eval/qa_pairs.json"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import settings
from src.evaluation import run_evaluation
from src.rag import RAG


def main():
    qa_path = sys.argv[1] if len(sys.argv) > 1 else "eval/qa_pairs.json"
    rag = RAG(settings)
    result = run_evaluation(rag, qa_path)
    df = result.to_pandas()
    print(df.to_string())
    df.to_csv("eval/results.csv", index=False)
    print("\nSaved -> eval/results.csv")


if __name__ == "__main__":
    main()
