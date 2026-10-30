import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError:  # older ragas layout
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

DEFAULT_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def load_qa(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_evaluation(rag, qa_path: str, metrics=None):
    qa = load_qa(qa_path)
    rows = []
    for item in qa:
        docs = rag.retrieve(item["question"])
        answer = rag.ask(item["question"])["answer"]
        rows.append({
            "question": item["question"],
            "answer": answer,
            "contexts": [d.page_content for d in docs],
            "ground_truth": item["ground_truth"],
        })
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=metrics or DEFAULT_METRICS,
        llm=LangchainLLMWrapper(rag.llm),
        embeddings=LangchainEmbeddingsWrapper(rag.dense),
    )
    return result
