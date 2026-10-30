class Reranker:
    """Cross-encoder reranker (BGE). Lazy-loaded so it can be disabled via config."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs, top_n: int = 5):
        if not docs:
            return docs
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]
