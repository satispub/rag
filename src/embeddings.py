from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_core.embeddings import Embeddings
from qdrant_client.models import SparseVector


class FastEmbedDense(Embeddings):
    """LangChain-compatible dense embedding (BGE) backed by fastembed (ONNX, no torch)."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [list(v) for v in self._model.embed(texts)]

    def embed_query(self, text):
        return list(next(self._model.embed([text])))

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


class FastEmbedSparse:
    """BM25 sparse embeddings for keyword/hybrid retrieval."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_query(self, text) -> SparseVector:
        v = list(self._model.embed([text]))[0]
        return SparseVector(indices=v.indices.tolist(), values=v.values.tolist())

    def embed_documents(self, texts):
        return list(self._model.embed(texts))
