import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from langchain_core.documents import Document


class VectorStore:
    """Qdrant store with dense (BGE) + sparse (BM25) named vectors and RRF-style fusion."""

    def __init__(self, settings, dense_embedder, sparse_embedder):
        self.s = settings
        self.dense = dense_embedder
        self.sparse = sparse_embedder
        self.client = QdrantClient(url=settings.qdrant_url, timeout=30)
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.collection_exists(self.s.collection_name):
            return
        self.client.create_collection(
            collection_name=self.s.collection_name,
            vectors_config={"dense": VectorParams(size=self.s.vector_size, distance=Distance.COSINE)},
            sparse_vectors_config={"bm25": SparseVectorParams(index=SparseIndexParams())},
        )

    def add_documents(self, texts, metadatas):
        dense_vecs = self.dense.embed_documents(texts)
        sparse_vecs = self.sparse.embed_documents(texts)
        points = []
        for dv, sv, text, meta in zip(dense_vecs, sparse_vecs, texts, metadatas):
            payload = dict(meta)
            payload["text"] = text
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dv,
                        "bm25": SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                    },
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.s.collection_name, points=points)

    def count(self):
        return self.client.count(self.s.collection_name).count

    def _rrf_fuse(self, dense_hits, sparse_hits, alpha):
        scores, merged = {}, {}
        for rank, hit in enumerate(dense_hits):
            scores[hit.id] = scores.get(hit.id, 0.0) + alpha * (1.0 / (60 + rank))
        for rank, hit in enumerate(sparse_hits):
            scores[hit.id] = scores.get(hit.id, 0.0) + (1 - alpha) * (1.0 / (60 + rank))
        for hit in dense_hits + sparse_hits:
            merged[hit.id] = hit
        return sorted(merged.values(), key=lambda p: scores[p.id], reverse=True)

    def hybrid_search(self, query: str, top_k: int = None, alpha: float = None):
        top_k = top_k or self.s.top_k
        alpha = self.s.hybrid_alpha if alpha is None else alpha
        dv = self.dense.embed_query(query)
        sv = self.sparse.embed_query(query)
        common = dict(collection_name=self.s.collection_name, limit=top_k, with_payload=True)
        dense_hits = self.client.query_points(query=dv, using="dense", **common).points
        sparse_hits = self.client.query_points(query=sv, using="bm25", **common).points
        fused = self._rrf_fuse(dense_hits, sparse_hits, alpha)[:top_k]
        return [
            Document(
                page_content=p.payload["text"],
                metadata={k: v for k, v in p.payload.items() if k != "text"},
            )
            for p in fused
        ]

    def scroll_all(self, limit: int = 100):
        return self.client.scroll(self.s.collection_name, limit=limit)[0]

    def delete_by_source(self, source: str):
        self.client.delete(
            collection_name=self.s.collection_name,
            points_selector=Filter(must=[{"key": "source", "match": {"value": source}}]),
        )
