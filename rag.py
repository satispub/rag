
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from .cache import RedisSemanticCache
from .config import Settings
from .embeddings import FastEmbedDense, FastEmbedSparse
from .ingestion import ingest_dir, ingest_file
from .observability import TraceHandler
from .reranker import Reranker
from .vectorstore import VectorStore


PROMPT = ChatPromptTemplate.from_template(
""" You are a banking customer service AI assistant.
Answer the customer's question using the provided context.

If the issue involves a delegate enrollment failure, duplicate delegate error, or a failed attempt to delete a delegate, instruct the user to open a support ticket through chat option.


If the answer is not contained in the context, say
"I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)


class RAG:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings()

        # Embeddings and vector store
        self.dense = FastEmbedDense(self.s.embed_model)
        self.sparse = FastEmbedSparse(self.s.sparse_model)
        self.vs = VectorStore(
            self.s,
            self.dense,
            self.sparse,
        )

        # Local LLM
        self.llm = ChatOllama(
            base_url=self.s.ollama_base_url,
            model=self.s.llm_model,
            temperature=self.s.temperature,
            num_ctx=8192,
        )

        # Semantic cache
        self.cache = RedisSemanticCache(
            self.s,
            self.dense,
        )

        # Optional reranker
        self.reranker = (
            Reranker(self.s.reranker_model)
            if self.s.use_reranker
            else None
        )

        # Observability and tracing
        self.trace = TraceHandler(self.s.log_dir)

    def retrieve(self, question: str):
        """Retrieve relevant documents using hybrid search."""
        docs = self.vs.hybrid_search(question)

        if self.reranker:
            docs = self.reranker.rerank(
                question,
                docs,
                self.s.rerank_top_n,
            )

        return docs

    @staticmethod
    def _serialize_sources(docs):
        """Convert retrieved documents into JSON-friendly sources."""
        return [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content[:400],
            }
            for doc in docs
        ]

    def ask(self, question: str) -> dict:
        """Answer a question using retrieval, the LLM, and caching."""
        start_time = time.time()

        # Check semantic cache first
        cached, hit_kind = self.cache.lookup(
            question,
            self.s.llm_model,
        )

        if cached:
            return {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "cache_hit": hit_kind,
                "latency_s": round(
                    time.time() - start_time, 2
                ),
            }

        # Retrieve relevant context
        docs = self.retrieve(question)

        context = "\n\n".join(
            f"[{index + 1}] {doc.page_content}"
            for index, doc in enumerate(docs)
        )

        # Generate the answer
        chain = PROMPT | self.llm | StrOutputParser()

        answer = chain.invoke(
            {
                "context": context,
                "question": question,
            },
            config={"callbacks": [self.trace]},
        )

        result = {
            "answer": answer.strip(),
            "sources": self._serialize_sources(docs),
            "cache_hit": None,
            "latency_s": round(
                time.time() - start_time, 2
            ),
        }

        # Cache the answer and its sources
        self.cache.store(
            question,
            self.s.llm_model,
            {
                "answer": result["answer"],
                "sources": result["sources"],
            },
        )

        return result

    def ingest_path(self, path: str) -> int:
        """Ingest a single file or all supported files in a directory."""
        path_obj = Path(path)

        if path_obj.is_dir():
            counts = ingest_dir(
                self.vs,
                str(path_obj),
                self.s,
            )
            return sum(counts.values())

        return ingest_file(
            self.vs,
            path_obj,
            self.s,
        )

    def collection_count(self) -> int:
        """Return the number of documents in the vector store."""
        return self.vs.count()
