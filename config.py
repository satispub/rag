from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"  # switch to llama3.1:8b if preferred

    embed_model: str = "BAAI/bge-base-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = True
    rerank_top_n: int = 5

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "docs"
    vector_size: int = 768
    hybrid_alpha: float = 0.7  # dense weight; sparse gets 1 - alpha

    redis_url: str = "redis://localhost:6379/0"
    cache_threshold: float = 0.97  # cosine similarity for semantic cache hit
    cache_ttl_seconds: int = 604800  # 7 days

    chunk_size: int = 512
    chunk_overlap: int = 100
    top_k: int = 10
    temperature: float = 0.2


    data_dir: str = "data"
    log_dir: str = "logs"

settings = Settings()
