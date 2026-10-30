import json
import os
import time
import uuid
from langchain_core.callbacks import BaseCallbackHandler


class TraceHandler(BaseCallbackHandler):
    """Lightweight JSONL tracer for LLM calls, retrieval, and chain events.
    Complements optional LangSmith tracing (enable via LANGCHAIN_TRACING_V2=true)."""

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "traces.jsonl")
        self.trace_id = uuid.uuid4().hex[:12]
        self._starts = {}

    def _write(self, event: dict):
        event.setdefault("trace_id", self.trace_id)
        event.setdefault("ts", time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def on_llm_start(self, serialized, prompts, run_id, **kwargs):
        self._starts[str(run_id)] = time.time()
        self._write({"event": "llm_start", "run_id": str(run_id),
                     "model": kwargs.get("invocation_params", {}).get("model")})

    def on_llm_end(self, response, run_id, **kwargs):
        usage = getattr(response, "llm_output", None) or {}
        self._write({
            "event": "llm_end", "run_id": str(run_id),
            "latency_s": round(time.time() - self._starts.get(str(run_id), time.time()), 3),
            "output": (response.generations[0][0].text or "")[:500],
            "usage": usage.get("token_usage") or usage.get("usage"),
        })

    def on_retriever_start(self, serialized, query, run_id, **kwargs):
        self._write({"event": "retriever_start", "run_id": str(run_id), "query": query})

    def on_retriever_end(self, documents, run_id, **kwargs):
        self._write({"event": "retriever_end", "run_id": str(run_id), "num_docs": len(documents)})

    def on_chain_start(self, serialized, inputs, run_id, **kwargs):
        self._write({"event": "chain_start", "run_id": str(run_id), "keys": list(inputs.keys())})




def read_traces(log_dir: str = "logs", limit: int = 50):
    path = os.path.join(log_dir, "traces.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines[-limit:]]
