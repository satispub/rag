import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src.config import Settings
from src.rag import RAG

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # 16 KB request limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Create the RAG pipeline once when the server starts.
# Ensure Ollama, Redis, and Qdrant are running and configured in src/config.py.
settings = Settings(
    llm_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    use_reranker=env_bool("USE_RERANKER", True),
)
rag = RAG(settings)


@app.get("/")
def home():
    """Serve chat.html from the project root."""
    return send_from_directory(BASE_DIR, "chat.html")


@app.get("/health")
def health():
    """Basic process health check; does not verify every dependency."""
    return jsonify(status="ok"), 200


# Serve background.png and other static files from the project root directory
@app.get("/<path:filename>")
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)


@app.post("/api/chat")
def chat():
    if not request.is_json:
        return jsonify(error="Content-Type must be application/json"), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object"), 400

    question = payload.get("message")
    if not isinstance(question, str) or not question.strip():
        return jsonify(error="Please enter a question"), 400

    question = question.strip()
    if len(question) > 2000:
        return jsonify(error="Message must be 2000 characters or fewer"), 400

    try:
        # This is the correct method from your src/rag.py.
        # RAG.ask() returns a dict containing answer, sources, cache_hit,
        # and latency_s.
        result = rag.ask(question)

        if not isinstance(result, dict):
            logger.error("RAG.ask returned unexpected type: %s", type(result).__name__)
            return jsonify(error="Unexpected response from the RAG service"), 500

        # Define answer first so it is available in scope
        answer = str(result.get("answer", ""))

        # Keywords indicating the user needs to submit a ticket
        ticket_keywords = ["duplicate", "failed", "error", "cannot delete", "deletion failure"]
        
        if any(keyword in question.lower() or keyword in answer.lower() for keyword in ticket_keywords):
            answer += "\n\nIf you are unable to resolve this issue, please click the link below to submit a ticket:\n👉 https://bank.com/create/ticket"

        return jsonify(
            answer=answer,
            sources=result.get("sources", []),
            cache_hit=result.get("cache_hit"),
            latency_s=result.get("latency_s"),
        ), 200

    except Exception:
        logger.exception("RAG request failed")
        return jsonify(
            error="The support service is temporarily unavailable. Please try again."
        ), 500


if __name__ == "__main__":
    # Local development only. Running on port 5002 to avoid macOS AirPlay conflict on port 5000.
    app.run(host="127.0.0.1", port=5002, debug=False)
