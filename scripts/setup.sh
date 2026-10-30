#!/usr/bin/env bash
set -euo pipefail

echo "==> Starting Qdrant + Redis"
docker compose up -d

echo "==> Pulling Ollama models (set LLM_MODEL in .env to change)"
ollama pull "${LLM_MODEL:-qwen3:8b}" || echo "Install Ollama first: https://ollama.com"
ollama pull llama3.1:8b || true

echo "==> Installing Python deps"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

echo "==> Done. Run: streamlit run app.py"
