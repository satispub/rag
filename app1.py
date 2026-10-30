import os
import time
import uuid
import json
import random
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
from flask import Flask, render_template, request, jsonify

from src.config import Settings
from src.rag import RAG

# Change this line in app.py:
# aapp = Flask(__name__, template_folder="templates", static_folder=".")

# To standard default initialization:
app = Flask(__name__)

# Initialize RAG Settings
settings = Settings()
rag = RAG(settings)

# Jira Cloud Configuration
JIRA_CLOUD_URL = "https://rdnodelegate.atlassian.net"
JIRA_USER_EMAIL = os.getenv("JIRA_USER_EMAIL", "athakhan.mohammed@gmail.com")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "ATCTT3xFfGN0C9NW3hjW80QheXdM8zEApcnTlV2IsY7uvpXLjeDa0DIKurOUK5mnKHx7SbJ92Of-9mdWenvStpT_lyXOq-jp7I9WflDlYECAkCu0z7ZsmszRgbvYM0gnGouUEFQvHkCs7z2zAIqfzWIjdV0o6Y-C9Xr7lVNP36iL_Trtpf8HtG8=1453FAD3")

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        # Run RAG pipeline query
        res = rag.ask(message)
        answer = res.get("answer", "I couldn't find an answer to that question.")
        
        return jsonify({
            "status": "success",
            "answer": answer,
            "latency_s": res.get("latency_s", 0),
            "sources": res.get("sources", [])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/create-ticket", methods=["POST"])
def create_ticket():
    data = request.get_json() or {}
    
    business_id = data.get("business_id", "N/A")
    user_id = data.get("user_id", "N/A")
    delegate_id = data.get("delegate_id", "N/A")
    error_message = data.get("error_message", "Delegate enrollment or removal failure")
    details = data.get("details", "User requested automated ticket creation via chat assistant.")
    
    # Generate local high-reliability ticket key
    ticket_num = random.randint(1000, 9999)
    ticket_key = f"TICK-{ticket_num}"
    ticket_url = f"/support/tickets/{ticket_key}"
    
    try:
        # Log structured ticket event into logs/traces.jsonl for Streamlit dashboard metrics
        log_ticket_event(ticket_key, ticket_url, business_id, user_id, delegate_id, error_message, details)
        
        return jsonify({
            "status": "success",
            "ticket_key": ticket_key,
            "ticket_url": ticket_url,
            "message": f"Successfully created local support ticket {ticket_key}."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500






def log_ticket_event(ticket_key, ticket_url, b_id, u_id, d_id, err, details):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    event_entry = {
        "event": "ticket_created",
        "run_id": str(uuid.uuid4()),
        "trace_id": "jira_integration_action",
        "ticket_payload": {
            "ticket_key": ticket_key,
            "ticket_url": ticket_url,
            "business_id": b_id,
            "user_id": u_id,
            "delegate_id": d_id,
            "error_message": err,
            "details": details
        },
        "ts": time.time()
    }
    with open(log_dir / "traces.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event_entry) + "\n")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
