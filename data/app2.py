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
# Set JIRA_USER_EMAIL and JIRA_API_TOKEN in your environment; never hard-code secrets.
JIRA_CLOUD_URL = os.getenv("JIRA_CLOUD_URL", "https://rdnodelegate.atlassian.net").rstrip("/")
JIRA_USER_EMAIL = os.getenv("JIRA_USER_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "KAN")
JIRA_ISSUE_TYPE = os.getenv("JIRA_ISSUE_TYPE", "Task")
JIRA_TIMEOUT_SECONDS = int(os.getenv("JIRA_TIMEOUT_SECONDS", "20"))

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
    """Create a real Jira issue from the chat UI."""
    data = request.get_json(silent=True) or {}

    if not JIRA_USER_EMAIL or not JIRA_API_TOKEN:
        app.logger.error("Jira credentials are not configured.")
        return jsonify({
            "status": "error",
            "message": "Ticket creation is not configured. Please contact support."
        }), 503

    business_id = str(data.get("business_id") or "N/A").strip()[:200]
    user_id = str(data.get("user_id") or "N/A").strip()[:200]
    delegate_id = str(data.get("delegate_id") or "N/A").strip()[:200]
    error_message = str(
        data.get("error_message") or "Delegate enrollment or removal failure"
    ).strip()[:2000]
    details = str(
        data.get("details") or "User requested automated ticket creation via chat assistant."
    ).strip()[:5000]

    description_text = (
        "Automated escalation from Secure Banking Support RAG Chatbot.\n\n"
        f"Business Name / ID: {business_id}\n"
        f"User Name / ID: {user_id}\n"
        f"Delegate Name / ID: {delegate_id}\n"
        f"Error Message: {error_message}\n"
        f"Additional Information: {details}"
    )
    adf_content = []
    for line in description_text.split("\n"):
        paragraph = {"type": "paragraph", "content": []}
        if line:
            paragraph["content"].append({"type": "text", "text": line})
        adf_content.append(paragraph)

    payload = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": f"Delegate Failure Support Ticket: Business {business_id}"[:255],
            "description": {
                "type": "doc",
                "version": 1,
                "content": adf_content
            },
            "issuetype": {"name": JIRA_ISSUE_TYPE}
        }
    }

    url = f"{JIRA_CLOUD_URL}/rest/api/3/issue"
    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            auth=HTTPBasicAuth(JIRA_USER_EMAIL, JIRA_API_TOKEN),
            timeout=JIRA_TIMEOUT_SECONDS
        )

        if response.status_code == 201:
            issue_data = response.json()
            ticket_key = issue_data["key"]
            ticket_url = f"{JIRA_CLOUD_URL}/browse/{ticket_key}"
            log_ticket_event(
                ticket_key, ticket_url, business_id, user_id,
                delegate_id, error_message, details
            )
            app.logger.info("Created Jira issue %s", ticket_key)
            return jsonify({
                "status": "success",
                "ticket_key": ticket_key,
                "ticket_url": ticket_url,
                "message": f"Successfully created Jira ticket {ticket_key}."
            }), 201

        app.logger.warning(
            "Jira issue creation failed with HTTP %s: %s",
            response.status_code, response.text[:1000]
        )
        if response.status_code in (401, 403):
            return jsonify({
                "status": "error",
                "message": "Jira authentication or project permissions prevented ticket creation."
            }), 502
        if response.status_code == 400:
            return jsonify({
                "status": "error",
                "message": "Jira rejected the ticket fields. Verify the project, issue type, and required fields."
            }), 400
        return jsonify({
            "status": "error",
            "message": f"Jira ticket creation failed (HTTP {response.status_code})."
        }), 502

    except requests.Timeout:
        app.logger.exception("Timed out while creating Jira issue.")
        return jsonify({
            "status": "error",
            "message": "Jira did not respond in time. Please retry in a moment."
        }), 504
    except requests.RequestException:
        app.logger.exception("Network error while creating Jira issue.")
        return jsonify({
            "status": "error",
            "message": "Could not connect to Jira. Please try again later."
        }), 502
    except Exception:
        app.logger.exception("Unexpected error while creating Jira issue.")
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred while creating the ticket."
        }), 500


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
    app.run(host="0.0.0.0", port=5002, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
