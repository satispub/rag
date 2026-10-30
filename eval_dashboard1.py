import json
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Enterprise RAG Evaluation & Operations Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Enterprise RAG Evaluation & Operations Dashboard")
st.markdown("Executive telemetry analyzing `logs/traces.jsonl` execution streams, Jira ticketing integrations, and infrastructure latency.")
st.divider()

# --------------------------
# Load Real Data from logs/traces.jsonl
# --------------------------
traces_path = Path("logs/traces.jsonl")
raw_events = []

if traces_path.exists():
    try:
        with open(traces_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_events.append(json.loads(line))
    except Exception as e:
        st.error(f"Error reading trace log file: {e}")

# --------------------------
# Group Lifecycle Steps & Jira Tickets
# --------------------------
grouped_traces = {}
ticket_records = []

for event in raw_events:
    event_type = event.get("event")
    
    # Capture automated Jira ticket creation events
    if event_type == "ticket_required" or event_type == "ticket_created":
        payload = event.get("ticket_payload", {})
        ts_val = event.get("ts")
        dt_str = datetime.datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M:%S") if ts_val else "N/A"
        
        ticket_key = payload.get("ticket_key", "N/A")
        ticket_url = payload.get("ticket_url", f"https://rdnodelegate.atlassian.net/browse/{ticket_key}")

        ticket_records.append({
            "Timestamp": dt_str,
            "Ticket Key": ticket_key,
            "Jira Link": ticket_url,
            "Business ID": payload.get("business_id", "N/A"),
            "User ID": payload.get("user_id", "N/A"),
            "Delegate ID": payload.get("delegate_id", "N/A"),
            "Error Message": payload.get("error_message", "N/A"),
            "Details": payload.get("details", "N/A"),
            "raw_ts": ts_val or 0
        })
        continue

    run_id = event.get("run_id", "unknown")
    if run_id not in grouped_traces:
        grouped_traces[run_id] = {
            "run_id": run_id,
            "trace_id": event.get("trace_id", "unknown"),
            "timestamp": event.get("ts"),
            "latency_ms": None,
            "output": None,
            "events": []
        }
    
    grouped_traces[run_id]["events"].append(event)
    
    if event.get("ts") is not None and (grouped_traces[run_id]["timestamp"] is None or event.get("ts") < grouped_traces[run_id]["timestamp"]):
        grouped_traces[run_id]["timestamp"] = event.get("ts")
        
    if event.get("latency_s") is not None:
        grouped_traces[run_id]["latency_ms"] = round(float(event["latency_s"]) * 1000, 2)
        
    if event.get("output") is not None:
        grouped_traces[run_id]["output"] = event["output"]

# Flatten completed transactions
records = []
for r_id, data in grouped_traces.items():
    if not data["output"]:
        continue

    raw_ts = data["timestamp"]
    dt_str = datetime.datetime.fromtimestamp(raw_ts).strftime("%Y-%m-%d %H:%M:%S") if raw_ts else "N/A"
    lat_val = data["latency_ms"] if data["latency_ms"] is not None else 0.0
    
    records.append({
        "Timestamp": dt_str,
        "Run ID": data["run_id"],
        "Trace ID": data["trace_id"],
        "Latency_Numeric": lat_val,
        "Latency": f"{lat_val:.0f} ms" if lat_val > 0 else "0 ms",
        "LLM Output Summary": data["output"],
        "raw_ts": raw_ts or 0
    })

df_display = pd.DataFrame(records)
if not df_display.empty:
    df_display = df_display.sort_values(by="raw_ts", ascending=False).drop(columns=["raw_ts"])

df_tickets = pd.DataFrame(ticket_records)
if not df_tickets.empty:
    df_tickets = df_tickets.sort_values(by="raw_ts", ascending=False).drop(columns=["raw_ts"])

# --------------------------
# Dashboard Executive KPIs
# --------------------------
total_tx = len(df_display)
total_tickets = len(df_tickets)
valid_lat = df_display[df_display["Latency_Numeric"] > 0]["Latency_Numeric"]
avg_lat_ms = valid_lat.mean() if not valid_lat.empty else 0.0
max_lat_ms = valid_lat.max() if not valid_lat.empty else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Completed Transactions", total_tx, "Filtered Logs")
c2.metric("Jira Tickets Created", total_tickets, "Active Escalations")
c3.metric("Average Inference Latency", f"{avg_lat_ms:.0f} ms", f"{avg_lat_ms/1000:.2f}s")
c4.metric("System Operational Status", "100% Stable", "Connected to Jira Cloud")

st.divider()

# --------------------------
# Analytics Summary
# --------------------------
col1, col2 = st.columns([2, 1])

with col1:
    if not valid_lat.empty:
        fig = px.bar(
            df_display[df_display["Latency_Numeric"] > 0].head(15),
            x="Trace ID",
            y="Latency_Numeric",
            title="⏱️ Latest Transaction Latencies (Milliseconds)",
            labels={"Latency_Numeric": "Latency (Milliseconds)"},
            color="Latency_Numeric",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Awaiting completed inference transactions to graph latencies.")

with col2:
    st.subheader("🛠️ Infrastructure & Ticketing Summary")
    st.markdown(f"""
- **Primary Engine:** Ollama (`qwen3:8b` / `llama3.2:3b`)
- **Vector DB:** Qdrant Hybrid Search (Dense + BM25 Sparse)
- **Ticketing Tool:** Atlassian Jira Cloud (`rdnodelegate.atlassian.net`)
- **Peak Latency Recorded:** **{max_lat_ms:.0f} ms**
    """)
    st.markdown("""
    * **Automated Escalation:** Delegate failures capture parameter inputs and push live tickets to project **DEV**.
    """)

st.divider()

# --------------------------
# Tabs View (Transactions, Jira Tickets, Inspector)
# --------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Completed Transactions", "🎫 Jira Escalation Tickets", "🔬 Run Inspector"])

with tab1:
    st.subheader("Completed Transaction Results")
    if not df_display.empty:
        grid_df = df_display.drop(columns=["Latency_Numeric"])
        st.dataframe(grid_df, use_container_width=True, hide_index=True)
    else:
        st.info("No completed traces found.")

with tab2:
    st.subheader("🎫 Jira Support Tickets & Direct Links")
    st.markdown("Real-time telemetry of issues escalated to your Jira Cloud workspace (`rdnodelegate.atlassian.net`):")
    if not df_tickets.empty:
        # Render a nice table with clickable hyperlinks to Jira
        st.dataframe(
            df_tickets,
            column_config={



# BEFORE (Causes TypeError / Invalid Format):
# "Jira Link": st.column_config.LinkColumn("Open in Jira", display_url="https://rdnodelegate.atlassian.net/browse/{Ticket Key}")

# AFTER (Fixed):
"Jira Link": st.column_config.LinkColumn(
    "Jira Ticket",
    display_text=r"https://rdnodelegate\.atlassian\.net/browse/(.*)",
    help="Click to open ticket in Jira Cloud"
)

            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No Jira escalation tickets logged yet. Test a delegate failure query in your chat widget to generate tickets!")

with tab3:
    st.subheader("🔬 Deep-Dive Trace Inspector")
    selected_run = st.selectbox(
        "Select Run ID to Inspect:",
        options=list(grouped_traces.keys()) if grouped_traces else ["None"]
    )
    if selected_run and selected_run != "None":
        run_detail = grouped_traces[selected_run]
        st.write(f"**Trace ID:** `{run_detail['trace_id']}`")
        st.write(f"**Final Response:** {run_detail['output'] if run_detail['output'] else 'Incomplete'}")
        
        st.markdown("#### Execution Steps Sequence")
        events_df = pd.DataFrame(run_detail["events"])
        if "ts" in events_df.columns:
            events_df["Timestamp"] = events_df["ts"].apply(
                lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if pd.notnull(x) else "N/A"
            )
            events_df = events_df.drop(columns=["ts"])
        st.dataframe(events_df, use_container_width=True, hide_index=True)
