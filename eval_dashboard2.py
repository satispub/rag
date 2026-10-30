# scripts/eval_dashboard.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
st.title("📊 RAGAS Evaluation Dashboard")

results_path = Path("eval/results.csv")
if results_path.exists():
    df = pd.read_csv(results_path)
    
    st.subheader("Metric Summaries")
    col1, col2, col3 = st.columns(3)
    for i, col in enumerate([col1, col2, col3]):
        if i < len(df.columns):
            metric_name = df.columns[i]
            if pd.api.types.is_numeric_dtype(df[metric_name]):
                col.metric(label=metric_name.replace("_", " ").title(), value=round(df[metric_name].mean(), 3))

    st.subheader("Detailed Evaluation Results")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No evaluation results found. Run `python scripts/evaluate.py` first!")
