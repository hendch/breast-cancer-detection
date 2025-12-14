import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.utils.io import load_json


st.set_page_config(page_title="Breast Cancer MLP Dashboard", layout="centered")

st.title("Breast Cancer Detection – MLP")
st.caption("Paper baseline vs our run (Accuracy, TPR/TNR/FPR/FNR)")

paper = load_json("assets/paper_metrics.json")["mlp"]
current = load_json("results/latest_metrics.json")

metrics_order = ["accuracy", "tpr", "tnr", "fpr", "fnr"]

rows = []
for m in metrics_order:
    p = float(paper.get(m, 0.0))
    c = float(current.get(m, 0.0))
    rows.append({
        "metric": m.upper(),
        "paper": p,
        "ours": c,
        "delta (ours - paper)": c - p
    })

df = pd.DataFrame(rows)

st.subheader("Metrics comparison")
st.dataframe(df, use_container_width=True)

st.subheader("Bar chart")
x = range(len(metrics_order))
paper_vals = [paper[m] for m in metrics_order]
ours_vals = [current[m] for m in metrics_order]

width = 0.35

fig = plt.figure()
plt.bar([i - width/2 for i in x], paper_vals, width=width, label="paper")
plt.bar([i + width/2 for i in x], ours_vals, width=width, label="ours")
plt.xticks(list(x), [m.upper() for m in metrics_order])
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
st.pyplot(fig)

st.subheader("Run metadata")
st.json({
    "timestamp": current.get("timestamp"),
    "random_state": current.get("random_state"),
    "test_size": current.get("test_size"),
    "model_artifact": "artifacts/model.joblib",
    "scaler_artifact": "artifacts/scaler.joblib"
})
