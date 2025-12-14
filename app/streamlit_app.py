import streamlit as st
import pandas as pd
from pathlib import Path
import json

# --------------------------
# Page config + small CSS
# --------------------------
st.set_page_config(
    page_title="Breast Cancer Detection — MLP Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
      h1 { margin-bottom: 0.2rem; }
      [data-testid="stMetricValue"] { font-size: 1.6rem; }
      [data-testid="stMetricDelta"] { font-size: 0.9rem; }
      .stTabs [data-baseweb="tab-list"] button { padding-top: 6px; padding-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Helpers
# --------------------------
def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def signed_pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta*100:.2f}%"

import matplotlib.pyplot as plt
import numpy as np

def plot_paper_vs_ours(paper: dict, run: dict):
    metrics = [
        ("accuracy", "Accuracy"),
        ("tpr", "TPR (Recall)"),
        ("tnr", "TNR (Specificity)"),
        ("fpr", "FPR"),
        ("fnr", "FNR"),
    ]

    labels = [m[1] for m in metrics]
    paper_vals = [paper[m[0]] for m in metrics]
    ours_vals = [run[m[0]] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(x - width/2, paper_vals, width, label="Paper", alpha=0.8)
    ax.bar(x + width/2, ours_vals, width, label="Ours", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_ylim(0.9 if "fpr" not in labels else 0.0, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# --------------------------
# Load data
# --------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root

PAPER_PATH = ROOT / "assets" / "paper_metrics.json"
RUN_PATH = ROOT / "results" / "latest_metrics.json"

def normalize_metrics(obj: dict | list, *, top_key: str | None = None) -> dict:
    """
    Accepts:
      - dict with metrics at top-level
      - dict wrapped under a key (e.g. {"mlp": {...}})
      - list of runs (uses last element)
    Returns: dict of metrics at top-level.
    """
    # list -> last run
    if isinstance(obj, list):
        if not obj:
            raise ValueError("Metrics JSON is an empty list.")
        obj = obj[-1]

    if not isinstance(obj, dict):
        raise TypeError(f"Metrics JSON must be dict or list, got {type(obj)}")

    # unwrap under a top key (like "mlp") if present
    if top_key and top_key in obj and isinstance(obj[top_key], dict):
        obj = obj[top_key]

    return obj

paper_raw = load_json(PAPER_PATH)
run_raw = load_json(RUN_PATH)

paper = normalize_metrics(paper_raw, top_key="mlp")
run = normalize_metrics(run_raw)  # no wrapper expected; handles list case

# Optional: fail early with a clear error instead of KeyError later
required = {"accuracy", "tpr", "tnr", "fpr", "fnr"}
missing_p = required - set(paper.keys())
missing_r = required - set(run.keys())
if missing_p:
    st.error(f"paper_metrics.json missing keys: {sorted(missing_p)}. Loaded keys: {sorted(paper.keys())}")
    st.stop()
if missing_r:
    st.error(f"latest_metrics.json missing keys: {sorted(missing_r)}. Loaded keys: {sorted(run.keys())}")
    st.stop()



# Metrics keys you compare
METRICS = [
    ("Accuracy", "accuracy"),
    ("TPR (Recall)", "tpr"),
    ("TNR (Specificity)", "tnr"),
    ("FPR", "fpr"),
    ("FNR", "fnr"),
]

# Build comparison table
rows = []
for label, key in METRICS:
    p = paper.get(key, None)
    r = run.get(key, None)
    if p is None or r is None:
        continue
    rows.append(
        {
            "Metric": label,
            "Paper": p,
            "Ours": r,
            "Δ (Ours - Paper)": r - p,
        }
    )

df = pd.DataFrame(rows)

# --------------------------
# Header
# --------------------------
st.title("Breast Cancer Detection — MLP")
st.caption("Static paper metrics vs. latest deployed MLP run (WDBC).")

# --------------------------
# KPI row (top)
# --------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

paper_acc = paper["accuracy"]
ours_acc = run["accuracy"]
kpi1.metric("Accuracy", fmt_pct(ours_acc), signed_pct(ours_acc - paper_acc))

paper_fnr = paper["fnr"]
ours_fnr = run["fnr"]
kpi2.metric("FNR (Missed cancers)", fmt_pct(ours_fnr), signed_pct(ours_fnr - paper_fnr))

paper_tpr = paper["tpr"]
ours_tpr = run["tpr"]
kpi3.metric("TPR (Recall)", fmt_pct(ours_tpr), signed_pct(ours_tpr - paper_tpr))

paper_tnr = paper["tnr"]
ours_tnr = run["tnr"]
kpi4.metric("TNR (Specificity)", fmt_pct(ours_tnr), signed_pct(ours_tnr - paper_tnr))

st.divider()

# --------------------------
# Main area: Tabs (reduces scrolling)
# --------------------------
tab1, tab2, tab3 = st.tabs(["Comparison", "Plots", "Run details"])

with tab1:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Metric comparison")
        # Show formatted table
        pretty = df.copy()
        pretty["Paper"] = pretty["Paper"].map(fmt_pct)
        pretty["Ours"] = pretty["Ours"].map(fmt_pct)
        pretty["Δ (Ours - Paper)"] = pretty["Δ (Ours - Paper)"].map(signed_pct)
        st.dataframe(pretty, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Quick notes")
        st.write(
            "- Paper metrics are stored as a static reference.\n"
            "- “Ours” metrics come from the latest exported model run.\n"
            "- FNR matters clinically (missed malignant cases)."
        )

with tab2:
    st.subheader("Paper vs Ours (compact view)")

    # Native Streamlit chart (quick overview)
    plot_df = df[["Metric", "Paper", "Ours"]].set_index("Metric")
    st.bar_chart(plot_df, height=260, use_container_width=True)

    st.subheader("Delta (Ours - Paper)")
    delta_df = df[["Metric", "Δ (Ours - Paper)"]].set_index("Metric")
    st.bar_chart(delta_df, height=200, use_container_width=True)

    st.divider()

    st.subheader("Paper vs Ours — scale-aware comparison")

    import matplotlib.pyplot as plt
    import numpy as np

    # Split metrics by scale
    high_metrics = ["Accuracy", "TPR (Recall)", "TNR (Specificity)"]
    low_metrics = ["FPR", "FNR"]

    high_df = df[df["Metric"].isin(high_metrics)]
    low_df = df[df["Metric"].isin(low_metrics)]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(8, 3),   # compact figure
        constrained_layout=True
    )

    # High-rate metrics
    x = np.arange(len(high_df))
    axes[0].bar(x - 0.15, high_df["Paper"], width=0.3, label="Paper")
    axes[0].bar(x + 0.15, high_df["Ours"], width=0.3, label="Ours")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(high_df["Metric"], rotation=20)
    axes[0].set_ylim(0.94, 1.0)
    axes[0].set_title("Performance metrics")
    axes[0].legend(fontsize=8)

    # Error metrics
    x = np.arange(len(low_df))
    axes[1].bar(x - 0.15, low_df["Paper"], width=0.3, label="Paper")
    axes[1].bar(x + 0.15, low_df["Ours"], width=0.3, label="Ours")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(low_df["Metric"])
    axes[1].set_ylim(
        0,
        max(low_df["Paper"].max(), low_df["Ours"].max()) * 1.4
    )
    axes[1].set_title("Error metrics (lower is better)")

    st.pyplot(fig)


with tab3:
    st.subheader("Run metadata")

    meta_cols = st.columns(3)
    meta_cols[0].write(f"**Timestamp:** {run.get('timestamp', '-')}")
    meta_cols[1].write(f"**Random state:** {run.get('random_state', '-')}")
    meta_cols[2].write(f"**Trained epochs:** {run.get('trained_epochs', '-')}")

    with st.expander("Show full config"):
        st.json(run.get("config", {}))

    with st.expander("Show raw JSON"):
        st.json(run)
