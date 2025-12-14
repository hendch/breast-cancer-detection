import streamlit as st
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Page config + small CSS
# --------------------------
st.set_page_config(
    page_title="Breast Cancer Detection — Models Dashboard",
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

def plot_scale_aware(df):
    high_metrics = ["Accuracy", "TPR (Recall)", "TNR (Specificity)"]
    low_metrics = ["FPR", "FNR"]

    high_df = df[df["Metric"].isin(high_metrics)]
    low_df = df[df["Metric"].isin(low_metrics)]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(8, 3),
        constrained_layout=True
    )

    # -------------------------
    # High metrics
    # -------------------------
    x = np.arange(len(high_df))
    axes[0].bar(x - 0.15, high_df["Paper"], width=0.3, label="Paper")
    axes[0].bar(x + 0.15, high_df["Ours"], width=0.3, label="Ours")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(high_df["Metric"], rotation=20)
    axes[0].set_title("Performance metrics")
    axes[0].legend(fontsize=8)

    # Dynamic y-scale
    min_val = min(high_df["Paper"].min(), high_df["Ours"].min())
    max_val = max(high_df["Paper"].max(), high_df["Ours"].max())
    lower = max(0.0, min_val - 0.03)
    upper = min(1.0, max_val + 0.01)
    axes[0].set_ylim(lower, upper)

    # -------------------------
    # Low metrics
    # -------------------------
    x = np.arange(len(low_df))
    axes[1].bar(x - 0.15, low_df["Paper"], width=0.3, label="Paper")
    axes[1].bar(x + 0.15, low_df["Ours"], width=0.3, label="Ours")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(low_df["Metric"])
    axes[1].set_title("Error metrics (lower is better)")
    axes[1].set_ylim(0, max(low_df["Paper"].max(), low_df["Ours"].max()) * 1.4)

    return fig

# --------------------------
# Load paper + runtime
# --------------------------
ROOT = Path(__file__).resolve().parents[1]

PAPER_PATH = ROOT / "assets" / "paper_metrics.json"
RESULTS_DIR = ROOT / "results"

paper_all = load_json(PAPER_PATH)

# Detect available models
AVAILABLE = {
    "mlp": "latest_mlp.json",
    "svm": "latest_svm.json",
    "linear_regression": "latest_linear_regression.json",
    "softmax_regression": "latest_softmax_regression.json",
    "gru_svm": "latest_gru_svm.json"
}

available_models = [
    model for model, file in AVAILABLE.items()
    if (RESULTS_DIR / file).exists()
]

if not available_models:
    st.error("No trained model results found in /results. Run training first.")
    st.stop()

st.title("Breast Cancer Detection — Model Comparison Dashboard")
st.caption("Dynamically showing models with available results.")

model_tabs = st.tabs(available_models)

# --------------------------
# Loop over models
# --------------------------
for model_name, tab in zip(available_models, model_tabs):

    with tab:
        st.header(model_name.upper())

        # ----- Load paper metrics -----
        if model_name not in paper_all:
            st.error(f"paper_metrics.json has no entry for '{model_name}'.")
            st.stop()

        paper = paper_all[model_name]

        # ----- Load results -----
        run_path = RESULTS_DIR / AVAILABLE[model_name]
        run = load_json(run_path)

        # ----- Build comparison table -----
        METRICS = [
            ("Accuracy", "accuracy"),
            ("TPR (Recall)", "tpr"),
            ("TNR (Specificity)", "tnr"),
            ("FPR", "fpr"),
            ("FNR", "fnr"),
        ]

        rows = []
        for lbl, key in METRICS:
            rows.append(
                {
                    "Metric": lbl,
                    "Paper": paper[key],
                    "Ours": run[key],
                    "Δ (Ours - Paper)": run[key] - paper[key],
                }
            )
        df = pd.DataFrame(rows)

        # --------------------------
        # KPI Row
        # --------------------------
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", fmt_pct(run["accuracy"]), signed_pct(run["accuracy"] - paper["accuracy"]))
        k2.metric("FNR", fmt_pct(run["fnr"]), signed_pct(run["fnr"] - paper["fnr"]))
        k3.metric("TPR", fmt_pct(run["tpr"]), signed_pct(run["tpr"] - paper["tpr"]))
        k4.metric("TNR", fmt_pct(run["tnr"]), signed_pct(run["tnr"] - paper["tnr"]))

        st.divider()

        # --------------------------
        # Sub-tabs inside each model
        # --------------------------
        comp, plots, meta = st.tabs(["Comparison", "Plots", "Run details"])

        # ===== COMPARISON TABLE =====
        with comp:
            st.subheader("Metric comparison")
            pretty = df.copy()
            pretty["Paper"] = pretty["Paper"].map(fmt_pct)
            pretty["Ours"] = pretty["Ours"].map(fmt_pct)
            pretty["Δ (Ours - Paper)"] = pretty["Δ (Ours - Paper)"].map(signed_pct)

            st.dataframe(pretty, use_container_width=True, hide_index=True)

        # ===== PLOTS =====
        with plots:
            st.subheader("Paper vs Ours (simple view)")
            st.bar_chart(
                df[["Metric", "Paper", "Ours"]].set_index("Metric"),
                height=260,
                use_container_width=True
            )

            st.subheader("Delta (Ours - Paper)")
            st.bar_chart(
                df[["Metric", "Δ (Ours - Paper)"]].set_index("Metric"),
                height=200,
                use_container_width=True
            )

            st.divider()

            st.subheader("Scale-aware comparison")
            fig = plot_scale_aware(df)
            st.pyplot(fig)

        # ===== METADATA =====
        with meta:
            st.subheader("Run metadata")

            c1, c2, c3 = st.columns(3)
            c1.write(f"**Timestamp:** {run.get('timestamp', '-')}")
            c2.write(f"**Random state:** {run.get('random_state', '-')}")
            c3.write(f"**Trained epochs:** {run.get('trained_epochs', '-')}")

            with st.expander("Show full config"):
                st.json(run.get("config", {}))

            with st.expander("Show raw JSON"):
                st.json(run)
