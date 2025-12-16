# Breast Cancer Detection (WDBC)

Early and reliable breast cancer detection is critical for improving patient outcomes.  
This project implements and compares several machine‑learning models on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, wrapped in a small **MLOps-style training pipeline** and an **interactive Streamlit dashboard**.

---

## 🚀 Features

- Multiple ML models trained on the same preprocessed dataset:
  - **MLP (Keras)**
  - **SVM (RBF kernel)**
  - **Linear Regression (Ridge classifier on polynomial features)**
  - **Softmax Regression (Logistic Regression)**
  - **GRU‑SVM (GRU + hinge loss)**
- Centralized **training & export pipeline**: `training/train_and_export.py`
- Unified **configuration** via `config.yaml`
- Paper‑style metrics for each model:
  - Accuracy, TPR (Recall), TNR (Specificity), FPR, FNR
- **Streamlit dashboard** to:
  - Compare “paper vs ours”
  - Visualize metrics and deltas
  - Inspect run metadata and configs
- Reproducible runs via global random seed + deterministic options.

---

## 📂 Repository Structure

```text
.
├── app/
│   └── streamlit_app.py        # Streamlit dashboard
├── assets/
│   └── paper_metrics.json      # Reference metrics from the literature
├── config.yaml                 # All model + project configs
├── results/                    # JSON metrics per model (generated at runtime, ignored by git)
├── artifacts/                  # Trained models + scalers + meta (generated at runtime, ignored by git)
├── src/
│   ├── data/
│   │   ├── load_data.py        # WDBC CSV loader
│   │   └── preprocess.py       # Split + scaling
│   ├── evaluation/
│   │   └── metrics.py          # Accuracy, TPR, TNR, FPR, FNR
│   ├── models/                 # Model definitions (MLP, GRU‑SVM, classical models)
│   ├── preprocessing/
│   │   └── scaler.py           # Preprocessing helpers (if used)
│   └── utils/
│       ├── io.py               # JSON / IO helpers
│       └── save_results.py     # Optional result helpers
├── training/
│   └── train_and_export.py     # Main training/exports pipeline
├── run_pipeline.py             # Optional script to run end‑to‑end
└── README.md
