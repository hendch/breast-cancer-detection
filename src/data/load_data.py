from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def load_wdbc_csv(path: str | Path):
    df = pd.read_csv(path)

    # Drop typical junk columns (common in Kaggle exports)
    junk_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if junk_cols:
        df = df.drop(columns=junk_cols)

    # Target column handling
    # Common case: 'diagnosis' with values 'M'/'B'
    if "diagnosis" in df.columns:
        y_raw = df["diagnosis"].astype(str).str.strip()
        y = (y_raw == "M").astype(int).to_numpy()
        df = df.drop(columns=["diagnosis"])
    else:
        # If your file already has numeric target, adjust here if needed
        raise ValueError("Expected a 'diagnosis' column (M/B) in WDBC CSV.")

    # Drop 'id' if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Keep only numeric features (defensive)
    X = df.select_dtypes(include=[np.number]).to_numpy()

    # Sanity check (WDBC should be 30 features)
    if X.shape[1] != 30:
        raise ValueError(
            f"WDBC should have 30 numeric features after cleaning, got {X.shape[1]}. "
            f"Remaining columns: {list(df.columns)}"
        )

    return X, y
