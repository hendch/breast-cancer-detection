import os
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import joblib
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from src.data.load_data import load_wdbc_csv
from src.data.preprocess import split_and_scale
from src.evaluation.metrics import compute_paper_metrics
from src.utils.io import save_json, now_iso


def set_seeds(seed: int) -> None:
    """Make runs as reproducible as reasonably possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    # Best-effort determinism (may not be available on all TF builds)
    try:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except Exception:
        pass

    try:
        import tensorflow as tf  # local import
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Not critical; continue without hard determinism
        pass


def build_keras_mlp(input_dim: int, cfg: dict):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    dropout = float(cfg["dropout"])
    for units in cfg["hidden_layers"]:
        model.add(layers.Dense(int(units), activation="relu"))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"]))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    """
    Ensure labels are {0,1} with Malignant=1, Benign=0.
    If your load_wdbc_csv already does this, this will just validate it.
    """
    y = np.asarray(y)

    # If string labels exist, enforce mapping explicitly
    if y.dtype.kind in {"U", "S", "O"}:
        # Common WDBC: 'M' and 'B'
        y01 = (y == "M").astype(int)
        return y01

    # Numeric labels: validate they are binary
    uniq = set(np.unique(y).tolist())
    if uniq <= {0, 1}:
        return y.astype(int)

    raise ValueError(f"Labels are not binary and not (B/M). Found unique values: {sorted(uniq)}")


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f) or {}

    project_cfg = config.get("project", {})
    seed = int(project_cfg.get("random_state", 42))
    test_size = float(project_cfg.get("test_size", 0.2))

    # Deterministic-ish run
    set_seeds(seed)

    # Ensure output dirs exist
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    # Load dataset (make sure this is the SAME CSV you use in the notebook)
    X, y_raw = load_wdbc_csv("src/data/wdbc.csv")
    y = _ensure_binary_labels(y_raw)

    # IMPORTANT: do scaling only once. This function should fit scaler on TRAIN ONLY.
    X_train_s, X_test_s, y_train, y_test, scaler = split_and_scale(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=True,  # keep class proportions consistent (recommended)
    )

    mlp_cfg = config.get("mlp", {})
    model = build_keras_mlp(X_train_s.shape[1], mlp_cfg)

    callbacks = []
    es_cfg = mlp_cfg.get("early_stopping", {})

    if es_cfg.get("enabled", True):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=int(es_cfg.get("patience", 20)),  # MATCH NOTEBOOK DEFAULT
                restore_best_weights=bool(es_cfg.get("restore_best_weights", True)),
            )
        )

    history = model.fit(
        X_train_s,
        y_train,
        validation_split=float(mlp_cfg.get("validation_split", 0.1)),
        epochs=int(mlp_cfg.get("epochs", 300)),
        batch_size=int(mlp_cfg.get("batch_size", 16)),
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    # Evaluate paper-style metrics on test split
    probs = model.predict(X_test_s, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    metrics = compute_paper_metrics(y_test, preds)

    # Save artifacts
    model.save("artifacts/model.keras")
    joblib.dump(scaler, "artifacts/scaler.joblib")

    trained_epochs = len(history.history.get("loss", []))

    run_payload = {
        "model": "mlp",
        "timestamp": now_iso(),
        "random_state": seed,
        "test_size": test_size,
        "trained_epochs": trained_epochs,
        "config": {
            "hidden_layers": mlp_cfg.get("hidden_layers", [256, 256, 256]),
            "dropout": mlp_cfg.get("dropout", 0.3),
            "learning_rate": mlp_cfg.get("learning_rate", 0.0001),
            "batch_size": mlp_cfg.get("batch_size", 16),
            "epochs_max": mlp_cfg.get("epochs", 300),
            "validation_split": mlp_cfg.get("validation_split", 0.1),
            "early_stopping": {
                "enabled": es_cfg.get("enabled", True),
                "monitor": es_cfg.get("monitor", "val_loss"),
                "patience": int(es_cfg.get("patience", 20)),
                "restore_best_weights": bool(es_cfg.get("restore_best_weights", True)),
            },
        },
        **metrics,
    }

    save_json("results/latest_metrics.json", run_payload)
    save_json("artifacts/training_meta.json", run_payload)

    print("Saved:")
    print("- artifacts/model.keras")
    print("- artifacts/scaler.joblib")
    print("- results/latest_metrics.json")
    print("- artifacts/training_meta.json")


if __name__ == "__main__":
    main()
