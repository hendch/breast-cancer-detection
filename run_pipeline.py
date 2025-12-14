import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import joblib
import numpy as np

from src.data.load_data import load_wdbc_csv
from src.data.preprocess import split_and_scale
from src.preprocessing.scaler import fit_scaler, transform
from src.models.mlp import build_keras_mlp
from src.evaluation.metrics import compute_paper_metrics
from src.utils.io import save_json, now_iso


def full_pipeline():
    with open("config.yaml") as f:
        config = yaml.safe_load(f) or {}

    X, y = load_wdbc_csv("src/data/wdbc.csv")

    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X,
        y,
        test_size=config["project"]["test_size"],
        random_state=config["project"]["random_state"]
    )

    scaler = fit_scaler(X_train)
    X_train_s = transform(scaler, X_train)
    X_test_s = transform(scaler, X_test)

    mlp_cfg = config["mlp"]
    model = build_keras_mlp(X_train_s.shape[1], mlp_cfg)

    history = model.fit(
        X_train_s,
        y_train,
        validation_split=float(mlp_cfg.get("validation_split", 0.1)),
        epochs=int(mlp_cfg["epochs"]),
        batch_size=int(mlp_cfg["batch_size"]),
        verbose=1,
    )

    probs = model.predict(X_test_s).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    metrics = compute_paper_metrics(y_test, preds)

    model.save("artifacts/model.keras")
    joblib.dump(scaler, "artifacts/scaler.joblib")

    trained_epochs = len(history.history.get("loss", []))

    run_payload = {
        "model": "mlp",
        "timestamp": now_iso(),
        "trained_epochs": trained_epochs,
        **config["project"],
        "config": mlp_cfg,
        **metrics
    }

    save_json("results/latest_metrics.json", run_payload)

    print("Pipeline done.")
    print("Saved artifacts + metrics.")
    return run_payload


if __name__ == "__main__":
    full_pipeline()
