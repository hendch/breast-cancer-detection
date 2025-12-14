import os
import sys
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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

# ============================================================
# Reproducibility
# ============================================================
def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    try:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except:
        pass

    try:
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()
    except:
        pass


# ============================================================
# MLP
# ============================================================
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


# ============================================================
# Ensure labels {0,1}
# ============================================================
def _ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)

    if y.dtype.kind in {"U", "S", "O"}:
        return (y == "M").astype(int)

    uniq = set(np.unique(y).tolist())
    if uniq <= {0, 1}:
        return y.astype(int)

    raise ValueError(f"Unexpected labels: {uniq}")


# ============================================================
# SVM
# ============================================================
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_svm(X_train, y_train, X_test, y_test, svm_cfg):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=svm_cfg.get("kernel", "rbf"),
            C=float(svm_cfg.get("C", 1.0)),
            gamma=svm_cfg.get("gamma", "scale"),
            probability=bool(svm_cfg.get("probability", False))
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = compute_paper_metrics(y_test, preds)
    return pipeline, metrics


# ============================================================
# LINEAR REGRESSION (RIDGE CLASSIFIER)
# ============================================================
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline as SkPipeline

def train_linear_regression(X_train, y_train, X_test, y_test, lr_cfg):
    best_degree = int(lr_cfg["best_params"]["degree"])
    best_alpha = float(lr_cfg["best_params"]["alpha"])

    pipeline = SkPipeline([
        ("poly", PolynomialFeatures(degree=best_degree)),
        ("ridge", Ridge(alpha=best_alpha))
    ])

    pipeline.fit(X_train, y_train)

    preds_prob = pipeline.predict(X_test)
    preds = (preds_prob >= 0.5).astype(int)

    metrics = compute_paper_metrics(y_test, preds)
    return pipeline, metrics


# ============================================================
# SOFTMAX / LOGISTIC REGRESSION
# ============================================================
from sklearn.linear_model import LogisticRegression

def train_softmax_regression(X_train, y_train, X_test, y_test, sr_cfg):
    C = float(sr_cfg["best_params"]["C"])
    solver = sr_cfg["best_params"]["solver"]

    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=1000
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = compute_paper_metrics(y_test, preds)
    return model, metrics


# ============================================================
# GRU‑SVM
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

def train_gru_svm(X_train, y_train, X_test, y_test, cfg):

    units = int(cfg["best_params"]["units"])
    layers_n = int(cfg["best_params"]["layers"])
    dropout = float(cfg["best_params"]["dropout"])
    lr = float(cfg["best_params"]["learning_rate"])
    batch_size = int(cfg["best_params"]["batch_size"])
    epochs = int(cfg["best_params"]["epochs"])

    # reshape into sequence format
    X_train_seq = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_seq  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # convert labels to {-1,+1}
    y_train_svm = np.where(y_train == 0, -1, 1)
    y_test_svm  = np.where(y_test == 0, -1, 1)

    model = Sequential()
    model.add(GRU(
        units,
        dropout=dropout,
        recurrent_dropout=dropout,
        return_sequences=(layers_n > 1),
        input_shape=(X_train_seq.shape[1], 1)
    ))

    for _ in range(1, layers_n):
        model.add(GRU(units, dropout=dropout, recurrent_dropout=dropout))

    model.add(Dense(1))  # hinge score

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="hinge",
        metrics=["accuracy"]
    )

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_seq, y_train_svm,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[es],
        verbose=1
    )

    scores = model.predict(X_test_seq, verbose=0).ravel()
    pred_svm = np.sign(scores)
    pred_svm[scores == 0] = 1

    preds = np.where(pred_svm == -1, 0, 1)

    metrics = compute_paper_metrics(y_test, preds)
    trained_epochs = len(history.history["loss"])

    return model, metrics, trained_epochs


# ============================================================
# MAIN
# ============================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="all",
                        choices=[
                            "all",
                            "mlp",
                            "svm",
                            "linear_regression",
                            "softmax_regression",
                            "gru_svm"
                        ])
    args = parser.parse_args()
    model_choice = args.model.lower()

    # ---------- load config ----------
    with open("config.yaml") as f:
        config = yaml.safe_load(f) or {}

    seed = int(config["project"]["random_state"])
    test_size = float(config["project"]["test_size"])

    # ---------- reproducibility ----------
    set_seeds(seed)

    # ---------- folders ----------
    Path("artifacts").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # ---------- load data ----------
    X, y_raw = load_wdbc_csv("src/data/wdbc.csv")
    y = _ensure_binary_labels(y_raw)

    X_train_s, X_test_s, y_train, y_test, scaler = split_and_scale(
        X, y, test_size=test_size, random_state=seed, stratify=True
    )

    # ================================================================
    # RUN MLP
    # ================================================================
    def run_mlp():
        print("\n▶ Running MLP...")
        mlp_cfg = config["mlp"]

        model = build_keras_mlp(X_train_s.shape[1], mlp_cfg)

        es_cfg = mlp_cfg["early_stopping"]
        callbacks = []
        if es_cfg["enabled"]:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor=es_cfg["monitor"],
                    patience=int(es_cfg["patience"]),
                    restore_best_weights=bool(es_cfg["restore_best_weights"])
                )
            )

        history = model.fit(
            X_train_s, y_train,
            validation_split=float(mlp_cfg["validation_split"]),
            epochs=int(mlp_cfg["epochs"]),
            batch_size=int(mlp_cfg["batch_size"]),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        preds = (model.predict(X_test_s, verbose=0).reshape(-1) >= 0.5).astype(int)
        metrics = compute_paper_metrics(y_test, preds)
        trained_epochs = len(history.history["loss"])

        model.save("artifacts/model_mlp.keras")
        joblib.dump(scaler, "artifacts/scaler_mlp.joblib")

        payload = {
            "model": "mlp",
            "timestamp": now_iso(),
            "random_state": seed,
            "test_size": test_size,
            "trained_epochs": trained_epochs,
            "config": mlp_cfg,
            **metrics
        }

        save_json("results/latest_mlp.json", payload)
        save_json("artifacts/training_meta_mlp.json", payload)


    # ================================================================
    # RUN SVM
    # ================================================================
    def run_svm():
        print("\n▶ Running SVM...")
        cfg = config["svm"]

        model = SVC(
            C=float(cfg["C"]),
            gamma=cfg["gamma"],
            kernel="rbf",
            probability=False
        )

        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        metrics = compute_paper_metrics(y_test, preds)

        joblib.dump(model, "artifacts/model_svm.joblib")
        joblib.dump(scaler, "artifacts/scaler_svm.joblib")

        payload = {
            "model": "svm",
            "timestamp": now_iso(),
            "random_state": seed,
            "test_size": test_size,
            "trained_epochs": None,
            "config": cfg,
            **metrics
        }

        save_json("results/latest_svm.json", payload)
        save_json("artifacts/training_meta_svm.json", payload)


    # ================================================================
    # RUN LINEAR REGRESSION
    # ================================================================
    def run_linear_regression():
        print("\n▶ Running Linear Regression (Ridge)...")

        model, metrics = train_linear_regression(
            X_train_s, y_train, X_test_s, y_test,
            config["linear_regression"]
        )

        joblib.dump(model, "artifacts/model_linear_regression.joblib")
        joblib.dump(scaler, "artifacts/scaler_linear_regression.joblib")

        payload = {
            "model": "linear_regression",
            "timestamp": now_iso(),
            "random_state": seed,
            "test_size": test_size,
            "trained_epochs": None,
            "config": config["linear_regression"],
            **metrics
        }

        save_json("results/latest_linear_regression.json", payload)
        save_json("artifacts/training_meta_linear_regression.json", payload)


    # ================================================================
    # RUN SOFTMAX REGRESSION
    # ================================================================
    def run_softmax_regression():
        print("\n▶ Running Softmax Regression...")

        model, metrics = train_softmax_regression(
            X_train_s, y_train, X_test_s, y_test,
            config["softmax_regression"]
        )

        joblib.dump(model, "artifacts/model_softmax_regression.joblib")
        joblib.dump(scaler, "artifacts/scaler_softmax_regression.joblib")

        payload = {
            "model": "softmax_regression",
            "timestamp": now_iso(),
            "random_state": seed,
            "test_size": test_size,
            "trained_epochs": None,
            "config": config["softmax_regression"],
            **metrics
        }

        save_json("results/latest_softmax_regression.json", payload)
        save_json("artifacts/training_meta_softmax_regression.json", payload)


    # ================================================================
    # RUN GRU‑SVM
    # ================================================================
    def run_gru_svm():
        print("\n▶ Running GRU‑SVM...")

        cfg = config.get("gru_svm", {})
        best = cfg["best_params"]

        # Extract parameters
        units = int(best["units"])
        layers_n = int(best["layers"])
        dropout = float(best["dropout"])
        learning_rate = float(best["learning_rate"])
        batch_size = int(best["batch_size"])
        epochs = int(best["epochs"])

        # ====== Match notebook preprocessing EXACTLY ======
        # Reshape input for GRU: (samples, timesteps=features, channels=1)
        X_train_seq = X_train_s.reshape(X_train_s.shape[0], X_train_s.shape[1], 1)
        X_test_seq  = X_test_s.reshape(X_test_s.shape[0], X_test_s.shape[1], 1)

        # Map {0,1} → {-1,+1}   (Notebook did this)
        y_train_svm = np.where(y_train == 0, -1, 1)
        y_test_svm  = np.where(y_test == 0, -1, 1)

        # ===== Build GRU‑SVM model identical to notebook ======
        model = Sequential()
        model.add(GRU(
            units,
            dropout=dropout,
            recurrent_dropout=dropout,
            return_sequences=(layers_n > 1),
            input_shape=(X_train_seq.shape[1], 1)
        ))

        for _ in range(1, layers_n):
            model.add(GRU(units, dropout=dropout, recurrent_dropout=dropout))

        model.add(Dense(1))  # raw output for hinge loss

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="hinge",
            metrics=["accuracy"]
        )

        # Notebook early stopping
        es = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_seq, y_train_svm,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[es],
            verbose=1
        )

        # ===== Prediction step identical to notebook ======
        scores = model.predict(X_test_seq, verbose=0).ravel()
        y_pred_svm = np.sign(scores)
        y_pred_svm[scores == 0] = 1

        # Convert back to 0/1
        preds = np.where(y_pred_svm == -1, 0, 1)

        metrics = compute_paper_metrics(y_test, preds)
        trained_epochs = len(history.history["loss"])

        # Save artifacts
        model.save("artifacts/model_gru_svm.keras")
        joblib.dump(scaler, "artifacts/scaler_gru_svm.joblib")

        run_payload = {
            "model": "gru_svm",
            "timestamp": now_iso(),
            "random_state": seed,
            "test_size": test_size,
            "trained_epochs": trained_epochs,
            "config": cfg,
            **metrics,
        }

        save_json("results/latest_gru_svm.json", run_payload)
        save_json("artifacts/training_meta_gru_svm.json", run_payload)

        print("\nSaved:")
        print("- artifacts/model_gru_svm.keras")
        print("- artifacts/scaler_gru_svm.joblib")
        print("- results/latest_gru_svm.json")



    # ================================================================
    # DISPATCH
    # ================================================================
    if model_choice == "mlp":
        run_mlp()

    elif model_choice == "svm":
        run_svm()

    elif model_choice == "linear_regression":
        run_linear_regression()

    elif model_choice == "softmax_regression":
        run_softmax_regression()

    elif model_choice == "gru_svm":
        run_gru_svm()

    elif model_choice == "all":
        run_mlp()
        run_svm()
        run_linear_regression()
        run_softmax_regression()
        run_gru_svm()

    print("\nDONE.")


if __name__ == "__main__":
    main()
