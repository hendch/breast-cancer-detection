import numpy as np
from src.models.classical_models import *
from src.models.mlp import build_mlp
from src.models.gru_svm import build_gru_svm
from src.training.train import evaluate


def run_classical(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = (preds > 0.5).astype(int)
    return evaluate(y_test, preds)


def run_mlp(X_train, X_test, y_train, y_test, config):
    model = build_mlp(X_train.shape[1], config)
    model.fit(X_train, y_train, epochs=config["epochs"], verbose=0)
    preds = (model.predict(X_test) > 0.5).astype(int)
    return evaluate(y_test, preds)


def run_gru_svm(X_train, X_test, y_train, y_test, config):
    X_train = X_train[..., None]
    X_test = X_test[..., None]

    model = build_gru_svm(X_train.shape[1:], config)
    model.fit(
        X_train,
        2 * y_train - 1,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=0,
    )

    preds = (model.predict(X_test) > 0).astype(int)
    return evaluate(y_test, preds)
