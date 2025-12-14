import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def _cv_scores(model_builder, X, y, folds, shuffle, random_state):
    """
    model_builder: function that returns a NEW sklearn model instance
    Returns dict: metric -> list[fold_score]
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)

    scores = {m: [] for m in ["accuracy", "precision", "recall", "f1"]}

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx] if hasattr(y, "iloc") else (y[train_idx], y[test_idx])

        model = model_builder()
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        # some models output probabilities; keep it safe
        if preds.dtype != int and preds.dtype != np.int64 and preds.dtype != np.int32:
            preds = (preds > 0.5).astype(int)

        m = _compute_metrics(y_te, preds)
        for k in scores:
            scores[k].append(m[k])

    return scores


def plot_cv_paper_vs_tuned(
    model_names,
    builders_paper,
    builders_tuned,
    X,
    y,
    folds=5,
    shuffle=True,
    random_state=42,
    output_path="results/plots/cv_paper_vs_tuned_metrics.png"
):
    """
    model_names: list[str]
    builders_paper: dict[str, callable returning sklearn model]
    builders_tuned: dict[str, callable returning sklearn model]
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1"]

    # For each model we compute mean±std over folds
    paper_means = {m: [] for m in metrics}
    paper_stds = {m: [] for m in metrics}
    tuned_means = {m: [] for m in metrics}
    tuned_stds = {m: [] for m in metrics}

    for name in model_names:
        paper_scores = _cv_scores(builders_paper[name], X, y, folds, shuffle, random_state)
        tuned_scores = _cv_scores(builders_tuned[name], X, y, folds, shuffle, random_state)

        for m in metrics:
            paper_means[m].append(float(np.mean(paper_scores[m])))
            paper_stds[m].append(float(np.std(paper_scores[m])))
            tuned_means[m].append(float(np.mean(tuned_scores[m])))
            tuned_stds[m].append(float(np.std(tuned_scores[m])))

    x = np.arange(len(model_names))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        p_mean = np.array(paper_means[metric])
        p_std = np.array(paper_stds[metric])
        t_mean = np.array(tuned_means[metric])
        t_std = np.array(tuned_stds[metric])

        # Learning-curve style: line + marker + shaded band
        ax.plot(x, p_mean, marker="o", label="paper")
        ax.fill_between(x, p_mean - p_std, p_mean + p_std, alpha=0.2)

        ax.plot(x, t_mean, marker="o", label="tuned")
        ax.fill_between(x, t_mean - t_std, t_mean + t_std, alpha=0.2)

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        # Dynamic y-axis scaling per metric
        all_vals = np.concatenate([
            p_mean - p_std,
            p_mean + p_std,
            t_mean - t_std,
            t_mean + t_std,
        ])

        y_min = max(0.0, all_vals.min() - 0.005)
        y_max = min(1.0, all_vals.max() + 0.005)

        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Paper vs Tuned (mean ± std over CV folds)")
    axes[0].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()

    print(f"Saved: {output_path}")
