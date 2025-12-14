import yaml
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_wdbc
from src.models.classical_models import knn_model, svm_model
from src.utils.cv_compare_plot import plot_cv_paper_vs_tuned


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f) or {}

    # Load full dataset (no train/test split here)
    X, y = load_wdbc()

    # Scale once for CV (same preprocessing you use in pipeline)
    if config.get("data", {}).get("scale", True):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.values  # convert DataFrame to numpy for sklearn splits

    folds = config.get("evaluation", {}).get("cv_folds", 5)
    shuffle = config.get("evaluation", {}).get("shuffle", True)
    seed = config.get("project", {}).get("random_state", 42)

    paper_cfg = config.get("models", {})
    tuned_cfg = config.get("tuned", {})

    model_names = ["knn", "svm"]

    builders_paper = {
        "knn": lambda: knn_model(paper_cfg["knn"]),
        "svm": lambda: svm_model(paper_cfg["svm"]),
    }

    builders_tuned = {
        "knn": lambda: knn_model({**paper_cfg["knn"], **tuned_cfg.get("knn", {})}),
        "svm": lambda: svm_model({**paper_cfg["svm"], **tuned_cfg.get("svm", {})}),
    }

    plot_cv_paper_vs_tuned(
        model_names=model_names,
        builders_paper=builders_paper,
        builders_tuned=builders_tuned,
        X=X,
        y=y,
        folds=folds,
        shuffle=shuffle,
        random_state=seed,
        output_path="results/plots/cv_paper_vs_tuned_metrics.png",
    )


if __name__ == "__main__":
    main()
