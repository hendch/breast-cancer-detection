import yaml

from src.data.load_data import load_wdbc
from src.data.preprocess import preprocess_data
from src.pipeline.run_models import *
from src.utils.save_results import save_results
from src.utils.save_comparison import save_comparison


def main():
    # ---- load config FIRST ----
    with open("config.yaml") as f:
        config = yaml.safe_load(f) or {}

    # ---- extract config sections ----
    project_cfg = config.get("project", {})
    data_cfg = config.get("data", {})
    run_cfg = config.get("run", {})
    paper_cfg = config.get("models", {})
    tuned_cfg = config.get("tuned", {})

    compare = run_cfg.get("compare_with_tuned", False)

    # ---- data ----
    X, y = load_wdbc()
    X_train, X_test, y_train, y_test = preprocess_data(
        X,
        y,
        project_cfg.get("test_size", 0.2),
        project_cfg.get("random_state", 42),
        data_cfg.get("scale", True),
    )

    run_mode = run_cfg.get("mode", "paper")
    models_to_run = run_cfg.get("models", {}).get(run_mode, [])

    paper_results = {}
    tuned_results = {}

    # ---- model loop ----
    for model_name in models_to_run:
        print(f"Running paper: {model_name}")

        if model_name == "knn":
            paper_results[model_name] = run_classical(
                knn_model(paper_cfg["knn"]),
                X_train, X_test, y_train, y_test
            )

        elif model_name == "svm":
            paper_results[model_name] = run_classical(
                svm_model(paper_cfg["svm"]),
                X_train, X_test, y_train, y_test
            )

        elif model_name == "mlp":
            paper_results[model_name] = run_mlp(
                X_train, X_test, y_train, y_test,
                paper_cfg["mlp"]
            )

        elif model_name == "gru_svm":
            paper_results[model_name] = run_gru_svm(
                X_train, X_test, y_train, y_test,
                paper_cfg["gru_svm"]
            )

        # ---- tuned run ----
        if compare and model_name in tuned_cfg:
            print(f"Running tuned: {model_name}")

            tuned_params = dict(paper_cfg.get(model_name, {}))
            tuned_params.update(tuned_cfg[model_name])

            if model_name == "knn":
                tuned_results[model_name] = run_classical(
                    knn_model(tuned_params),
                    X_train, X_test, y_train, y_test
                )

            elif model_name == "svm":
                tuned_results[model_name] = run_classical(
                    svm_model(tuned_params),
                    X_train, X_test, y_train, y_test
                )

            elif model_name == "mlp":
                tuned_results[model_name] = run_mlp(
                    X_train, X_test, y_train, y_test,
                    tuned_params
                )

            elif model_name == "gru_svm":
                tuned_results[model_name] = run_gru_svm(
                    X_train, X_test, y_train, y_test,
                    tuned_params
                )

    # ---- save results ----
    save_results(paper_results, "paper")

    if compare and tuned_results:
        save_results(tuned_results, "tuned")
        save_comparison(paper_results, tuned_results)


if __name__ == "__main__":
    main()

