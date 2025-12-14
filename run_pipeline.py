import yaml

from src.data.load_data import load_wdbc
from src.training.train_mlp import train_and_evaluate_mlp
from src.utils.io import save_json, now_iso


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f) or {}

    X, y = load_wdbc()

    metrics = train_and_evaluate_mlp(X, y, config)

    payload = {
        "model": "mlp",
        "timestamp": now_iso(),
        "random_state": config["project"]["random_state"],
        "test_size": config["project"]["test_size"],
        **metrics
    }

    save_json("results/latest_metrics.json", payload)
    print("Saved metrics to results/latest_metrics.json")
    print(payload)


if __name__ == "__main__":
    main()
