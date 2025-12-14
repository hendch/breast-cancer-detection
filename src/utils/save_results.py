import csv
import os
from datetime import datetime


def save_results(results, run_mode, output_dir="results"):
    """
    results: dict
        {
          model_name: {
            accuracy: float,
            precision: float,
            recall: float,
            f1: float
          }
        }
    """

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run_mode}_results_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)

        # header
        writer.writerow([
            "model",
            "run_mode",
            "accuracy",
            "precision",
            "recall",
            "f1_score"
        ])

        # rows
        for model_name, metrics in results.items():
            writer.writerow([
                model_name,
                run_mode,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            ])

    print(f"Results saved to {filepath}")
