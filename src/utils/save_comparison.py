import csv
import os
from datetime import datetime


def save_comparison(paper_results, tuned_results, output_dir="results"):
    """
    paper_results / tuned_results:
      { model_name: {accuracy, precision, recall, f1}, ... }
    """

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"comparison_{timestamp}.csv")

    metrics = ["accuracy", "precision", "recall", "f1"]

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "paper_accuracy", "tuned_accuracy", "delta_accuracy",
            "paper_precision", "tuned_precision", "delta_precision",
            "paper_recall", "tuned_recall", "delta_recall",
            "paper_f1", "tuned_f1", "delta_f1",
        ])

        all_models = sorted(set(paper_results.keys()) | set(tuned_results.keys()))

        for model in all_models:
            p = paper_results.get(model, {})
            t = tuned_results.get(model, {})

            row = [model]
            for m in metrics:
                pv = p.get(m, "")
                tv = t.get(m, "")
                dv = ""
                if pv != "" and tv != "":
                    dv = tv - pv
                row.extend([pv, tv, dv])

            writer.writerow(row)

    print(f"Comparison saved to {filepath}")
    return filepath
