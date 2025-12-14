import os
import glob
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def _latest_file(pattern: str):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def plot_latest_comparison(results_dir="results", output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = _latest_file(os.path.join(results_dir, "comparison_*.csv"))
    if not csv_path:
        print("No comparison CSV found.")
        return

    df = pd.read_csv(csv_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics = ["accuracy", "precision", "recall", "f1"]

    for metric in metrics:
        p = f"paper_{metric}"
        t = f"tuned_{metric}"
        d = f"delta_{metric}"

        if p not in df or t not in df or d not in df:
            continue

        values = df[[p, t]].values.flatten()
        ymin = values.min() - 0.01
        ymax = values.max() + 0.01

        x = range(len(df))
        width = 0.25

        # -------------------------
        # Paper vs Tuned (zoomed)
        # -------------------------
        plt.figure(figsize=(6, 4))
        plt.bar(
            [i - width / 2 for i in x],
            df[p],
            width=width,
            label="paper"
        )
        plt.bar(
            [i + width / 2 for i in x],
            df[t],
            width=width,
            label="tuned"
        )

        plt.title(f"Paper vs Tuned – {metric}")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.xticks(list(x), df["model"])
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, f"paper_vs_tuned_{metric}_{timestamp}.png"),
            dpi=160
        )
        plt.close()

        # -------------------------
        # Delta plot (centered)
        # -------------------------
        delta_max = max(abs(df[d].min()), abs(df[d].max())) + 0.002

        plt.figure(figsize=(6, 4))
        plt.bar(df["model"], df[d])
        plt.axhline(0, linewidth=1)
        plt.ylim(-delta_max, delta_max)

        plt.title(f"Delta (Tuned − Paper) – {metric}")
        plt.ylabel(f"Δ {metric}")
        plt.xlabel("Model")
        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, f"delta_{metric}_{timestamp}.png"),
            dpi=160
        )
        plt.close()

    print(f"Plots generated from {csv_path}")
