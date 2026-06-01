"""
scripts/dashboard.py — Kaggle-compatible dashboard displaying metrics and heatmaps.
Can be called standalone or from evaluate.py.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from IPython.display import display, HTML, Video as IPVideo
    _IN_NOTEBOOK = True
except ImportError:
    _IN_NOTEBOOK = False


def show_dashboard(output_dir: str = "/kaggle/working/outputs"):
    metrics_csv = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        print(f"No metrics.csv found in {output_dir}")
        return

    # ── Load metrics ──────────────────────────────────────────────────────────
    metrics_dict = {}
    with open(metrics_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header if present
        for row in reader:
            if len(row) == 2:
                name, val = row
                try:
                    val = float(val)
                except ValueError:
                    pass
                metrics_dict[name] = val

    print("\n📊 EAHN Evaluation Dashboard")
    print("=" * 50)

    # ── Metrics table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(metrics_dict.items(), columns=["Metric", "Value"])
    if _IN_NOTEBOOK:
        display(HTML("<h3>📋 Metrics</h3>"))
        display(df)
    else:
        print(df.to_string(index=False))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    plot_metrics = {
        k: v for k, v in metrics_dict.items()
        if isinstance(v, (int, float)) and not np.isnan(v)
    }
    if plot_metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(plot_metrics.keys(), plot_metrics.values(), color="steelblue")
        ax.set_ylabel("Value")
        ax.set_title("EAHN — Explanation & Detection Metrics")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "metrics_bar.png")
        plt.savefig(chart_path, dpi=120)
        plt.close()
        print(f"Bar chart saved: {chart_path}")
        if _IN_NOTEBOOK:
            display(HTML(f'<img src="{chart_path}" width="700"/>'))

    # ── Heatmap videos ────────────────────────────────────────────────────────
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    if not os.path.isdir(heatmap_dir):
        print("No heatmap directory found.")
    else:
        _SUFFIXES = ["_intrinsic.mp4", "_gradcam.mp4", "_rollout.mp4", "_shap.mp4"]
        sample_ids = set()
        for fname in os.listdir(heatmap_dir):
            for suffix in _SUFFIXES:
                if fname.endswith(suffix):
                    sample_ids.add(fname[: -len(suffix)])
        sample_ids = sorted(sample_ids)

        if not sample_ids:
            print("No heatmap videos found.")
        else:
            if _IN_NOTEBOOK:
                display(HTML("<h3>🔥 Heatmap Videos</h3>"))

            for sid in sample_ids[:5]:
                print(f"\nSample {sid}:")
                for method in ["intrinsic", "gradcam", "rollout", "shap"]:
                    fpath = os.path.join(heatmap_dir, f"{sid}_{method}.mp4")
                    if os.path.exists(fpath):
                        print(f"  [{method}] {fpath}")
                        if _IN_NOTEBOOK:
                            display(HTML(f"<b>{method}</b>"))
                            display(IPVideo(fpath, embed=True, width=400))
                    else:
                        print(f"  [{method}] MISSING")

    # ── Summary chart ─────────────────────────────────────────────────────────
    chart_path = os.path.join(output_dir, "summary_chart.png")
    if os.path.exists(chart_path):
        print(f"\nSummary chart: {chart_path}")
        if _IN_NOTEBOOK:
            display(HTML(f'<img src="{chart_path}" width="900"/>'))
    else:
        print("\nSummary chart: NOT FOUND (run evaluate first)")

    # ── Explanation strips ────────────────────────────────────────────────────
    explanation_dir = os.path.join(output_dir, "explanations")
    if os.path.isdir(explanation_dir):
        strip_files = [f for f in os.listdir(explanation_dir) if f.endswith("_strip.png")]
        print(f"\nExplanations directory: {explanation_dir}")
        print(f"  {len(strip_files)} annotated strip(s) found.")
    else:
        print("\nExplanations directory: NOT FOUND")

    print("\nDashboard complete.")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/outputs"
    show_dashboard(out)
