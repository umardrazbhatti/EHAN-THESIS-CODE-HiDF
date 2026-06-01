"""
scripts/summary_chart.py — Two-panel summary chart: dataset split and
detection breakdown for the EAHN evaluation report.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_summary_chart(metrics_dict: dict, split_counts: dict, output_dir: str) -> str:
    """
    Build and save a two-panel summary chart.

    Parameters
    ----------
    metrics_dict : dict
        Keys: tp, fp, tn, fn, auc_roc  (floats / ints)
    split_counts : dict
        Keys: total, train, train_real, train_fake, val, test, test_real, test_fake
    output_dir : str
        Directory where "summary_chart.png" will be saved.

    Returns
    -------
    output_path : str
    """
    tp      = int(metrics_dict.get("tp",      0))
    fp      = int(metrics_dict.get("fp",      0))
    tn      = int(metrics_dict.get("tn",      0))
    fn      = int(metrics_dict.get("fn",      0))
    auc_roc = float(metrics_dict.get("auc_roc", 0.0))

    total      = int(split_counts.get("total",      0))
    train      = int(split_counts.get("train",      0))
    train_real = int(split_counts.get("train_real", 0))
    train_fake = int(split_counts.get("train_fake", 0))
    val        = int(split_counts.get("val",        0))
    test       = int(split_counts.get("test",       0))
    test_real  = int(split_counts.get("test_real",  0))
    test_fake  = int(split_counts.get("test_fake",  0))

    fig = plt.figure(figsize=(14, 6), facecolor="#1a1a2e")

    # ── LEFT PANEL — Dataset Split ────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor("#16213e")

    labels_ds = ["Total", "Train", "Validation", "Test"]
    values_ds = [total, train, val, test]
    colors_ds = ["#e94560", "#0f3460", "#533483", "#e94560"]

    bars = ax1.bar(
        labels_ds, values_ds, color=colors_ds, edgecolor="white", linewidth=0.5
    )

    y_max = max(values_ds) if values_ds else 1
    for bar, val_ in zip(bars, values_ds):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.01,
            str(val_),
            ha="center", va="bottom",
            color="white", fontweight="bold", fontsize=11,
        )

    # Real/Fake breakdown inside Train bar
    if train > 0:
        ax1.text(
            bars[1].get_x() + bars[1].get_width() / 2,
            bars[1].get_height() / 2,
            f"Real: {train_real}\nFake: {train_fake}",
            ha="center", va="center",
            color="white", fontsize=9,
        )

    # Real/Fake breakdown inside Test bar
    if test > 0:
        ax1.text(
            bars[3].get_x() + bars[3].get_width() / 2,
            bars[3].get_height() / 2,
            f"Real: {test_real}\nFake: {test_fake}",
            ha="center", va="center",
            color="white", fontsize=9,
        )

    ax1.set_title("Dataset Split", color="white", fontweight="bold", fontsize=14)
    ax1.set_ylabel("Number of Samples", color="white")
    ax1.set_ylim(0, y_max * 1.15)
    ax1.tick_params(colors="white")
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")

    # ── RIGHT PANEL — Detection Breakdown ────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#16213e")

    labels_det = [
        "Fake → Fake\n(True Positive)",
        "Fake → Real\n(False Negative)",
        "Real → Real\n(True Negative)",
        "Real → Fake\n(False Positive)",
    ]
    values_det = [tp, fn, tn, fp]
    colors_det = ["#2ecc71", "#e74c3c", "#3498db", "#e67e22"]

    bars2 = ax2.bar(
        labels_det, values_det, color=colors_det, edgecolor="white", linewidth=0.5
    )

    max_det = max(values_det) if any(v > 0 for v in values_det) else 10
    for bar, val_ in zip(bars2, values_det):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_det * 0.01,
            str(val_),
            ha="center", va="bottom",
            color="white", fontweight="bold", fontsize=11,
        )

    total_preds = max(tp + tn + fp + fn, 1)
    accuracy    = (tp + tn) / total_preds
    ax2.set_title(
        f"Detection Breakdown  |  Accuracy: {accuracy:.1%}  |  AUC-ROC: {auc_roc:.3f}",
        color="white", fontweight="bold", fontsize=12,
    )
    ax2.set_ylabel("Number of Videos", color="white")
    ax2.set_ylim(0, max_det * 1.2)
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout(pad=2.0)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "summary_chart.png")
    plt.savefig(
        output_path, dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()
    print(f"Summary chart saved -> {output_path}")
    return output_path
