"""
scripts/HiDF_package_paper_assets.py
====================================
Build a CURATED, paper-ready asset bundle from a finished run's results folder.

Unlike `analysis_essentials` (a flat dump that foregrounds the OLD pixel-occlusion
explanation numbers), this bundle contains ONLY what a paper / thesis examiner
needs, organised into sections, with ONE consistent set of headline numbers:

  paper_assets/
    README.md                      plain-language guide: which number is headline + why
    paper_metrics.csv              clean consolidated numbers (no confounded metrics)
    01_detection/                  ROC / PR / confusion / score-distribution
    02_cross_dataset/              AUC bar chart + table
    03_explanation/                contribution del/ins curve + authoritative report + heatmaps
    04_training/                   loss / val-accuracy curves + history
    appendix_standard_metrics/     ROAD curve + NOTE (confounded standard metrics, for completeness)

The headline explanation numbers are the CONTRIBUTION-SPACE del/ins (the metric
appropriate for the additive head); the standard input-perturbation metrics
(pixel del/ins, ROAD) are clearly demoted to the appendix with a one-line caveat.

Usage
-----
    python scripts/HiDF_package_paper_assets.py --results_dir "<run folder>" [--zip]

Runs standalone on any downloaded results folder; safe to re-run (idempotent).
"""
import os
import csv
import json
import glob
import shutil
import zipfile
import argparse
from pathlib import Path


def _find(root: Path, *patterns):
    """Return the first existing file matching any of the candidate globs."""
    for pat in patterns:
        hits = sorted(glob.glob(str(root / pat)))
        if hits:
            return Path(hits[0])
    return None


def _copy(src, dst_dir: Path, dst_name: str):
    if src is None:
        print(f"  [paper] SKIP (not found): {dst_name}")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst_dir / dst_name))
    print(f"  [paper] OK  {dst_name}")
    return True


def _load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def package_paper_assets(results_dir: str, do_zip: bool = True) -> Path:
    root = Path(results_dir)
    out = root / "paper_assets"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- read headline numbers ------------------------------------------------
    det = _load_json(_find(root, "eval/ffpp_test_metrics.json", "eval/metrics.json",
                           "analysis_essentials/01_hidf_metrics.json", "metrics.json"))
    ex  = _load_json(_find(root, "explanation_metrics.json",
                           "analysis_essentials/08_explanation_metrics.json"))
    hl  = ex.get("faithfulness_headline", {})
    intr = ex.get("intrinsic", {})

    # cross-dataset table
    cross_csv = _find(root, "cross_dataset_summary.csv",
                      "analysis_essentials/10_cross_dataset_summary.csv")
    cross_rows = []
    if cross_csv:
        with open(cross_csv) as f:
            for r in csv.DictReader(f):
                cross_rows.append((r["cell"], float(r["auc"]), r.get("kind", "")))

    # ---- 01 detection ---------------------------------------------------------
    d1 = out / "01_detection"
    _copy(_find(root, "roc_curve.png", "plots/*roc*.png"), d1, "roc.png")
    _copy(_find(root, "pr_curve.png", "plots/*pr*.png"), d1, "pr.png")
    _copy(_find(root, "confusion_matrix.png", "plots/*confusion*.png"), d1, "confusion_matrix.png")
    _copy(_find(root, "score_distribution.png", "plots/*score*.png"), d1, "score_distribution.png")

    # ---- 02 cross-dataset -----------------------------------------------------
    d2 = out / "02_cross_dataset"
    _copy(_find(root, "cross_dataset_summary.png",
                "analysis_essentials/11_cross_dataset_summary.png"), d2, "cross_dataset_auc.png")
    _copy(cross_csv, d2, "cross_dataset.csv")

    # ---- 03 explanation (headline) -------------------------------------------
    d3 = out / "03_explanation"
    _copy(_find(root, "contribution_del_ins_curve.png"), d3, "contribution_del_ins_curve.png")
    _copy(_find(root, "faithfulness_report.txt"), d3, "faithfulness_report.txt")
    _copy(_find(root, "heatmaps/heatmap_strip_fake_correct_*.png", "heatmaps/*fake_correct*.png"),
          d3, "heatmap_fake_example.png")
    _copy(_find(root, "heatmaps/heatmap_strip_real_correct_*.png", "heatmaps/*real_correct*.png"),
          d3, "heatmap_real_example.png")

    # ---- 04 training ----------------------------------------------------------
    d4 = out / "04_training"
    _copy(_find(root, "loss_curves.png", "plots/loss_curves.png"), d4, "loss_curves.png")
    _copy(_find(root, "analysis_essentials/13_val_accuracy_curves.png",
                "val_accuracy_curves.png", "metric_curves.png"), d4, "val_accuracy_curves.png")
    _copy(_find(root, "training_history.csv"), d4, "training_history.csv")

    # ---- appendix: standard (confounded) metrics ------------------------------
    da = out / "appendix_standard_metrics"
    _copy(_find(root, "road_curve.png"), da, "road_curve.png")
    da.mkdir(parents=True, exist_ok=True)
    (da / "NOTE.txt").write_text(
        "Standard input-perturbation explanation metrics (pixel-occlusion insertion/\n"
        "deletion, ROAD). These are CONFOUNDED for this model: the prediction pools\n"
        "globally-mixed transformer tokens, so occluding input pixels lets the network\n"
        "rebuild the evidence from other regions -> the metric reads near-random and is\n"
        "NOT a valid faithfulness verdict here. Reported for completeness only. The\n"
        "headline faithfulness is the contribution-space del/ins in ../03_explanation/\n"
        "and ../paper_metrics.csv. See the paper's explanation section for the caveat.\n",
        encoding="ascii", errors="replace")

    # ---- paper_metrics.csv (clean headline only) ------------------------------
    def g(d, k, default=""):
        v = d.get(k, default)
        return f"{v:.4f}" if isinstance(v, float) else v

    rows = [("section", "metric", "value")]
    rows += [
        ("detection", "auc_roc", g(det, "auc_roc")),
        ("detection", "balanced_accuracy@0.5", g(det, "balanced_accuracy")),
        ("detection", "real_accuracy@0.5", g(det, "real_accuracy")),
        ("detection", "fake_accuracy@0.5", g(det, "fake_accuracy")),
        ("detection", "balanced_accuracy@optimal", g(det, "balanced_accuracy_at_optimal")),
        ("detection", "optimal_threshold", g(det, "optimal_threshold")),
        ("explanation", "faithful", "YES" if hl.get("faithful") else "NO"),
        ("explanation", "insertion_fake_deployed", g(hl, "blended_ins_fake")),
        ("explanation", "deletion_fake_deployed", g(hl, "blended_del_fake")),
        ("explanation", "insertion_fake_head", g(hl, "head_ins_fake")),
        ("explanation", "deletion_fake_head", g(hl, "head_del_fake")),
        ("explanation", "insertion_gain_vs_random", g(hl, "blended_ins_gain")),
        ("explanation", "deletion_gain_vs_random", g(hl, "blended_del_gain")),
        ("explanation", "faithfulness_corr", g(intr, "faithfulness_corr")),
        ("explanation", "n_fake_evaluated", str(hl.get("n_fake", ""))),
    ]
    for cell, auc, kind in cross_rows:
        if kind != "in-dist":
            rows.append(("cross_dataset", f"{cell}_auc", f"{auc:.4f}"))
    with open(out / "paper_metrics.csv", "w", newline="", encoding="ascii", errors="replace") as f:
        csv.writer(f).writerows(rows)
    print("  [paper] OK  paper_metrics.csv")

    # ---- README.md (plain language) -------------------------------------------
    auc = g(det, "auc_roc"); ba = g(det, "balanced_accuracy")
    ra = g(det, "real_accuracy"); fa = g(det, "fake_accuracy")
    ins = g(hl, "blended_ins_fake"); dele = g(hl, "blended_del_fake")
    ig = g(hl, "blended_ins_gain"); dg = g(hl, "blended_del_gain")
    faith = "YES" if hl.get("faithful") else "NO"
    readme = f"""# Paper assets -- EAHN-HiDF

This folder contains ONLY the files needed for the paper/thesis, with ONE
consistent set of numbers. (The full run folder has everything else.)

## Headline numbers (use these)

**Detection (in-distribution):** AUC-ROC {auc}, balanced accuracy {ba}
(real {ra} / fake {fa} at threshold 0.5).

**Explanation faithfulness (contribution space, all {hl.get('n_fake','?')} fakes):**
insertion {ins}, deletion {dele}; the model's own saliency beats a random ordering
by +{ig} (insertion) / +{dg} (deletion) -> **FAITHFUL = {faith}**.
Full statement: `03_explanation/faithfulness_report.txt`.

## Which explanation number is the headline, and why (one paragraph for the paper)

The model predicts by SUMMING per-region evidence (an additive head), so the
faithfulness of the saliency is measured in that exact additive space
(contribution-space insertion/deletion) -- the numbers above. Standard
input-perturbation metrics (pixel insertion/deletion, ROAD) are placed in
`appendix_standard_metrics/` for completeness; they are confounded for a
globally-mixed transformer (occluded evidence is rebuilt from other regions),
so they read near-random and are NOT used as the headline.

## Contents
- `paper_metrics.csv`        -- all headline numbers in one table
- `01_detection/`            -- ROC, PR, confusion matrix, score distribution
- `02_cross_dataset/`        -- cross-dataset AUC chart + table (honest limitation)
- `03_explanation/`          -- contribution del/ins curve, authoritative report, example heatmaps
- `04_training/`             -- loss / validation-accuracy curves + per-epoch history
- `appendix_standard_metrics/` -- confounded standard metrics, for completeness only
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print("  [paper] OK  README.md")

    # ---- zip ------------------------------------------------------------------
    if do_zip:
        zip_path = str(out) + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(out.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(out.parent))
        sz = os.path.getsize(zip_path) / 1e6
        print(f"[paper] Zipped -> {zip_path} ({sz:.1f} MB)")
    print(f"[paper] Done -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Curate a paper-ready asset bundle from a run folder.")
    ap.add_argument("--results_dir", required=True, help="Path to a finished run's results folder")
    ap.add_argument("--no_zip", action="store_true", help="Do not create the .zip")
    args = ap.parse_args()
    package_paper_assets(args.results_dir, do_zip=not args.no_zip)
