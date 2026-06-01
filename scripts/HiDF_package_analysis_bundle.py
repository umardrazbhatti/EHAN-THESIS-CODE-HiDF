"""
scripts/package_analysis_bundle.py — Copy the 20 most-important analysis files into
a flat "analysis essentials" bundle for easy download and inspection.

Usage
-----
    python scripts/package_analysis_bundle.py \\
        --output_dir /kaggle/working/outputs \\
        --manipulation Deepfakes \\
        --bundle_dir  /kaggle/working/outputs/analysis_essentials \\
        [--zip]       # optionally zip the bundle to {bundle_dir}.zip

Called automatically at the end of scripts/evaluate.py when
EAHNConfig.bundle_analysis is True (default True).

Output layout
-------------
All files are flat (no sub-directories).  Each file is named:
    {mnp}_{canonical_name}
where {mnp} is the lowercased manipulation name (e.g. "deepfakes").

Also writes:
    {mnp}_README.txt   — run date, epoch count, config hash, headline numbers

Missing source files are skipped with a warning rather than crashing.
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import datetime
from pathlib import Path
from typing import Optional


def _safe_copy(src: Path, dst: Path) -> bool:
    """Copy src → dst. Return True on success; print warning and return False on failure."""
    if not src.exists():
        print(f"  [bundle] SKIP (not found): {src}")
        return False
    try:
        shutil.copy2(str(src), str(dst))
        print(f"  [bundle] OK  {src.name}  →  {dst.name}")
        return True
    except Exception as e:
        print(f"  [bundle] WARN copy failed {src} → {dst}: {e}")
        return False


def _config_hash(output_dir: Path) -> str:
    """Return a short MD5 of the config dict stored in ffpp_test_metrics.json."""
    try:
        metrics_path = output_dir / "eval" / "ffpp_test_metrics.json"
        if not metrics_path.exists():
            metrics_path = output_dir / "eval" / "metrics.json"
        with open(metrics_path) as f:
            raw = f.read()
        return hashlib.md5(raw.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


def _headline_numbers(output_dir: Path) -> dict:
    """Extract key headline numbers from metrics JSON files."""
    nums = {
        "ffpp_auc": "N/A", "ffpp_fake_acc": "N/A",
        "celebdf_auc": "N/A", "celebdf_fake_acc": "N/A",
        "epoch_count": "N/A",
    }
    try:
        for fname in ("ffpp_test_metrics.json", "metrics.json"):
            p = output_dir / "eval" / fname
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                nums["ffpp_auc"]      = f"{d.get('auc_roc', 0.0):.4f}"
                nums["ffpp_fake_acc"] = f"{d.get('fake_accuracy', 0.0):.4f}"
                break
    except Exception:
        pass
    try:
        p = output_dir / "eval" / "celebdf_test_metrics.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            nums["celebdf_auc"]      = f"{d.get('auc_roc', 0.0):.4f}"
            nums["celebdf_fake_acc"] = f"{d.get('fake_accuracy', 0.0):.4f}"
    except Exception:
        pass
    try:
        hist = output_dir / "training_history.csv"
        if hist.exists():
            with open(hist) as f:
                lines = [l.strip() for l in f if l.strip()]
            nums["epoch_count"] = str(len(lines) - 1)  # minus header
    except Exception:
        pass
    return nums


def package_analysis_bundle(
    output_dir: str,
    manipulation: str,
    bundle_dir: str,
    do_zip: bool = False,
) -> Path:
    """
    Core bundling function.

    Parameters
    ----------
    output_dir   : path to the run's output directory (e.g. /kaggle/working/outputs)
    manipulation : manipulation name (e.g. "Deepfakes")
    bundle_dir   : flat destination directory
    do_zip       : if True, also write {bundle_dir}.zip

    Returns
    -------
    Path to bundle_dir
    """
    out   = Path(output_dir)
    bdl   = Path(bundle_dir)
    bdl.mkdir(parents=True, exist_ok=True)

    mnp = manipulation.lower()   # e.g. "deepfakes"

    # ── Copy table ────────────────────────────────────────────────────────────
    # Format: (source_path_relative_to_output_dir, bundle_filename_suffix)
    copy_plan = [
        # Metrics
        ("eval/ffpp_test_metrics.json",       f"{mnp}_ffpp_metrics.json"),
        ("eval/celebdf_test_metrics.json",     f"{mnp}_celebdf_metrics.json"),
        ("explanation_metrics.json",           f"{mnp}_explanation_metrics.json"),
        ("eval/report.txt",                    f"{mnp}_report.txt"),
        # History
        ("training_history.csv",               f"{mnp}_training_history.csv"),
        ("logs.csv",                           f"{mnp}_logs.csv"),
        # Loss / metric plots
        ("plots/loss_curves.png",              f"{mnp}_loss_curves.png"),
        ("plots/metric_curves.png",            f"{mnp}_metric_curves.png"),
        ("plots/val_accuracy_curves.png",      f"{mnp}_val_accuracy_curves.png"),
        # FF++ detection visuals
        ("plots/ffpp_roc.png",                 f"{mnp}_ffpp_roc.png"),
        ("plots/ffpp_pr.png",                  f"{mnp}_ffpp_pr.png"),
        ("plots/ffpp_confusion.png",           f"{mnp}_ffpp_confusion.png"),
        ("plots/ffpp_score_distribution.png",  f"{mnp}_ffpp_score_dist.png"),
        # Celeb-DF detection visuals
        ("plots/celebdf_roc.png",              f"{mnp}_celebdf_roc.png"),
        ("plots/celebdf_pr.png",               f"{mnp}_celebdf_pr.png"),
        ("plots/celebdf_confusion.png",        f"{mnp}_celebdf_confusion.png"),
        ("plots/celebdf_score_distribution.png", f"{mnp}_celebdf_score_dist.png"),
        # Summary charts
        ("plots/ffpp_summary_chart.png",       f"{mnp}_ffpp_summary.png"),
    ]

    n_ok = 0
    for rel_src, dst_name in copy_plan:
        ok = _safe_copy(out / rel_src, bdl / dst_name)
        if ok:
            n_ok += 1

    # ── Heatmap strips (glob — pick one per category) ─────────────────────────
    for strip_glob, dst_name in [
        ("heatmaps/heatmap_strip_fake_correct_*.png",  f"{mnp}_strip_fake_correct.png"),
        ("heatmaps/heatmap_strip_real_correct_*.png",  f"{mnp}_strip_real_correct.png"),
    ]:
        import glob as _glob
        matches = sorted(_glob.glob(str(out / strip_glob)))
        if matches:
            ok = _safe_copy(Path(matches[0]), bdl / dst_name)
            if ok:
                n_ok += 1
        else:
            print(f"  [bundle] SKIP (no match): {strip_glob}")

    # ── README ────────────────────────────────────────────────────────────────
    nums         = _headline_numbers(out)
    cfg_hash     = _config_hash(out)
    run_date     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    readme_lines = [
        f"EAHN Analysis Bundle — {manipulation}",
        f"Generated : {run_date}",
        f"Config MD5 (partial): {cfg_hash}",
        f"Epoch count: {nums['epoch_count']}",
        "",
        "Headline numbers",
        "----------------",
        f"  FF++ AUC-ROC      : {nums['ffpp_auc']}",
        f"  FF++ fake_acc@0.5 : {nums['ffpp_fake_acc']}",
        f"  Celeb-DF AUC-ROC  : {nums['celebdf_auc']}",
        f"  Celeb-DF fake@0.5 : {nums['celebdf_fake_acc']}",
        "",
        "Files",
        "-----",
        f"  All {n_ok} files listed above, named {mnp}_* (flat, no sub-dirs).",
    ]
    readme_path = bdl / f"{mnp}_README.txt"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"  [bundle] README written → {readme_path}")

    # ── Optional zip ──────────────────────────────────────────────────────────
    if do_zip:
        import zipfile
        zip_path = str(bdl) + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in sorted(bdl.iterdir()):
                zf.write(fpath, fpath.name)
        print(f"[bundle] Zipped → {zip_path}")

    total_files = sum(1 for _ in bdl.iterdir())
    print(f"[bundle] Done. {total_files} files in {bdl}")
    return bdl


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package analysis essentials into a flat bundle directory."
    )
    parser.add_argument("--output_dir",   required=True,
                        help="Run output directory (e.g. /kaggle/working/outputs)")
    parser.add_argument("--manipulation", required=True,
                        choices=["Deepfakes", "Face2Face", "FaceShifter",
                                 "FaceSwap", "NeuralTextures"],
                        help="Manipulation type for this fork")
    parser.add_argument("--bundle_dir",   required=True,
                        help="Destination flat directory for the bundle")
    parser.add_argument("--zip", dest="do_zip", action="store_true", default=False,
                        help="Also compress the bundle to {bundle_dir}.zip")
    args = parser.parse_args()

    package_analysis_bundle(
        output_dir=args.output_dir,
        manipulation=args.manipulation,
        bundle_dir=args.bundle_dir,
        do_zip=args.do_zip,
    )
