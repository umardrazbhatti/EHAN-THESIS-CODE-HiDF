"""
scripts/verify_dataset.py — Pre-training dataset sanity check.

Usage (FF++):
    python scripts/verify_dataset.py \\
        --dataset_name ff++ \\
        --data_root /kaggle/input/.../ffpp_data \\
        --active_manipulation Deepfakes

Usage (HiDF):
    python scripts/verify_dataset.py \\
        --dataset_name hidf \\
        --hidf_root /kaggle/input/.../hidf_data

Exits with code 0 if all checks pass, code 1 if any check fails.
"""

import argparse
import glob
import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def check_directory(label: str, path: str) -> tuple:
    """Return (exists, file_count, sample_filename)."""
    if not os.path.isdir(path):
        return False, 0, ""
    files = glob.glob(os.path.join(path, "*.mp4"))
    sample = os.path.basename(files[0]) if files else ""
    return True, len(files), sample


def main():
    parser = argparse.ArgumentParser(description="Verify dataset layout")
    parser.add_argument("--dataset_name", type=str, default="ff++",
                        choices=["ff++", "hidf"],
                        help="Dataset type to verify.")
    parser.add_argument("--data_root", default=None,
                        help="Root directory of the FF++ dataset (required for ff++)")
    parser.add_argument("--active_manipulation", required=False, default=None,
                        choices=["Deepfakes", "Face2Face", "FaceShifter",
                                 "FaceSwap", "NeuralTextures"],
                        help="Specialist manipulation type (required for ff++).")
    parser.add_argument("--hidf_root", type=str, default=None,
                        help="HiDF dataset root containing Real-vid/ and Fake-vid/ (required for hidf)")
    parser.add_argument("--ffpp_cross_eval", action="store_true",
                        help="Verify FF++ cross-eval root directories")
    parser.add_argument("--ffpp_cross_root", type=str, default=None,
                        help="FF++ ffpp_data/ root for cross-evaluation verification")
    parser.add_argument("--celebdf_eval", action="store_true",
                        help="Verify Celeb-DF root directories")
    parser.add_argument("--celebdf_root", type=str, default=None,
                        help="Celeb-DF v2 root for verification")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="(no-op, accepted for caller compatibility)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="(no-op, accepted for caller compatibility)")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from HiDF_config import EAHNConfig
    from data.HiDF_datasets import DeepfakeDataset
    from data.HiDF_collate import deepfake_collate_fn

    compression = "c23"
    MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

    failures = []

    # ── FF++ checks ───────────────────────────────────────────────────────────
    if args.dataset_name == "ff++":
        if not args.data_root:
            failures.append("--data_root is required for --dataset_name ff++")
        else:
            data_root = args.data_root

            # 1. Directory table
            print("\n=== Directory Check ===")
            header = f"{'Directory':<60} {'Exists':<8} {'Count':<8} {'Sample'}"
            print(header)
            print("-" * len(header))

            real_dir = os.path.join(data_root, "original_sequences", "youtube", compression, "videos")
            exists, count, sample = check_directory("real", real_dir)
            print(f"{real_dir[-60:]:<60} {'YES' if exists else 'NO':<8} {count:<8} {sample}")
            if not exists or count == 0:
                failures.append(f"Real video directory missing or empty: {real_dir}")

            total_fake = 0
            for method in MANIPULATIONS:
                vdir = os.path.join(data_root, "manipulated_sequences", method, compression, "videos")
                exists, count, sample = check_directory(method, vdir)
                print(f"{vdir[-60:]:<60} {'YES' if exists else 'NO':<8} {count:<8} {sample}")
                total_fake += count

            if total_fake == 0:
                failures.append("Zero fake videos found across all manipulation methods.")

            # 2. Dataset loading check
            print("\n=== Dataset Loading Check ===")
            batch = None
            config = None
            try:
                config = EAHNConfig()
                config.data_root           = data_root
                config.active_manipulation = args.active_manipulation
                config.frame_size          = 224
                config.train_split         = 0.8
                config.val_split           = 0.1
                config.cache_dir           = "/kaggle/working/.face_cache_verify"
                config.device              = "cpu"

                ds = DeepfakeDataset(config, "train", "ff++")

                real_indices = [i for i, s in enumerate(ds.samples) if s["label"] == 0]
                fake_indices = [i for i, s in enumerate(ds.samples) if s["label"] == 1]

                if len(real_indices) == 0:
                    failures.append("Zero real samples found in dataset.")
                if len(fake_indices) == 0:
                    failures.append("Zero fake samples found in dataset.")

                balanced_indices = (
                    random.sample(real_indices, min(2, len(real_indices))) +
                    random.sample(fake_indices, min(2, len(fake_indices)))
                )
                random.shuffle(balanced_indices)

                verify_subset = Subset(ds, balanced_indices)
                loader = DataLoader(
                    verify_subset,
                    batch_size=len(balanced_indices),
                    collate_fn=deepfake_collate_fn,
                    shuffle=False,
                    num_workers=0,
                )
                batch = next(iter(loader))

                labels_in_batch = [int(x) for x in batch["label"].tolist()]
                print(f"  Labels in batch : {labels_in_batch}")
                n_real_in_batch = labels_in_batch.count(0)
                n_fake_in_batch = labels_in_batch.count(1)
                print(f"  Real in batch   : {n_real_in_batch}")
                print(f"  Fake in batch   : {n_fake_in_batch}")
                print(f"  Frames shape    : {tuple(batch['frames'].shape)}")
                if n_real_in_batch == 0 or n_fake_in_batch == 0:
                    failures.append(
                        f"Batch is unbalanced: real={n_real_in_batch}, "
                        f"fake={n_fake_in_batch}. Sampler or shuffle may be broken."
                    )
            except Exception as exc:
                failures.append(f"Dataset loading raised: {exc}")
                print(f"  ERROR: {exc}")
                import traceback; traceback.print_exc()

            # 3. Forward-pass check
            print("\n=== Model Forward-Pass Check ===")
            if batch is not None:
                try:
                    from models.HiDF_eahn import EAHN
                    model = EAHN(config).to("cpu")
                    model.eval()
                    with torch.no_grad():
                        out = model(batch["frames"])
                    print(f"  prob values : {[f'{p:.3f}' for p in out.prob.cpu().tolist()]}")
                    mt = out.M_t
                    print(f"  M_t shape   : {tuple(mt.shape)}")
                    print(f"  M_t min/max : {mt.min():.4f} / {mt.max():.4f}")
                    if mt.min() == mt.max():
                        failures.append(
                            "M_t is constant (all same value). Model is not "
                            "producing spatial attention. Check cross-attention module."
                        )
                except Exception as exc:
                    failures.append(f"Forward pass raised: {exc}")
                    print(f"  ERROR: {exc}")
                    import traceback; traceback.print_exc()
            else:
                failures.append(
                    "Forward pass skipped — batch was not loaded. "
                    "Fix the dataset loading error above first."
                )

    # ── HiDF checks ───────────────────────────────────────────────────────────
    if args.dataset_name == "hidf":
        if not args.hidf_root:
            failures.append("--hidf_root is required for --dataset_name hidf")
        else:
            # 1. Directory table
            print("\n=== HiDF Directory Check ===")
            header = f"{'Directory':<60} {'Exists':<8} {'Count':<8} {'Sample'}"
            print(header)
            print("-" * len(header))

            real_vid = os.path.join(args.hidf_root, "Real-vid")
            fake_vid = os.path.join(args.hidf_root, "Fake-vid")

            exists_r, count_r, sample_r = check_directory("Real-vid", real_vid)
            print(f"{real_vid[-60:]:<60} {'YES' if exists_r else 'NO':<8} {count_r:<8} {sample_r}")
            if not exists_r or count_r == 0:
                failures.append(f"HiDF Real-vid missing or empty: {real_vid}")

            exists_f, count_f, sample_f = check_directory("Fake-vid", fake_vid)
            print(f"{fake_vid[-60:]:<60} {'YES' if exists_f else 'NO':<8} {count_f:<8} {sample_f}")
            if not exists_f or count_f == 0:
                failures.append(f"HiDF Fake-vid missing or empty: {fake_vid}")

            # 2. Dataset loading check
            print("\n=== HiDF Dataset Loading Check ===")
            hidf_batch = None
            config = None
            try:
                config = EAHNConfig()
                config.hidf_root   = args.hidf_root
                config.dataset_name = "hidf"
                config.frame_size  = 224
                config.train_split = 0.8
                config.val_split   = 0.1
                config.cache_dir   = "/kaggle/working/.face_cache_verify"
                config.device      = "cpu"

                ds = DeepfakeDataset(config, "train", "hidf")

                real_indices = [i for i, s in enumerate(ds.samples) if s["label"] == 0]
                fake_indices = [i for i, s in enumerate(ds.samples) if s["label"] == 1]

                if len(real_indices) == 0:
                    failures.append("Zero real samples in HiDF train split.")
                if len(fake_indices) == 0:
                    failures.append("Zero fake samples in HiDF train split.")

                balanced_indices = (
                    random.sample(real_indices, min(2, len(real_indices))) +
                    random.sample(fake_indices, min(2, len(fake_indices)))
                )
                random.shuffle(balanced_indices)

                verify_subset = Subset(ds, balanced_indices)
                loader = DataLoader(
                    verify_subset,
                    batch_size=len(balanced_indices),
                    collate_fn=deepfake_collate_fn,
                    shuffle=False,
                    num_workers=0,
                )
                hidf_batch = next(iter(loader))

                labels_in_batch = [int(x) for x in hidf_batch["label"].tolist()]
                print(f"  Labels in batch : {labels_in_batch}")
                n_real_in_batch = labels_in_batch.count(0)
                n_fake_in_batch = labels_in_batch.count(1)
                print(f"  Real in batch   : {n_real_in_batch}")
                print(f"  Fake in batch   : {n_fake_in_batch}")
                print(f"  Frames shape    : {tuple(hidf_batch['frames'].shape)}")
                if n_real_in_batch == 0 or n_fake_in_batch == 0:
                    failures.append(
                        f"HiDF batch is unbalanced: real={n_real_in_batch}, "
                        f"fake={n_fake_in_batch}."
                    )
            except Exception as exc:
                failures.append(f"HiDF dataset loading raised: {exc}")
                print(f"  ERROR: {exc}")
                import traceback; traceback.print_exc()

            # 3. Forward-pass check
            print("\n=== HiDF Model Forward-Pass Check ===")
            if hidf_batch is not None:
                try:
                    from models.HiDF_eahn import EAHN
                    model = EAHN(config).to("cpu")
                    model.eval()
                    with torch.no_grad():
                        out = model(hidf_batch["frames"])
                    print(f"  prob values : {[f'{p:.3f}' for p in out.prob.cpu().tolist()]}")
                    mt = out.M_t
                    print(f"  M_t shape   : {tuple(mt.shape)}")
                    print(f"  M_t min/max : {mt.min():.4f} / {mt.max():.4f}")
                    if mt.min() == mt.max():
                        failures.append(
                            "HiDF M_t is constant (all same value). Check cross-attention module."
                        )
                except Exception as exc:
                    failures.append(f"HiDF forward pass raised: {exc}")
                    print(f"  ERROR: {exc}")
                    import traceback; traceback.print_exc()
            else:
                failures.append(
                    "HiDF forward pass skipped — batch was not loaded. "
                    "Fix the dataset loading error above first."
                )

    # ── Cross-eval root checks (always, regardless of dataset_name) ───────────
    if args.ffpp_cross_eval and args.ffpp_cross_root:
        print("\n=== FF++ Cross-Eval Root Check ===")
        for manip in MANIPULATIONS:
            mdir = os.path.join(args.ffpp_cross_root, "manipulated_sequences",
                                manip, compression, "videos")
            if not os.path.isdir(mdir):
                failures.append(f"FF++ cross-eval dir missing: {mdir}")
                print(f"  MISSING: {mdir}")
            else:
                cnt = len(glob.glob(os.path.join(mdir, "*.mp4")))
                print(f"  OK: {manip} — {cnt} videos")

    if args.celebdf_eval and args.celebdf_root:
        print("\n=== Celeb-DF Root Check ===")
        required = ["Celeb-real", "YouTube-real", "Celeb-synthesis",
                    "List_of_testing_videos.txt"]
        for item in required:
            path = os.path.join(args.celebdf_root, item)
            ok = os.path.exists(path)
            print(f"  {'OK' if ok else 'MISSING'}: {path}")
            if not ok:
                failures.append(f"Celeb-DF required item missing: {path}")

    # ── Result ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED — {len(failures)} issue(s) found:")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED — dataset is ready for training.")
        sys.exit(0)


if __name__ == "__main__":
    main()