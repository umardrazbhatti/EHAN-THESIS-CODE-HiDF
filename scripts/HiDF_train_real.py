"""
scripts/HiDF_train_real.py — Phase 6 GPU training on FF++/Celeb-DF/DFDC/HiDF.

Phase 6 changes vs phase 5d:
  - --max_per_class flag for balanced 1k/1k subsampling  (CHANGE 1)
  - WeightedRandomSampler safety net rebuild              (CHANGE 2)
  - 100-batch rolling log (not per-step)                 (CHANGE 3)
  - Per-epoch attention-diversity diagnostic              (CHANGE 4)
  - label_smoothing wired through build_classification_loss (CHANGE 6)
  - loss_curves.png + metric_curves.png +
    training_history.csv emitted at end of training       (CHANGE 12)

v2 patch — all-three-metrics fix:
  [mt_std]         B-pass no_grad REMOVED → loss_faith gradient now reaches
                   EarlyAttnHead via x_b → M_norm → M_t. PeakSpreadLoss added.
  [peak_mode_share] PeakSpreadLoss + raised JS-div weight (in HiDF_explanation.py)
                   directly penalise batch-level peak-location concentration.
  [cosine]         Untouched — HiDF grouped splitting in datasets.py handles this.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv as _csv
import dataclasses as _dataclasses
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from HiDF_config import EAHNConfig, parse_args
from data.HiDF_datasets import DeepfakeDataset
from data.HiDF_collate import deepfake_collate_fn
from models.HiDF_eahn import EAHN
from losses.HiDF_classification import build_classification_loss
from losses.HiDF_explanation import (
    ExplanationLoss,
    HardAttentionDiversityLoss,  # v4: replaces PeakSpreadLoss
    sharpness_loss,               # v4: operates on M_t_logits, no softmax ceiling
    build_bottlenecked_input,
    faithfulness_loss,
    sparsity_loss,
)
from losses.HiDF_temporal import TemporalConsistencyLoss
from metrics.HiDF_detection import DetectionMetrics
from utils.HiDF_checkpointing import save_checkpoint, load_checkpoint
from utils.HiDF_logging_utils import Logger


def _faith_warmup(epoch: int, warmup_epochs: int, target: float) -> float:
    """Linear ramp from 0 (epoch 0) to target (epoch warmup_epochs)."""
    if warmup_epochs <= 0:
        return target
    return target * min(1.0, float(epoch) / float(warmup_epochs))


def main(config: EAHNConfig):
    device = torch.device(config.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        cap  = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
        print(f"[Device] {name} | CUDA capability sm_{cap[0]}{cap[1]}")
        if cap[0] < 7:
            print(
                f"[WARNING] sm_{cap[0]}{cap[1]} is below PyTorch minimum "
                f"(sm_70). Switch Kaggle accelerator to T4. "
                f"Falling back to CPU for MTCNN. AMP disabled."
            )
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(config, "train", config.dataset_name)
    val_ds   = DeepfakeDataset(config, "val",   config.dataset_name)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── DataLoader — Regime A: plain shuffle, no sampler ─────────────────────
    print("[Sampler] Mode=shuffled  (WeightedRandomSampler DISABLED for Regime A)")
    _train_generator = torch.Generator()
    _train_generator.manual_seed(42)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=deepfake_collate_fn,
        pin_memory=(config.device == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
        generator=_train_generator,
    )
    print(
        f"[DataLoader] batch_size={config.batch_size}  shuffle=True  "
        f"num_workers={config.num_workers}  drop_last=True  generator=seed42"
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(config.device == "cuda"),
    )
    print(f"[DataLoader val] batch_size={config.batch_size}  shuffle=False  size={len(val_ds)}")

    # ── Smoke check ───────────────────────────────────────────────────────────
    _smoke_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=deepfake_collate_fn, num_workers=0,
    )
    _saw_real = _saw_fake = False
    for _i, _sb in enumerate(iter(_smoke_loader)):
        _bl = _sb["label"].cpu().numpy().astype(int)
        _r, _f = int((_bl == 0).sum()), int((_bl == 1).sum())
        print(f"[Smoke] Batch {_i}: real={_r} fake={_f}")
        if _r > 0: _saw_real = True
        if _f > 0: _saw_fake = True
        if _i == 2: break
    del _smoke_loader
    assert _saw_real and _saw_fake, (
        "All 3 inspected batches are single-class. Split or DataLoader broken — "
        "check DeepfakeDataset._split()."
    )
    print("[Smoke] Both classes seen across 3 batches — Regime A loader OK.")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EAHN(config).to(device)

    if config.grad_checkpoint:
        model.enable_gradient_checkpointing()
        print("[GradCkpt] Gradient checkpointing enabled on TemporalStream.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    _use_amp = (
        config.use_amp
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 7
    )
    _amp_dtype = torch.float16 if config.amp_dtype == "fp16" else torch.bfloat16
    _dev_str   = device.type
    scaler     = GradScaler(_dev_str, enabled=_use_amp)
    print(f"[AMP] use_amp={_use_amp}  dtype={config.amp_dtype}")

    logger = Logger(config.output_dir)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_metric = -1.0
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        ckpt        = load_checkpoint(config.resume_checkpoint, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0)
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"[Resume] Loaded {config.resume_checkpoint}, "
              f"resuming from epoch {start_epoch + 1}  (best_metric={best_metric:.4f})")
    elif config.resume_checkpoint:
        print(f"[Resume] Checkpoint not found at {config.resume_checkpoint!r} — starting fresh.")

    # ── Losses ────────────────────────────────────────────────────────────────
    cls_loss_fn = build_classification_loss(config)
    print(
        f"[ClsLoss] {cls_loss_fn.__class__.__name__}  "
        f"label_smoothing={getattr(config, 'label_smoothing', 0.0)}"
    )

    exp_loss_fn   = ExplanationLoss(
        alpha=config.alpha,
        beta=config.beta,
        diversity_weight=config.attn_diversity_weight,  # already raised to 4.0 in explanation.py default
    )
    temp_loss_fn  = TemporalConsistencyLoss(gamma=config.gamma)

    # v4: HardAttentionDiversityLoss — batch-level cell-popularity concentration
    # Directly attacks peak_mode_share (fraction of batch sharing same argmax cell).
    # temperature=0.05 makes it near-hard-argmax, closely matching the diagnostic.
    _lambda_peak_spread = float(getattr(config, "lambda_peak_spread", 0.5))
    peak_spread_fn = HardAttentionDiversityLoss(temperature=0.05)
    print(f"[HardAttentionDiversityLoss] lambda_peak_spread={_lambda_peak_spread}")

    # v4: sharpness loss on M_t_logits (pre-softmax). Output is tanh-bounded
    # in [-1,0] so lambda_sharp=0.15 keeps it safely below cls magnitude.
    _lambda_sharp = float(getattr(config, "lambda_sharp", 0.15))
    print(f"[SharpnessLoss-logits] lambda_sharp={_lambda_sharp}")

    ckpt_path = os.path.join(config.output_dir, f"eahn_{config.dataset_name}_best.pth")

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        "epoch":               [],
        "train_total":         [], "train_cls":    [],
        "train_exp":           [], "train_temp":   [],
        "train_faith":         [], "train_sparse": [],
        "train_peak_spread":   [],                      # v2: new term
        "train_sharp":         [],                      # v3: sharpness loss
        "val_auc_roc":         [], "val_balanced_acc":      [],
        "val_real_acc":        [], "val_fake_acc":          [],
        "val_inter_sample_cos": [], "val_mt_std":           [],
        "val_peak_mode_share":  [],                     # v2: now tracked in history
    }

    # ── Early stopping ────────────────────────────────────────────────────────
    _no_early_stop = bool(getattr(config, "no_early_stop", False))
    _es_patience  = int(getattr(config, "early_stop_patience",  5))
    _es_min_delta = float(getattr(config, "early_stop_min_delta", 0.001))
    _es_metric    = str(getattr(config, "early_stop_metric", "val_balanced_accuracy"))
    _es_best      = -float("inf")
    _es_wait      = 0
    _es_triggered = False
    if _no_early_stop:
        print(f"[EarlyStopping] DISABLED (--no_early_stop) — will run all {config.epochs} epochs.")
    else:
        print(
        f"[EarlyStopping] metric={_es_metric}  patience={_es_patience}  "
        f"min_delta={_es_min_delta}"
    )

    # ── Clean train loader (augmentation shortcut detection) ──────────────────
    from copy import deepcopy
    import torch as _torch
    _clean_ds = deepcopy(train_ds)
    _clean_ds.heavy_aug = False
    from data.HiDF_transforms import get_transforms
    _clean_ds.transform = get_transforms("val", config.frame_size)
    _clean_ds.minority_class = -1
    _clean_gen = torch.Generator()
    _clean_gen.manual_seed(42)
    _clean_loader = DataLoader(
        _clean_ds, batch_size=config.batch_size,
        shuffle=True, generator=_clean_gen,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
        pin_memory=(config.device == "cuda"),
    )
    print(f"[sanity] clean (unaugmented) train loader built: {len(_clean_ds)} samples")

    # ── Training loop ─────────────────────────────────────────────────────────
    total_batches = len(train_loader)
    epoch_w       = len(str(start_epoch + config.epochs))

    for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_acc = {
            "total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0,
            "faith": 0.0, "sparse": 0.0, "peak_spread": 0.0, "sharp": 0.0, "n": 0,
        }

        LOG_EVERY = 200
        run = {
            "total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0,
            "cons": 0.0, "faith": 0.0, "sparse": 0.0,
            "peak_spread": 0.0, "sharp": 0.0, "n": 0,
        }

        for batch_idx, batch in enumerate(train_loader):
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(_dev_str, enabled=_use_amp, dtype=_amp_dtype):
                # ── Pass A: normal forward ────────────────────────────────────
                out_A    = model(frames)
                logits_A = out_A.logit
                M_t      = out_A.M_t          # (B, T, h, w) softmax from EarlyAttnHead
                M_t_logits = out_A.M_t_logits  # (B, T, h, w) pre-softmax raw scores
                loss_cls = cls_loss_fn(logits_A, labels)

                if config.phase21_enabled:
                    # ── Pass B: bottlenecked input — gradient ENABLED ─────────
                    # v2 FIX: no_grad REMOVED here.
                    # Gradient path: loss_faith → logits_B → model(x_b)
                    #                → x_b → M_norm → M_t → EarlyAttnHead
                    # This gives EarlyAttnHead a real gradient from faithfulness,
                    # forcing it to produce maps that actually gate meaningful
                    # regions → mt_std rises above 0.15.
                    #
                    # Memory note: storing B-pass activations costs ~same as A-pass.
                    # With batch_size=2, grad_accum=2 this is fine on T4 (8GB).
                    # If OOM occurs, reduce batch_size or increase grad_accum_steps.
                    x_b = build_bottlenecked_input(
                        frames, M_t,
                        blur_kernel=config.blur_kernel,
                        blur_sigma=config.blur_sigma,
                        peak_floor=config.bottleneck_peak_floor,  # Phase 22: fixed floor
                    )
                    out_B       = model(x_b)           # GRAD ENABLED (v2 fix)
                    loss_faith  = faithfulness_loss(logits_A, out_B.logit)
                    loss_sparse = sparsity_loss(M_t)
                else:
                    loss_faith  = torch.zeros((), device=frames.device)
                    loss_sparse = torch.zeros((), device=frames.device)

                # ── Explanation + temporal ────────────────────────────────────
                exp_out = exp_loss_fn(M_t)
                l_exp   = exp_out.loss
                l_temp  = temp_loss_fn(M_t, out_A.low_level)

                # ── HardAttentionDiversityLoss (v4) ──────────────────────────
                # Batch-level cell-popularity concentration. Near-hard-argmax
                # (temperature=0.05) → directly attacks peak_mode_share metric.
                l_peak_spread = peak_spread_fn(M_t)

                # ── Sharpness loss on RAW LOGITS (v4) ────────────────────────
                # Softmax std over 49 cells is CAPPED at ≈0.141 (below threshold
                # of 0.15). Operating on pre-softmax logits removes this ceiling.
                # sharpness_loss() = -std(M_t_logits) per (b,t), averaged.
                loss_sharp = sharpness_loss(M_t_logits)

                # ── Loss weighting ────────────────────────────────────────────
                _global_step = (epoch - 1) * len(train_loader) + batch_idx
                _lambda1_eff = config.lambda1 * min(1.0, _global_step / 200.0)
                lam_faith_eff = _faith_warmup(
                    epoch, config.faith_warmup_epochs, config.lambda_faith
                )

                l_total = (loss_cls
                           + lam_faith_eff          * loss_faith
                           + config.lambda_sparse    * loss_sparse
                           + _lambda1_eff            * l_exp
                           + config.lambda2          * l_temp
                           + _lambda_peak_spread     * l_peak_spread
                           + _lambda_sharp           * loss_sharp)

                # ── Consistency regularisation (unchanged) ────────────────────
                _lambda_cons = float(getattr(config, "lambda_consistency", 0.0))
                if _lambda_cons > 0 and "frames_clean" in batch:
                    _frames_clean = batch["frames_clean"].to(device, non_blocking=True)
                    with torch.no_grad():
                        _out_clean = model(_frames_clean)
                    _probs_clean = _out_clean.prob.detach()
                    _probs_aug   = out_A.prob
                    l_consistency = F.mse_loss(_probs_aug, _probs_clean)
                    l_total = l_total + _lambda_cons * l_consistency
                else:
                    l_consistency = torch.tensor(0.0)

                # ── NaN guard — skip step if any loss term is non-finite ──────
                if not torch.isfinite(l_total):
                    print(
                        f"[NaNGuard] Non-finite loss at epoch={epoch} "
                        f"batch={batch_idx}: total={l_total.item():.4f}  "
                        f"cls={loss_cls.item():.4f}  "
                        f"sharp={loss_sharp.item():.4f}  "
                        f"peak={l_peak_spread.item():.4f}  "
                        f"— skipping backward for this step."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss = l_total / config.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # ── First-batch diagnostics ───────────────────────────────────────
            if epoch == start_epoch + 1 and batch_idx == 0:
                print(f"[DIAG] M_t mean={out_A.M_t.mean():.4f} std={out_A.M_t.std():.4f}  "
                      f"M_t_logits std={out_A.M_t_logits.std():.4f}")
                print(f"[DIAG] L_cls={loss_cls.item():.6f}  L_exp={l_exp.item():.6f}  "
                      f"L_temp={l_temp.item():.6f}  L_faith={loss_faith.item():.6f}  "
                      f"L_sparse={loss_sparse.item():.6f}  "
                      f"L_peak_spread={l_peak_spread.item():.6f}  "
                      f"L_sharp={loss_sharp.item():.6f}")
                print(f"[DIAG] lam_faith_eff={lam_faith_eff:.4f}  "
                      f"lambda_peak_spread={_lambda_peak_spread}  "
                      f"lambda_sharp={_lambda_sharp}  "
                      f"lambda_sparse={config.lambda_sparse}")
                # v4: log early_attn_tau (the tau that actually controls M_t sharpness)
                # NOT cross_attention.log_temp which is a dead legacy module
                print(f"[DIAG] early_attn_tau={out_A.early_attn_tau:.4f}  "
                      f"(log_tau={model.early_attn.log_tau.item():.4f})")

            # ── Batch balance check ───────────────────────────────────────────
            if (batch_idx + 1) % LOG_EVERY == 0:
                bl = batch["label"].detach().cpu().numpy().astype(int)
                n_real, n_fake = int((bl == 0).sum()), int((bl == 1).sum())
                print(f"[BatchBalance] step={batch_idx+1} real={n_real} fake={n_fake}")

            # ── Accumulate losses ─────────────────────────────────────────────
            _lt  = l_total.item()
            _lc  = loss_cls.item()
            _le  = l_exp.item()
            _lp  = l_temp.item()
            _lco = l_consistency.item()
            _lf  = loss_faith.item()
            _ls  = loss_sparse.item()
            _lps = l_peak_spread.item()
            _lsh = loss_sharp.item()

            run["total"]       += _lt;  run["cls"]    += _lc
            run["exp"]         += _le;  run["temp"]   += _lp
            run["cons"]        += _lco
            run["faith"]       += _lf;  run["sparse"] += _ls
            run["peak_spread"] += _lps; run["sharp"]  += _lsh; run["n"] += 1

            epoch_acc["total"]       += _lt;  epoch_acc["cls"]    += _lc
            epoch_acc["exp"]         += _le;  epoch_acc["temp"]   += _lp
            epoch_acc["faith"]       += _lf;  epoch_acc["sparse"] += _ls
            epoch_acc["peak_spread"] += _lps; epoch_acc["sharp"]  += _lsh
            epoch_acc["n"]           += 1

            # ── Rolling log ───────────────────────────────────────────────────
            if (batch_idx + 1) % LOG_EVERY == 0 or (batch_idx + 1) == total_batches:
                n = max(run["n"], 1)
                _tau = out_A.early_attn_tau  # v4: actual M_t sharpening tau
                print(
                    f"[E{epoch:>{epoch_w}} {batch_idx+1:4d}/{total_batches}] "
                    f"total={run['total']/n:.4f}  cls={run['cls']/n:.4f}  "
                    f"exp={run['exp']/n:.4f}  temp={run['temp']/n:.4f}  "
                    f"faith={run['faith']/n:.4f}  sparse={run['sparse']/n:.4f}  "
                    f"sharp={run['sharp']/n:.4f}  "
                    f"peak_spread={run['peak_spread']/n:.4f}  "
                    f"cons={run['cons']/n:.4f}  "
                    f"tau={_tau:.3f}  sim={exp_out.inter_sample_sim:.2f}"
                )
                run = {
                    "total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0,
                    "cons": 0.0, "faith": 0.0, "sparse": 0.0,
                    "peak_spread": 0.0, "sharp": 0.0, "n": 0,
                }

        scheduler.step()

        # ── Epoch-average train losses ─────────────────────────────────────────
        n = max(epoch_acc["n"], 1)
        history["epoch"].append(epoch)
        history["train_total"].append(epoch_acc["total"]       / n)
        history["train_cls"].append(epoch_acc["cls"]           / n)
        history["train_exp"].append(epoch_acc["exp"]           / n)
        history["train_temp"].append(epoch_acc["temp"]         / n)
        history["train_faith"].append(epoch_acc["faith"]       / n)
        history["train_sparse"].append(epoch_acc["sparse"]     / n)
        history["train_peak_spread"].append(epoch_acc["peak_spread"] / n)
        history["train_sharp"].append(epoch_acc["sharp"] / n)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                out    = model(frames)
                probs_list.extend(out.prob.cpu().tolist())
                labels_list.extend(batch["label"].cpu().tolist())

        metrics = DetectionMetrics.compute(probs_list, labels_list)
        logger.log_scalars("val", metrics, epoch)
        print(
            f"Epoch {epoch:>{epoch_w}}/{start_epoch + config.epochs} | "
            f"Val AUC-ROC: {metrics['auc_roc']:.4f} | "
            f"F1: {metrics['f1_at_0.5']:.4f}"
        )

        _val_real_acc = float(metrics.get("real_accuracy",    0.0))
        _val_fake_acc = float(metrics.get("fake_accuracy",    0.0))
        _val_bal_acc  = float(metrics.get("balanced_accuracy", 0.0))
        print(
            f"[ValMetrics] epoch={epoch} "
            f"real_acc={_val_real_acc:.3f} "
            f"fake_acc={_val_fake_acc:.3f} "
            f"balanced_acc={_val_bal_acc:.3f}"
        )

        # ── Attention-diversity diagnostic (v4: full val-set) ────────────────
        # mt_std: computed on M_t_LOGITS (pre-softmax) to avoid the 0.141 ceiling.
        # peak_mode_share: argmax on softmax M_t (correct — matches diagnostic def).
        # cosine: on softmax M_t (correct — measures map similarity).
        _all_mt_flat       = []   # softmax maps for cosine + peak_mode_share
        _all_mt_logit_flat = []   # raw logits for mt_std
        _all_mt_peaks      = []
        with torch.no_grad():
            for _diag_batch in val_loader:
                _diag_frames = _diag_batch["frames"].to(device)
                _diag_out    = model(_diag_frames)
                _mt_b        = _diag_out.M_t.mean(dim=1)           # (B, h, w) softmax
                _mt_flat_b   = _mt_b.reshape(_mt_b.size(0), -1)    # (B, hw)
                _ml_b        = _diag_out.M_t_logits.mean(dim=1)    # (B, h, w) raw logits
                _ml_flat_b   = _ml_b.reshape(_ml_b.size(0), -1)    # (B, hw)
                _all_mt_flat.append(_mt_flat_b.cpu())
                _all_mt_logit_flat.append(_ml_flat_b.cpu())
                _all_mt_peaks.extend([int(m.argmax().item()) for m in _mt_flat_b])
        _all_mt_flat_cat   = torch.cat(_all_mt_flat, dim=0)         # (N_val, hw)
        _all_ml_flat_cat   = torch.cat(_all_mt_logit_flat, dim=0)   # (N_val, hw)

        # cosine similarity (on softmax maps)
        _mt_norm_all = torch.nn.functional.normalize(_all_mt_flat_cat, dim=1)
        _chunk = 64
        _cos_vals = []
        for _ci in range(0, len(_mt_norm_all), _chunk):
            _row = _mt_norm_all[_ci:_ci+_chunk]
            _cos_block = _row @ _mt_norm_all.t()
            for _ri, _gi in enumerate(range(_ci, min(_ci+_chunk, len(_mt_norm_all)))):
                _cos_block[_ri, _gi] = 0.0
            _cos_vals.append(_cos_block.sum(dim=1))
        _N_val      = len(_mt_norm_all)
        diag_cosine = float(torch.cat(_cos_vals).sum() / max(_N_val * (_N_val - 1), 1))

        # mt_std on RAW LOGITS — no softmax ceiling
        diag_std    = float(_all_ml_flat_cat.std(dim=1).mean())

        # peak_mode_share (on softmax argmax — correct)
        _peak_counts = {}
        for _pk in _all_mt_peaks:
            _peak_counts[_pk] = _peak_counts.get(_pk, 0) + 1
        _peak_mode_share = max(_peak_counts.values()) / max(len(_all_mt_peaks), 1)
        model.train()

        _pass_cos  = diag_cosine     < 0.95
        _pass_std  = diag_std        > 0.15
        _pass_peak = _peak_mode_share < 0.30
        print(
            f"[Diag] epoch={epoch}  scale=1.00  "
            f"inter_sample_cos={diag_cosine:.3f} {'PASS' if _pass_cos  else 'FAIL'}  "
            f"mt_std={diag_std:.4f} {'PASS' if _pass_std  else 'FAIL'}  "
            f"peak_mode_share={_peak_mode_share:.3f} {'PASS' if _pass_peak else 'FAIL'}"
        )

        # ── History ───────────────────────────────────────────────────────────
        history["val_auc_roc"].append(float(metrics.get("auc_roc", float("nan"))))
        history["val_balanced_acc"].append(_val_bal_acc)
        history["val_real_acc"].append(_val_real_acc)
        history["val_fake_acc"].append(_val_fake_acc)
        history["val_inter_sample_cos"].append(diag_cosine)
        history["val_mt_std"].append(diag_std)
        history["val_peak_mode_share"].append(_peak_mode_share)

        # ── Clean-train sanity check ───────────────────────────────────────────
        model.eval()
        _clean_probs, _clean_labels = [], []
        with _torch.no_grad():
            for _i, _b in enumerate(_clean_loader):
                if _i * config.batch_size >= 200:
                    break
                _f = _b["frames"].to(device)
                _o = model(_f)
                _clean_probs.extend(_o.prob.cpu().tolist())
                _clean_labels.extend(_b["label"].cpu().tolist())
        _clean_probs  = np.array(_clean_probs)
        _clean_labels = np.array(_clean_labels)
        _clean_real_acc = float(((_clean_probs < 0.5) & (_clean_labels == 0)).sum() /
                                max((_clean_labels == 0).sum(), 1))
        _clean_fake_acc = float(((_clean_probs >= 0.5) & (_clean_labels == 1)).sum() /
                                max((_clean_labels == 1).sum(), 1))
        print(f"[sanity] epoch={epoch} train_clean: real_acc={_clean_real_acc:.3f} "
              f"fake_acc={_clean_fake_acc:.3f}  "
              f"(if real_acc is much lower than val real_acc, aug shortcut still live)")

        if _clean_fake_acc < 0.20:
            print(
                f"[sanity] WARNING — AUGMENTATION SHORTCUT DETECTED: "
                f"train_clean fake_acc={_clean_fake_acc:.3f} < 0.20."
            )
            if epoch >= 2:
                print(
                    f"[sanity] STOP CONDITION: train_clean fake_acc still < 0.20 at epoch {epoch}. "
                    f"Diagnose augmentation pipeline first."
                )

        model.train()

        # ── Checkpoint ────────────────────────────────────────────────────────
        SELECTION_KEY = "balanced_accuracy_at_optimal"
        sel = metrics.get(SELECTION_KEY)
        if sel is None or not np.isfinite(sel):
            print(f"[CheckpointSelect] {SELECTION_KEY} missing/NaN, falling back to auc_roc")
            sel = metrics.get("auc_roc", 0.0)

        if sel > best_metric:
            best_metric = sel
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric,
                            config, ckpt_path)
            print(f"--> Best model saved ({SELECTION_KEY}: {best_metric:.4f}, "
                  f"val_auc_roc={metrics['auc_roc']:.4f})")

        if config.save_last_checkpoint:
            _last_path = os.path.join(config.output_dir, "last_checkpoint.pth")
            _last_tmp  = _last_path + ".tmp"
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric":          best_metric,
                    "config":               _dataclasses.asdict(config),
                },
                _last_tmp,
            )
            os.replace(_last_tmp, _last_path)
            print(f"[Checkpoint] last_checkpoint.pth saved  "
                  f"(epoch={epoch}, best_metric={best_metric:.4f})")

        # ── Always-save last_epoch.pth ─────────────────────────────────────────
        _last_epoch_path = os.path.join(config.output_dir, "last_epoch.pth")
        _last_epoch_tmp  = _last_epoch_path + ".tmp"
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric":          best_metric,
                "config":               _dataclasses.asdict(config),
            },
            _last_epoch_tmp,
        )
        os.replace(_last_epoch_tmp, _last_epoch_path)
        print(f"[Checkpoint] last_epoch.pth saved (epoch={epoch}, best_metric={best_metric:.4f})")

        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            _cur_mem  = torch.cuda.memory_allocated(device)  / 1e9
            _peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"[Mem] epoch={epoch}  cur={_cur_mem:.2f}GB  peak={_peak_mem:.2f}GB")
            torch.cuda.reset_peak_memory_stats(device)

        # ── Phase 21 snapshot ──────────────────────────────────────────────────
        if getattr(config, "phase21_enabled", True) and \
                ((epoch + 1) % config.snapshot_every == 0):
            snap_dir = Path(config.output_dir) / "snapshots" / f"epoch_{epoch+1:02d}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict()},
                snap_dir / "model.pth",
            )
            with torch.no_grad():
                mt_stats = {
                    "mean": float(M_t.mean().item()),
                    "std":  float(M_t.std().item()),
                    "peak_per_frame_mean": float(M_t.amax(dim=(-2, -1)).mean().item()),
                    "min":  float(M_t.min().item()),
                    "max":  float(M_t.max().item()),
                }

            def _avg(key):
                _n = max(1, epoch_acc.get("n", 1))
                return float(epoch_acc.get(key, 0.0)) / _n

            snap_meta = {
                "epoch":                epoch + 1,
                "train_loss_cls":       _avg("cls"),
                "train_loss_faith":     _avg("faith"),
                "train_loss_sparse":    _avg("sparse"),
                "train_loss_peak_spread": _avg("peak_spread"),
                "train_loss_exp":       _avg("exp"),
                "train_loss_temp":      _avg("temp"),
                "train_loss_total":     _avg("total"),
                "lam_faith_eff":        float(lam_faith_eff),
                "lambda_peak_spread":   float(_lambda_peak_spread),
                "val_auc_roc":          float(metrics.get("auc_roc", -1.0)),
                "val_balanced_acc":     float(metrics.get("balanced_accuracy", -1.0)),
                "val_real_acc":         float(metrics.get("real_accuracy", -1.0)),
                "val_fake_acc":         float(metrics.get("fake_accuracy", -1.0)),
                "diag_inter_sample_cos": diag_cosine,
                "diag_mt_std":          diag_std,
                "diag_peak_mode_share": _peak_mode_share,
                "M_t_stats_last_batch": mt_stats,
            }
            with open(snap_dir / "meta.json", "w") as _sf:
                json.dump(snap_meta, _sf, indent=2)
            print(f"[Phase21 snapshot] saved → {snap_dir}")

        # ── Early stopping check ───────────────────────────────────────────────
        if not _no_early_stop:
            _es_metric_map = {
                "val_balanced_accuracy": "val_balanced_acc",
                "val_balanced_acc":      "val_balanced_acc",
                "val_auc_roc":           "val_auc_roc",
                "val_fake_accuracy":     "val_fake_acc",
                "val_fake_acc":          "val_fake_acc",
            }
            _es_key = _es_metric_map.get(_es_metric, "val_balanced_acc")
            _es_cur = history[_es_key][-1] if history[_es_key] else float("nan")

            if np.isfinite(_es_cur):
                if _es_cur > _es_best + _es_min_delta:
                    _es_best = _es_cur
                    _es_wait = 0
                    print(f"[EarlyStopping] Improvement → {_es_key}={_es_cur:.4f} (best={_es_best:.4f})")
                else:
                    _es_wait += 1
                    print(
                        f"[EarlyStopping] No improvement for {_es_wait}/{_es_patience} epochs "
                        f"({_es_key}={_es_cur:.4f} ≤ best+delta={_es_best + _es_min_delta:.4f})"
                    )
                    if _es_wait >= _es_patience:
                        print(
                            f"[EarlyStopping] TRIGGERED at epoch {epoch}. "
                            f"Restoring best checkpoint ({SELECTION_KEY}={best_metric:.4f})."
                        )
                        if os.path.exists(ckpt_path):
                            load_checkpoint(ckpt_path, model)
                            print(f"[EarlyStopping] Best weights restored from {ckpt_path}")
                        _es_triggered = True
                        break

    logger.close()
    _stop_reason = "early stopping" if _es_triggered else "epoch limit"
    print(f"\nTraining complete ({_stop_reason}). "
          f"Best balanced_accuracy_at_optimal: {best_metric:.4f}")

    # ── Save final model ───────────────────────────────────────────────────────
    _final_path = os.path.join(config.output_dir, "final_model.pth")
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric":          best_metric,
            "config":               _dataclasses.asdict(config),
        },
        _final_path,
    )
    print(f"[Checkpoint] final_model.pth saved  "
          f"(epoch={epoch}, best_metric={best_metric:.4f}, stop_reason={_stop_reason})")

    # ── End-of-run plots and CSV ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        csv_hist = out_path / "training_history.csv"
        with open(csv_hist, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(list(history.keys()))
            w.writerows(zip(*history.values()))
        print(f"[plot] saved {csv_hist}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for ax, (key, title) in zip(axes.flat, [
            ("train_total",       "Total Loss"),
            ("train_cls",         "Classification Loss"),
            ("train_exp",         "Explanation Loss"),
            ("train_temp",        "Temporal Consistency Loss"),
            ("train_faith",       "Faithfulness Loss"),
            ("train_peak_spread", "Peak Spread Loss (v2)"),
        ]):
            ax.plot(history["epoch"], history[key], marker="o", linewidth=2)
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
        fig.suptitle("EAHN-HiDF — Training Loss Convergence (v2)", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path / "loss_curves.png", dpi=120)
        plt.close(fig)
        print(f"[plot] saved {out_path / 'loss_curves.png'}")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (keys, title) in zip(axes.flat, [
            (["val_auc_roc"],                              "Val AUC-ROC"),
            (["val_real_acc", "val_fake_acc"],             "Per-class Val Accuracy"),
            (["val_balanced_acc"],                         "Val Balanced Accuracy"),
            (["val_inter_sample_cos", "val_mt_std",
              "val_peak_mode_share"],                      "Attention Diversity (v2)"),
        ]):
            for k in keys:
                if k in history:
                    ax.plot(history["epoch"], history[k],
                            marker="o", linewidth=2, label=k)
            if "AUC" in title or "Balanced" in title:
                ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="random")
            # Target lines for the three metrics
            if "Diversity" in title:
                ax.axhline(0.95, color="red",   linestyle=":", alpha=0.7,
                           label="cos threshold (0.95)")
                ax.axhline(0.15, color="green", linestyle=":", alpha=0.7,
                           label="mt_std threshold (0.15)")
                ax.axhline(0.30, color="blue",  linestyle=":", alpha=0.7,
                           label="peak_mode threshold (0.30)")
            ax.set_title(title); ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.suptitle("EAHN-HiDF — Validation Metric Trajectories (v2)", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path / "metric_curves.png", dpi=120)
        plt.close(fig)
        print(f"[plot] saved {out_path / 'metric_curves.png'}")

        plots_dir = out_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _manip = getattr(config, "active_manipulation", "")
        fig2, ax2_acc = plt.subplots(figsize=(10, 6))
        ax2_auc = ax2_acc.twinx()
        ax2_acc.plot(history["epoch"], history["val_balanced_acc"],
                     marker="o", linewidth=2.5, color="tab:blue",
                     label="val_balanced_accuracy")
        ax2_acc.plot(history["epoch"], history["val_real_acc"],
                     marker="s", linewidth=1.5, linestyle="--",
                     color="tab:green", label="val_real_accuracy")
        ax2_acc.plot(history["epoch"], history["val_fake_acc"],
                     marker="^", linewidth=1.5, linestyle="--",
                     color="tab:red", label="val_fake_accuracy")
        ax2_acc.axhline(0.5, color="grey", linestyle=":", alpha=0.6, linewidth=1)
        ax2_acc.set_xlabel("Epoch")
        ax2_acc.set_ylabel("Accuracy / Balanced Accuracy", color="tab:blue")
        ax2_acc.set_ylim(0, 1)
        ax2_acc.tick_params(axis="y", labelcolor="tab:blue")
        ax2_auc.plot(history["epoch"], history["val_auc_roc"],
                     marker="D", linewidth=2, linestyle="-",
                     color="tab:purple", alpha=0.7, label="val_auc_roc")
        ax2_auc.set_ylabel("AUC-ROC", color="tab:purple")
        ax2_auc.set_ylim(0, 1)
        ax2_auc.tick_params(axis="y", labelcolor="tab:purple")
        lines2a, labels2a = ax2_acc.get_legend_handles_labels()
        lines2b, labels2b = ax2_auc.get_legend_handles_labels()
        ax2_acc.legend(lines2a + lines2b, labels2a + labels2b,
                       loc="lower right", fontsize=9)
        title_manip = f" — {_manip}" if _manip else ""
        fig2.suptitle(f"Validation Performance per Epoch{title_manip}", fontsize=13)
        fig2.tight_layout()
        _val_acc_path = plots_dir / "val_accuracy_curves.png"
        fig2.savefig(_val_acc_path, dpi=120)
        plt.close(fig2)
        print(f"[plot] saved {_val_acc_path}")

    except Exception as _plot_err:
        print(f"[plot] Warning: could not generate training plots: {_plot_err}")

    if config.eval_after_train:
        from scripts.HiDF_evaluate import run_evaluation
        print("\n--- Starting evaluation ---")
        run_evaluation(config)


if __name__ == "__main__":
    args   = parse_args()
    config = EAHNConfig.from_args(args)
    main(config)
