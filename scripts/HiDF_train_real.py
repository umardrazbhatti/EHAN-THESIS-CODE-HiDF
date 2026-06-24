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
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    temporal_sparsity_loss,       # Phase 25: pushes M_frame to be peaky → k1/k2/k4 > 1
    temporal_band_loss,           # Phase 30: bounded hinge on eff_fr (no one-hot ratchet)
    spatial_band_loss,            # Phase 31: bounded two-sided hinge on eff_sp (replaces -peak)
    full_blur_input,              # Phase 31: D-pass anchor step (full blur, no M_t dependence)
    cbm_diversity_loss,           # Phase 26: slot diversity (re-export wrapper)
    localization_loss,            # Phase 33: pull M_t onto the self-blend boundary
)
from data.HiDF_self_blend import make_sbi_batch   # Phase 33: online self-blend generator
from losses.HiDF_temporal import TemporalConsistencyLoss
from metrics.HiDF_detection import DetectionMetrics
from utils.HiDF_checkpointing import save_checkpoint, load_checkpoint
from utils.HiDF_logging_utils import Logger
from models.HiDF_grl import domain_warmup, domain_accuracy   # Phase 27


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

    # ── DataLoader — Phase 26: class-balanced sampler optional ──────────────
    # Phase 25 evidence: with plain shuffle on HiDF (real:fake = 3488:3175
    # = 1.10:1) plus the over-amplified Phase 25 hyperparameters, the model
    # collapsed to predicting "fake" for 86% of real samples.
    # WeightedRandomSampler with weight = 1/n_class per sample equalises the
    # expected per-batch class ratio and is the cheapest defense against
    # this failure mode.  Set --no_class_balanced_sampler to revert to
    # Regime A.
    _train_generator = torch.Generator()
    _train_generator.manual_seed(42)
    _use_cb_sampler = bool(getattr(config, "class_balanced_sampler", True))

    if _use_cb_sampler:
        # Compute per-sample weight = 1 / count(class_of_sample)
        _labels_arr = np.array(
            [s["label"] for s in train_ds.samples], dtype=int
        )
        _n_real = int((_labels_arr == 0).sum())
        _n_fake = int((_labels_arr == 1).sum())
        # Inverse-class-frequency weights
        _w = np.where(_labels_arr == 0, 1.0 / max(_n_real, 1),
                                         1.0 / max(_n_fake, 1))
        _w_t = torch.as_tensor(_w, dtype=torch.double)
        _sampler = WeightedRandomSampler(
            weights=_w_t,
            num_samples=len(train_ds),     # one epoch = one pass over dataset
            replacement=True,
            generator=_train_generator,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=_sampler,              # shuffle MUST be off when sampler is given
            num_workers=config.num_workers,
            collate_fn=deepfake_collate_fn,
            pin_memory=(config.device == "cuda"),
            persistent_workers=(config.num_workers > 0),
            drop_last=True,
        )
        print(
            f"[Sampler] Phase 26: WeightedRandomSampler enabled "
            f"(real={_n_real} w={1.0/max(_n_real,1):.6f}, "
            f"fake={_n_fake} w={1.0/max(_n_fake,1):.6f})"
        )
        print(
            f"[DataLoader] batch_size={config.batch_size}  sampler=weighted  "
            f"num_workers={config.num_workers}  drop_last=True"
        )
    else:
        print("[Sampler] Mode=shuffled  (WeightedRandomSampler DISABLED — Phase 24 behaviour)")
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
    # Confirms the split + collate actually yield BOTH classes.  Scans up to
    # _smoke_max batches and STOPS as soon as both are seen.  The old version
    # inspected exactly 3 batches (6 samples) and hard-asserted — but at the
    # HiDF 0.9:1 ratio a single-class draw of 6 samples is just an unlucky
    # shuffle (P ~ 0.476^6 ~ 1.2%), NOT a broken loader, and must never kill an
    # 11h run (it did on Exp3 of the 3-account sweep while Exp1/Exp2 drew lucky).
    # A genuinely broken split stays single-class across ALL _smoke_max batches
    # (P ~ 0.476^50 ~ 1e-16 for the real check), which still trips the assert.
    # Seeded so the draw is reproducible across runs (no more flaky gate).
    _smoke_gen = torch.Generator()
    _smoke_gen.manual_seed(1234)
    _smoke_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=deepfake_collate_fn, num_workers=0, generator=_smoke_gen,
    )
    _saw_real = _saw_fake = False
    _smoke_max = 25
    _i = -1
    for _i, _sb in enumerate(iter(_smoke_loader)):
        _bl = _sb["label"].cpu().numpy().astype(int)
        _r, _f = int((_bl == 0).sum()), int((_bl == 1).sum())
        print(f"[Smoke] Batch {_i}: real={_r} fake={_f}")
        if _r > 0: _saw_real = True
        if _f > 0: _saw_fake = True
        if _saw_real and _saw_fake:
            break
        if _i + 1 >= _smoke_max:
            break
    del _smoke_loader
    assert _saw_real and _saw_fake, (
        f"No mixed-class batch in {_i + 1} inspected batches "
        f"(batch_size={config.batch_size}). Split or DataLoader genuinely "
        f"broken — check DeepfakeDataset._split()."
    )
    print(f"[Smoke] Both classes seen by batch {_i} — Regime A loader OK.")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EAHN(config).to(device)

    if config.grad_checkpoint:
        model.enable_gradient_checkpointing()
        print("[GradCkpt] Gradient checkpointing enabled on TemporalStream.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(config.epochs), 1), eta_min=1e-6
    )  # Phase 42: max(.,1) so eval-only runs (--epochs 0 + --resume_checkpoint)
       # construct the scheduler without a divide-by-zero in get_lr.

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

    # ── Phase 33: self-blend (SBI) + boundary-supervised attention ────────────
    # Bounded additive aux pass (once per optimizer step) -- does NOT touch the
    # A/B/D detection regime, so met detection/deletion are protected.  The
    # localization term is the sweep axis across the 3 accounts.
    _sbi_enabled     = bool(getattr(config, "sbi_enabled", False))
    _lambda_localize = float(getattr(config, "lambda_localize", 0.0))
    _lambda_sbi_cls  = float(getattr(config, "lambda_sbi_cls", 0.5))
    _sbi_lo          = float(getattr(config, "sbi_blend_lo", 0.25))
    _sbi_hi          = float(getattr(config, "sbi_blend_hi", 0.55))
    _sbi_stride      = max(int(getattr(config, "sbi_stride", 8)), 1)
    _sbi_active      = _sbi_enabled and _lambda_localize > 0.0
    # ── Phase 35: enhanced generator (artifact modes + temporally-partial) ────
    _sbi_modes = tuple(s.strip() for s in
                       str(getattr(config, "sbi_modes", "blend")).split(",")
                       if s.strip()) or ("blend",)
    _sbi_plo   = float(getattr(config, "sbi_partial_frac_lo", 1.0))
    _sbi_phi   = float(getattr(config, "sbi_partial_frac_hi", 1.0))
    _sbi_freq  = float(getattr(config, "sbi_freq_mismatch", 0.0))
    print(
        f"[Phase33] sbi_enabled={_sbi_enabled}  lambda_localize={_lambda_localize}  "
        f"lambda_sbi_cls={_lambda_sbi_cls}  blend=[{_sbi_lo},{_sbi_hi}]  "
        f"stride={_sbi_stride}  active={_sbi_active}"
    )
    print(
        f"[Phase35] sbi_modes={list(_sbi_modes)}  partial_frac=[{_sbi_plo},{_sbi_phi}]  "
        f"(modes=cross-dataset artifacts; partial<1.0=key-frame for k-drops)"
    )
    print(
        f"[Phase37] sbi_freq_mismatch={_sbi_freq}  "
        f"(per-clip downscale-upscale on the blend source = resolution seam cue)"
    )

    # ── Phase 34: hard spatial top-k attention bottleneck ─────────────────────
    # Architectural fix for the insertion wall: the model's prediction is funnelled
    # through the top `spatial_topk_frac` of M_t's 49 cells (see EAHN.forward).
    # Lives in the model, so no per-step wiring here -- this just reports config
    # and computes the keep-cell count for the diagnostics.  0.0 = OFF (Phase 33).
    _spatial_topk        = float(getattr(config, "spatial_topk_frac", 0.0))
    _spatial_topk_active = 0.0 < _spatial_topk < 1.0
    _spatial_topk_k      = max(1, int(math.ceil(_spatial_topk * model.N))) if _spatial_topk_active else model.N
    print(
        f"[Phase34] spatial_topk_frac={_spatial_topk}  active={_spatial_topk_active}  "
        f"keep_cells={_spatial_topk_k}/{model.N}  (insertion bottleneck; STE + convex renorm)"
    )

    # ── Phase 35: dual-lens (necessity M_nec + sufficiency M_suff) ────────────
    _dual_lens = bool(getattr(config, "dual_lens_enabled", False))
    _suff_frac = float(getattr(config, "suff_topk_frac", 0.0))
    _suff_k    = (max(1, int(math.ceil(_suff_frac * model.N)))
                  if 0.0 < _suff_frac < 1.0 else model.N)
    print(
        f"[Phase35] dual_lens={_dual_lens}  suff_topk_frac={_suff_frac}  "
        f"suff_keep_cells={_suff_k}/{model.N}  "
        f"(B-pass+faith -> M_suff; D-pass+seam -> M_nec; classifier reads both pools)"
    )

    # ── Phase 36: intrinsic multi-layer evidence decomposition ────────────────
    # L complementary attention maps + convex per-layer weights, in the forward
    # pass; the combined map replaces M_t downstream so every metric scores the
    # decomposition.  parallel = L heads + diversity loss; sequential = 1 head x L
    # steps with suppression.  Lives in EAHN.forward -- this just reports config
    # and the diversity weight added to l_total below.  OFF = single-map P33/34/35.
    _decomp_enabled = bool(getattr(config, "decomp_enabled", False))
    _decomp_mode    = str(getattr(config, "decomp_mode", "parallel"))
    _decomp_L       = int(getattr(config, "decomp_layers", 4))
    _lam_decomp_div = float(getattr(config, "lambda_decomp_div", 0.1))
    _decomp_active  = _decomp_enabled
    print(
        f"[Phase36] decomp_enabled={_decomp_enabled}  mode={_decomp_mode}  "
        f"layers={_decomp_L}  lambda_div={_lam_decomp_div}  "
        f"gate_init={getattr(config, 'decomp_gate_init', -0.5)}  "
        f"(intrinsic multi-layer decomposition; combined map replaces M_t downstream)"
    )

    # ── Phase 38: PRE-transformer hard spatial bottleneck (steep-curve axis) ──
    # Lives in EAHN.forward (binary top-k gate on the conv feature map BEFORE the
    # transformer).  This just reports config + keep-cell count.  0.0 = OFF.
    _early_topk   = float(getattr(config, "early_topk_frac", 0.0))
    _early_active = 0.0 < _early_topk < 1.0
    _early_k      = (max(1, int(math.ceil(_early_topk * model.N)))
                     if _early_active else model.N)
    print(
        f"[Phase38] early_topk_frac={_early_topk}  active={_early_active}  "
        f"keep_cells={_early_k}/{model.N}  "
        f"(PRE-transformer hard gate; STE; forces locally-removable evidence)"
    )
    _aeh_on = bool(getattr(config, "aeh_enabled", False))
    print(
        f"[Phase39] aeh_enabled={_aeh_on}  "
        f"warmup_epochs={getattr(config, 'aeh_warmup_epochs', 3)}  "
        f"lambda_aeh_aux={getattr(config, 'lambda_aeh_aux', 0.5)}  "
        f"(additive local-evidence head; logit=scale*sum M_frame*M_t*e+bias; "
        f"faithful by construction; gamma 0->1; M_t_up rebinds to M_t*e)"
    )
    if _aeh_on:
        print(
            f"[Phase40] aeh_topk_frac={getattr(config, 'aeh_topk_frac', 0.0)} "
            f"(EXP-1 hard bottleneck)  "
            f"lambda_aeh_concentrate={getattr(config, 'lambda_aeh_concentrate', 0.0)}  "
            f"lambda_aeh_suff={getattr(config, 'lambda_aeh_suff', 0.0)}  "
            f"aeh_suff_topk_frac={getattr(config, 'aeh_suff_topk_frac', 0.0)} "
            f"(EXP-2 concentrate evidence -> insertion sufficiency)"
        )
        print(
            f"[Phase41] lambda_aeh_topk_mass={getattr(config, 'lambda_aeh_topk_mass', 0.0)}  "
            f"aeh_mass_topk_frac={getattr(config, 'aeh_mass_topk_frac', 0.15)}  "
            f"lambda_aeh_temporal_conc={getattr(config, 'lambda_aeh_temporal_conc', 0.0)}  "
            f"(SOFT concentration as PURE AUX losses; prediction NOT bottlenecked; "
            f"replaces P40 EXP-1 hard top-k which backfired -> more diffuse)"
        )
        _attn_freq = bool(getattr(config, "aeh_attn_freq", False))
        _freq_mode = str(getattr(config, "aeh_freq_mode", "srm"))
        _n_taps    = 6 if _freq_mode == "multiband" else 3
        print(
            f"[Phase44] aeh_attn_freq={_attn_freq}  aeh_freq_mode={_freq_mode} ({_n_taps} taps)  "
            f"lambda_calib={getattr(config, 'lambda_calib', 0.0)}  "
            f"(attn_freq=route spectral score into displayed M_t; multiband=richer "
            f"high-pass bank; calib=anchor the 0.5 operating point)"
        )

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
        "train_ins":           [],                      # Phase 24: insertion-AUC training loss
        "train_temp_sparse":   [],                      # Phase 25: temporal-gate sparsity
        "train_cbm_aux":       [],                      # Phase 26: CBM auxiliary cls loss
        "train_cbm_div":       [],                      # Phase 26: CBM slot diversity loss
        "train_cbm_main_aux":  [],                      # Phase 27: main_logit aux supervision
        "train_domain":        [],                      # Phase 27: DANN domain CE loss
        "train_domain_acc":    [],                      # Phase 27: DANN domain top-1 accuracy
        "train_del":           [],                      # Phase 30: deletion (necessity) loss
                                                        # on the D-pass — inverse bottleneck
                                                        # classified toward REAL
        "train_temp_band":     [],                      # Phase 30: bounded eff_fr hinge
        "train_spatial_band":  [],                      # Phase 31: bounded eff_sp hinge
        "train_eff_frames":    [],                      # Phase 29: effective frame count of
                                                        # M_frame (1/Herfindahl, max T)
        "train_eff_spatial":   [],                      # Phase 29: effective spatial tokens of
                                                        # M_t per frame (1/Herfindahl, max N)
        "train_slot_on_top":   [],                      # Phase 29: CBM slot-attention mass on
                                                        # the top-M_frame frame (prior adherence;
                                                        # 1/T ≈ 0.0625 = uniform / decoupled)
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

    # ── Per-epoch overhead controls ───────────────────────────────────────────
    _sanity_check_every = getattr(config, "sanity_check_every", 5)
    _val_every          = getattr(config, "val_every", 1)
    print(f"[Config] sanity_check_every={_sanity_check_every}  val_every={_val_every}  "
          f"snapshot_every={config.snapshot_every}")

    # Safe initial values so checkpoint / early-stop references work even when
    # validation is skipped on the very first epoch (val_every > 1).
    metrics          = {}
    diag_cosine      = 0.0
    diag_std         = 0.0
    _peak_mode_share = 1.0

    # ── Training loop ─────────────────────────────────────────────────────────
    total_batches = len(train_loader)
    epoch_w       = len(str(start_epoch + config.epochs))

    for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_acc = {
            "total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0,
            "faith": 0.0, "ins": 0.0,                   # Phase 24
            "del": 0.0, "temp_band": 0.0,                # Phase 30
            "spatial_band": 0.0,                         # Phase 31
            "temp_sparse": 0.0,                          # Phase 25
            "cbm_aux": 0.0, "cbm_div": 0.0,              # Phase 26
            "cbm_main_aux": 0.0,                          # Phase 27
            "domain": 0.0, "domain_acc": 0.0,             # Phase 27
            "eff_frames": 0.0, "eff_spatial": 0.0,        # Phase 29
            "slot_on_top": 0.0,                           # Phase 29
            "localize": 0.0, "sbi_cls": 0.0, "n_sbi": 0,  # Phase 33
            "kept_mass": 0.0,                             # Phase 34
            "suff_kept": 0.0, "lens_gate": 0.0,          # Phase 35 (dual-lens)
            "decomp_div": 0.0, "decomp_overlap": 0.0,    # Phase 36
            "decomp_gate": 0.0,                           # Phase 36
            "decomp_share": np.zeros(_decomp_L),         # Phase 36 per-layer shares
            "sparse": 0.0, "peak_spread": 0.0, "sharp": 0.0, "n": 0,
            "n_b": 0, "n_d": 0,                           # Phase 30: pass counts
        }

        LOG_EVERY = 1000
        run = {
            "total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0,
            "cons": 0.0, "faith": 0.0, "ins": 0.0,      # Phase 24
            "del": 0.0, "temp_band": 0.0,                # Phase 30
            "spatial_band": 0.0,                         # Phase 31
            "temp_sparse": 0.0,                          # Phase 25
            "cbm_aux": 0.0, "cbm_div": 0.0,              # Phase 26
            "cbm_main_aux": 0.0,                          # Phase 27
            "domain": 0.0, "domain_acc": 0.0,             # Phase 27
            "eff_frames": 0.0, "eff_spatial": 0.0,        # Phase 29
            "slot_on_top": 0.0,                           # Phase 29
            "sparse": 0.0, "peak_spread": 0.0,
            "sharp": 0.0, "n": 0,
            "n_b": 0, "n_d": 0,                           # Phase 30: pass counts
        }

        for batch_idx, batch in enumerate(train_loader):
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(_dev_str, enabled=_use_amp, dtype=_amp_dtype):
                # ── Phase 27: DANN warmup (lambda_grl + lambda_domain) ────────
                # Linear ramp from 0 to target over domain_warmup_epochs so the
                # domain head learns a signal before GRL starts reversing
                # useful gradient.
                lam_grl_eff = domain_warmup(
                    epoch,
                    int(getattr(config, "domain_warmup_epochs", 3)),
                    float(getattr(config, "lambda_grl", 1.0)),
                )
                lam_dom_eff = domain_warmup(
                    epoch,
                    int(getattr(config, "domain_warmup_epochs", 3)),
                    float(getattr(config, "lambda_domain", 0.10)),
                )

                # ── Phase 39: additive-head blend gamma ramp (0 -> 1) ─────────
                # Set the model's gamma buffer BEFORE the forward so out_A.logit
                # uses it.  Ramps over aeh_warmup_epochs (detection-protective: the
                # proven base head carries the prediction while the additive head
                # learns), then holds at 1.0 so the FAITHFUL head is the predictor.
                if getattr(config, "aeh_enabled", False) and hasattr(model, "aeh_gamma_current"):
                    _aeh_gamma_eff = _faith_warmup(
                        epoch, int(getattr(config, "aeh_warmup_epochs", 2)),
                        float(getattr(config, "aeh_gamma_max", 1.0)))
                    model.aeh_gamma_current.fill_(float(_aeh_gamma_eff))

                # ── Pass A: normal forward (with GRL strength for DANN) ───────
                out_A    = model(frames, lambda_grl=lam_grl_eff)
                logits_A = out_A.logit
                M_t      = out_A.M_t          # (B, T, h, w) softmax from EarlyAttnHead
                M_t_logits = out_A.M_t_logits  # (B, T, h, w) pre-softmax raw scores
                # Phase 35: B-pass (sufficiency) targets M_suff; equals M_t when
                # dual_lens is OFF, so the single-lens path is byte-identical.
                M_suff   = out_A.M_suff
                loss_cls = cls_loss_fn(logits_A, labels)

                # ── Phase 39: aux focal-cls on the additive head's own logit ──
                # Trains aeh_score/scale/bias from epoch 1 even while the blend
                # gamma is still small, so the faithful head is a competent
                # classifier by the time gamma reaches 1.0 and it becomes the
                # sole predictor.  Zero when the head is OFF.
                if getattr(config, "aeh_enabled", False) and (out_A.aeh_logit is not None):
                    loss_aeh_aux = cls_loss_fn(out_A.aeh_logit, labels)
                else:
                    loss_aeh_aux = torch.zeros((), device=frames.device)

                # ── Phase 40: evidence-CONCENTRATION losses (EXP-2) ───────────
                # (a) concentrate: minimise the entropy of the normalised positive
                #     contribution ReLU(M_t*e) over cells, on FAKE clips only, so a
                #     few cells carry the fake-evidence -> insertion SUFFICIENCY.
                # (b) sufficiency: the top-k% partial logit (aeh_topk_logit) must
                #     already classify correctly -> trains the insertion objective
                #     directly.  Both weights 0.0 by default (exact Phase 39).
                loss_aeh_conc = torch.zeros((), device=frames.device)
                loss_aeh_suff = torch.zeros((), device=frames.device)
                if getattr(config, "aeh_enabled", False) and (out_A.aeh_contrib is not None):
                    if float(getattr(config, "lambda_aeh_concentrate", 0.0)) > 0.0:
                        _pos  = torch.relu(out_A.aeh_contrib)                 # (B,T,N)
                        _den  = _pos.sum(dim=2, keepdim=True).clamp_min(1e-6)
                        _p    = _pos / _den
                        _ent  = -(_p * (_p + 1e-9).log()).sum(dim=2)         # (B,T)
                        _entw = (_ent * out_A.M_frame).sum(dim=1)            # (B,) frame-weighted
                        _fakem = (labels == 1).float()                       # (B,) fake-only
                        loss_aeh_conc = (_entw * _fakem).sum() / _fakem.sum().clamp_min(1.0)
                    if (float(getattr(config, "lambda_aeh_suff", 0.0)) > 0.0
                            and out_A.aeh_topk_logit is not None):
                        loss_aeh_suff = cls_loss_fn(out_A.aeh_topk_logit, labels)

                # ── Phase 41: SOFT concentration as PURE AUX losses ───────────
                # Replaces P40 EXP-1's hard top-k STE bottleneck (which made
                # evidence MORE diffuse — the dense backward let the model
                # equalise cells to game the mask).  These add gradient pressure
                # on the M_t*e / M_frame DISTRIBUTIONS without touching the dense
                # prediction, so detection (the gamma-blend logit) is unchanged.
                # (a) spatial top-k mass: maximise the share of the top-k cells in
                #     the normalised ReLU(M_t*e) -> few cells SUFFICIENT.  Cannot be
                #     gamed by equalising (uniform -> top-k mass = k/N, the MINIMUM).
                # (b) temporal concentration: minimise M_frame entropy over frames
                #     -> a few frames carry the evidence (sharper k-drop).  All
                #     fake-gated; weights 0.0 by default = exact Phase 39/40.
                loss_aeh_mass  = torch.zeros((), device=frames.device)
                loss_aeh_tconc = torch.zeros((), device=frames.device)
                if getattr(config, "aeh_enabled", False) and (out_A.aeh_contrib is not None):
                    _fakem = (labels == 1).float()                       # (B,) fake-only
                    _fden  = _fakem.sum().clamp_min(1.0)
                    if float(getattr(config, "lambda_aeh_topk_mass", 0.0)) > 0.0:
                        _posm = torch.relu(out_A.aeh_contrib)            # (B,T,N)
                        _Nn   = _posm.shape[2]
                        _kk   = max(1, int(round(
                            float(getattr(config, "aeh_mass_topk_frac", 0.15)) * _Nn)))
                        _pm   = _posm / _posm.sum(dim=2, keepdim=True).clamp_min(1e-6)
                        _tkm  = _pm.topk(_kk, dim=2).values.sum(dim=2)   # (B,T) top-k share
                        _tkmw = (_tkm * out_A.M_frame).sum(dim=1)        # (B,) frame-weighted
                        loss_aeh_mass = ((1.0 - _tkmw) * _fakem).sum() / _fden
                    if float(getattr(config, "lambda_aeh_temporal_conc", 0.0)) > 0.0:
                        _mf   = out_A.M_frame.clamp_min(1e-6)            # (B,T)
                        _mfp  = _mf / _mf.sum(dim=1, keepdim=True)
                        _tent = -(_mfp * (_mfp + 1e-9).log()).sum(dim=1) # (B,)
                        loss_aeh_tconc = (_tent * _fakem).sum() / _fden

                # Phase 30: B-pass and D-pass ALTERNATE by step parity so the
                # per-step forward count stays at 2 (A + one bottleneck pass).
                # With grad_accum_steps=8 each optimizer step still averages
                # 4 B-batches and 4 D-batches — both signals land every update.
                _is_del_step = bool(
                    config.phase21_enabled
                    and float(getattr(config, "lambda_del", 0.0)) > 0.0
                    and (batch_idx % 2 == 1)
                )
                _ins_frac = 0.0   # Phase 32: per-B-step hard keep fraction (set in B-pass)
                if config.phase21_enabled and not _is_del_step:
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
                    # ── Phase 32: B-pass sufficiency hard-mask ────────────────
                    # Sample the keep fraction per B-step in [lo, hi] so loss_ins
                    # trains the insertion metric's HARD top-k pixel reveal across
                    # the steep low-reveal range, not a single point.  lo=hi=0 =
                    # P31 soft behaviour (peak_floor path).  The D-pass below uses
                    # bottleneck_hard_topk_frac (0.0 = soft), so the met deletion
                    # number is structurally untouched by this axis.
                    _ins_lo = float(getattr(config, "ins_hard_topk_frac_lo", 0.0))
                    _ins_hi = float(getattr(config, "ins_hard_topk_frac_hi", 0.0))
                    if _ins_hi > 0.0:
                        _lo_c     = min(max(_ins_lo, 1e-3), _ins_hi)
                        _ins_frac = float(
                            torch.empty(1, device=frames.device)
                            .uniform_(_lo_c, _ins_hi).item()
                        )
                    else:
                        _ins_frac = 0.0
                    x_b = build_bottlenecked_input(
                        frames, M_suff,                                 # Phase 35: sufficiency lens
                        blur_kernel=config.blur_kernel,
                        blur_sigma=config.blur_sigma,
                        peak_floor=config.bottleneck_peak_floor,        # Phase 22 (soft path only)
                        hard_topk_frac=_ins_frac,                       # Phase 32: B-pass hard top-K
                    )
                    out_B       = model(x_b)           # GRAD ENABLED (v2 fix)
                    loss_faith  = faithfulness_loss(logits_A, out_B.logit)
                    # ── Phase 24: insertion-AUC training loss ─────────────────
                    # Re-use the existing bottleneck forward.  Where loss_faith
                    # only requires logits_B to MATCH logits_A (which the model
                    # can satisfy without M_t identifying the right pixels),
                    # loss_ins requires logits_B to be CORRECT — forcing M_t to
                    # select pixels whose preservation alone is sufficient for
                    # classification.  This is the insertion-AUC objective.
                    # ZERO additional forward passes (out_B already computed).
                    # Phase 25: combined with hard_topk_frac=0.20 above, x_b
                    # now keeps EXACTLY the top-K pixels and blurs the rest —
                    # the same construction the insertion-AUC metric uses at
                    # eval time, so this loss directly minimises the metric.
                    loss_ins    = cls_loss_fn(out_B.logit, labels)
                    loss_del    = torch.zeros((), device=frames.device)
                    # ── Phase 25: temporal sparsity on M_frame ────────────────
                    # Pushes the temporal_gate (Phase 23) toward a peaky
                    # distribution. Without this, M_frame stayed near-uniform
                    # (e.g. 6-9-26 0800hrs sample 2: values 0.033–0.079 vs
                    # uniform 0.0625), which pinned k1/k2/k4 ratios below 1.0.
                    loss_temp_sparse = temporal_sparsity_loss(out_A.M_frame)
                elif config.phase21_enabled:
                    # ── Phase 31 Pass D: DELETION (necessity), detoxed ────────
                    # P30 post-mortem (run 6-12-26 1300hrs): BCE-to-0 on ALL
                    # samples demanded CERTAIN-REAL on fakes whose erased
                    # footprint was ~2/49 cells — 96% of the face still visible
                    # — on half of all steps.  That is systematic "visible fake
                    # evidence → REAL" supervision: train_clean fake_acc fell to
                    # 0.40, val fake_acc hit 0.082 the first epoch the warmup
                    # reached full strength (E4), test AUC 0.879 → 0.796.  The
                    # necessity SIGNAL worked (del_gain_over_random +0.098, the
                    # first positive ever) — only the loss form was poisonous.
                    #
                    # Three sub-passes now:
                    #  ANCHOR (every del_anchor_every-th batch): x_d = full blur,
                    #    target REAL for all.  No evidence visible ⇒ REAL is the
                    #    correct label for BOTH classes; anchors the blur end of
                    #    the del/ins eval curves (P30 blurred_conf 0.40 floored
                    #    deletion AUC at ~0.40).  No M_t dependence — pure
                    #    classifier shaping; the B-pass ("blur canvas + sharp
                    #    top region → TRUE label") trains the opposing case so
                    #    blur itself cannot become a class marker.
                    #  ERASE, real samples: cls_loss_fn(logit_D, 0) — REAL is
                    #    the true label (blur-spot augmentation; also kills the
                    #    "blur present → fake" shortcut direction).
                    #  ERASE, fake samples: hinge relu(logit_D − margin), only
                    #    on fakes the A-pass currently detects (p_A > 0.5).
                    #    Pushes p(fake|evidence erased) to ≤ sigmoid(margin)
                    #    then STOPS — necessity without certainty-REAL.  The
                    #    detected-gate skips fakes the model cannot classify
                    #    anyway (no point flipping an undetected sample, and
                    #    early in training that is most of them).
                    _is_anchor_step = bool(
                        int(getattr(config, "del_anchor_every", 8)) > 0
                        and (batch_idx % int(getattr(config, "del_anchor_every", 8))
                             == int(getattr(config, "del_anchor_every", 8)) - 1)
                    )
                    if _is_anchor_step:
                        x_d = full_blur_input(
                            frames,
                            blur_kernel=config.blur_kernel,
                            blur_sigma=config.blur_sigma,
                        )
                        out_D    = model(x_d)
                        loss_del = cls_loss_fn(
                            out_D.logit, torch.zeros_like(labels)
                        )
                    else:
                        x_d = build_bottlenecked_input(
                            frames, M_t,
                            blur_kernel=config.blur_kernel,
                            blur_sigma=config.blur_sigma,
                            peak_floor=config.bottleneck_peak_floor,
                            hard_topk_frac=float(getattr(
                                config, "bottleneck_hard_topk_frac", 0.0
                            )),
                            invert=True,
                        )
                        out_D = model(x_d)
                        _lbl_f       = labels.float().view_as(out_D.logit)
                        _real_mask   = _lbl_f < 0.5
                        _detected    = (torch.sigmoid(
                            logits_A.detach().view_as(out_D.logit)) > 0.5)
                        _fake_mask   = (_lbl_f >= 0.5) & _detected
                        _del_margin  = float(getattr(
                            config, "del_margin_logit", 0.0))
                        _del_terms   = []
                        if _real_mask.any():
                            _del_terms.append(cls_loss_fn(
                                out_D.logit[_real_mask],
                                torch.zeros_like(out_D.logit[_real_mask]),
                            ))
                        if _fake_mask.any():
                            _del_terms.append(F.relu(
                                out_D.logit[_fake_mask] - _del_margin
                            ).mean())
                        loss_del = (torch.stack(_del_terms).sum()
                                    if _del_terms
                                    else torch.zeros((), device=frames.device))
                    loss_temp_sparse = temporal_sparsity_loss(out_A.M_frame)
                    loss_faith       = torch.zeros((), device=frames.device)
                    loss_ins         = torch.zeros((), device=frames.device)
                else:
                    loss_faith       = torch.zeros((), device=frames.device)
                    loss_ins         = torch.zeros((), device=frames.device)
                    loss_del         = torch.zeros((), device=frames.device)
                    loss_temp_sparse = torch.zeros((), device=frames.device)

                # ── Phase 31: sparsity is DIAGNOSTIC-ONLY now ─────────────────
                # Computed every phase21 step for log continuity; run with
                # lambda_sparse=0.0 — the -peak reward is an open-ended ratchet
                # that squeezed eff_sp from 3.4 (P29) to 1.8/49 cells (P30) and
                # starved detection.  spatial_band_loss below is the bounded
                # replacement.
                if config.phase21_enabled:
                    loss_sparse = sparsity_loss(M_t)
                else:
                    loss_sparse = torch.zeros((), device=frames.device)

                # ── Phase 30: bounded temporal band (every step, A-pass only) ──
                # Hinge on eff_fr = 1/Σp²: penalty above temp_band_target, ZERO
                # below — concentrates M_frame to ~target effective frames and
                # then lets go (no one-hot ratchet; that was the P27 failure).
                if (config.phase21_enabled
                        and float(getattr(config, "lambda_temp_band", 0.0)) > 0.0):
                    loss_temp_band = temporal_band_loss(
                        out_A.M_frame,
                        float(getattr(config, "temp_band_target", 6.0)),
                    )
                else:
                    loss_temp_band = torch.zeros((), device=frames.device)

                # ── Phase 31: bounded spatial band (every step, A-pass only) ──
                # Two-sided hinge on per-frame eff_sp = 1/Σp² over the 49
                # cells: penalty above spatial_band_hi (too diffuse — B-pass
                # needs a concentrated keep-region) and below spatial_band_lo
                # (too concentrated — P30 collapsed to 1.8 cells, starving
                # detection and reducing the insertion ordering to ~2 cells of
                # signal).  Zero gradient inside the band.
                if (config.phase21_enabled
                        and float(getattr(config, "lambda_spatial_band", 0.0)) > 0.0):
                    loss_spatial_band = spatial_band_loss(
                        M_t,
                        lo=float(getattr(config, "spatial_band_lo", 4.0)),
                        hi=float(getattr(config, "spatial_band_hi", 10.0)),
                    )
                else:
                    loss_spatial_band = torch.zeros((), device=frames.device)

                # ── Phase 26+27: CBM auxiliary losses ─────────────────────────
                # Phase 27 serial mode:
                #   out.logit IS cbm_logit, so loss_cls already trains the CBM.
                #   loss_cbm_aux is therefore redundant with loss_cls but we
                #   keep computing it (zero its weight via --lambda_cbm_aux 0
                #   on the CLI if you want a clean ablation).
                # Phase 26 parallel mode:
                #   out.logit is the blended logit; loss_cbm_aux supervises
                #   the CBM path alone so concepts learn even when blend
                #   weighs them low.
                #
                # loss_cbm_div: K slot attention vectors should attend to
                # different positions (push slots apart).  Independent of mode.
                #
                # Phase 27 main_aux: supervise main_logit even though it's
                # never used for prediction in serial mode. Acts as a
                # regulariser ("attn_pool should still be classifiable")
                # and as a diagnostic of how much information lives in the
                # M_t-gated pool.
                if getattr(config, "cbm_enabled", True) and out_A.cbm_logit is not None:
                    loss_cbm_aux      = cls_loss_fn(out_A.cbm_logit, labels)
                    loss_cbm_div      = cbm_diversity_loss(out_A.slot_attn)
                    loss_cbm_main_aux = cls_loss_fn(out_A.main_logit, labels)
                else:
                    loss_cbm_aux      = torch.zeros((), device=frames.device)
                    loss_cbm_div      = torch.zeros((), device=frames.device)
                    loss_cbm_main_aux = torch.zeros((), device=frames.device)

                # ── Phase 27: DANN domain CE loss ─────────────────────────────
                # Only the samples with domain_id >= 0 contribute. Val/test
                # batches default to -1 sentinel and are filtered.
                if (getattr(config, "dann_enabled", True)
                        and out_A.domain_logits is not None
                        and "domain" in batch):
                    _dom_labels = batch["domain"].to(device, non_blocking=True)
                    _valid_mask = (_dom_labels >= 0)
                    if _valid_mask.any():
                        loss_domain = F.cross_entropy(
                            out_A.domain_logits[_valid_mask],
                            _dom_labels[_valid_mask],
                        )
                        _dom_acc = domain_accuracy(out_A.domain_logits, _dom_labels)
                    else:
                        loss_domain = torch.zeros((), device=frames.device)
                        _dom_acc    = float("nan")
                else:
                    loss_domain = torch.zeros((), device=frames.device)
                    _dom_labels = None
                    _dom_acc    = float("nan")

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

                # ── Phase 36: decomposition layer-diversity loss ──────────────
                # Push the L layer maps to be COMPLEMENTARY (cover different
                # regions) so the decomposition is a real multi-region partition,
                # not L copies of one blob.  Mean off-diagonal pairwise inner
                # product of the L softmax maps; minimised.  Parallel mode only
                # (sequential separates by construction via suppression -> run
                # with lambda_decomp_div 0).  Zero when decomp is OFF.
                if (out_A.M_layers is not None) and (_lam_decomp_div > 0.0):
                    _ml   = out_A.M_layers                          # (B, T, L, N)
                    _Ld   = _ml.shape[2]
                    _gram = torch.einsum("btln,btmn->btlm", _ml, _ml)  # (B,T,L,L)
                    _diag = _gram.diagonal(dim1=-2, dim2=-1).sum(-1)    # (B, T)
                    loss_decomp_div = (
                        (_gram.sum(dim=(-1, -2)) - _diag)
                        / max(_Ld * (_Ld - 1), 1)
                    ).mean()
                else:
                    loss_decomp_div = torch.zeros((), device=frames.device)

                # ── Loss weighting ────────────────────────────────────────────
                _global_step = (epoch - 1) * len(train_loader) + batch_idx
                _lambda1_eff = config.lambda1 * min(1.0, _global_step / 200.0)
                lam_faith_eff = _faith_warmup(
                    epoch, config.faith_warmup_epochs, config.lambda_faith
                )
                # Phase 24: re-use same warmup curve for lambda_ins.  At random
                # init, loss_ins ≈ loss_cls (~0.15) — without warmup it would
                # dominate and destabilise early training.  Linear ramp over
                # faith_warmup_epochs gives M_t time to start peaking before
                # the insertion objective gets full weight.
                lam_ins_eff = _faith_warmup(
                    epoch, config.faith_warmup_epochs,
                    float(getattr(config, "lambda_ins", 0.5)),
                )
                # Phase 25: temporal sparsity uses the same warmup curve.
                # Without warmup it would dominate epoch 1 and collapse M_frame
                # onto frame 0 before the classifier learns to discriminate.
                lam_temp_sparse_eff = _faith_warmup(
                    epoch, config.faith_warmup_epochs,
                    float(getattr(config, "lambda_temp_sparse", 0.05)),
                )
                # Phase 30: deletion (necessity) loss — same warmup as faith/ins
                # so M_t has time to find evidence before being held responsible
                # for ALL of it.
                lam_del_eff = _faith_warmup(
                    epoch, config.faith_warmup_epochs,
                    float(getattr(config, "lambda_del", 0.0)),
                )
                # Phase 30: temporal band — same warmup; the hinge is already
                # bounded but ramping avoids an epoch-1 shock to temporal_gate.
                lam_tband_eff = _faith_warmup(
                    epoch, config.faith_warmup_epochs,
                    float(getattr(config, "lambda_temp_band", 0.0)),
                )
                # Phase 31: spatial band — same warmup; the lower hinge must
                # not fight the early concentration that loss_ins needs to get
                # the peak above the 0.25 floor, so it ramps in alongside it.
                lam_sband_eff = _faith_warmup(
                    epoch, config.faith_warmup_epochs,
                    float(getattr(config, "lambda_spatial_band", 0.0)),
                )

                # Phase 26: CBM weight hyperparameters (no warmup — CBM is a
                # parallel head that benefits from learning concept structure
                # from epoch 1).  Defaults: lambda_cbm_aux=0.10, lambda_cbm_div=0.05.
                _lam_cbm_aux      = float(getattr(config, "lambda_cbm_aux", 0.10))
                _lam_cbm_div      = float(getattr(config, "lambda_cbm_div", 0.05))
                # Phase 27 auxiliary supervision on main_logit (not in prediction path)
                _lam_cbm_main_aux = float(getattr(config, "lambda_cbm_main_aux", 0.05))

                l_total = (loss_cls
                           + lam_faith_eff          * loss_faith
                           + lam_ins_eff            * loss_ins
                           + lam_del_eff             * loss_del          # Phase 30/31
                           + config.lambda_sparse    * loss_sparse
                           + lam_temp_sparse_eff     * loss_temp_sparse  # Phase 25
                           + lam_tband_eff           * loss_temp_band    # Phase 30
                           + lam_sband_eff           * loss_spatial_band # Phase 31
                           + _lambda1_eff            * l_exp
                           + config.lambda2          * l_temp
                           + _lambda_peak_spread     * l_peak_spread
                           + _lambda_sharp           * loss_sharp
                           + _lam_cbm_aux            * loss_cbm_aux       # Phase 26
                           + _lam_cbm_div            * loss_cbm_div       # Phase 26
                           + _lam_cbm_main_aux       * loss_cbm_main_aux  # Phase 27
                           + lam_dom_eff             * loss_domain        # Phase 27
                           + _lam_decomp_div         * loss_decomp_div    # Phase 36
                           + float(getattr(config, "lambda_aeh_aux", 0.5))
                                                     * loss_aeh_aux       # Phase 39
                           + float(getattr(config, "lambda_aeh_concentrate", 0.0))
                                                     * loss_aeh_conc      # Phase 40
                           + float(getattr(config, "lambda_aeh_suff", 0.0))
                                                     * loss_aeh_suff      # Phase 40
                           + float(getattr(config, "lambda_aeh_topk_mass", 0.0))
                                                     * loss_aeh_mass      # Phase 41
                           + float(getattr(config, "lambda_aeh_temporal_conc", 0.0))
                                                     * loss_aeh_tconc)    # Phase 41

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

                # ── Phase 44: operating-point calibration regularizer ─────────
                # Pins the batch mean predicted prob to the batch fake-rate so the
                # 0.5 decision threshold stops drifting epoch-to-epoch (the val
                # fake/real-acc oscillation; optimal_threshold swung 0.31-0.71).
                # It anchors the score OFFSET only; ranking (AUC) is untouched.
                # 0.0 = off (byte-identical).
                _lambda_calib = float(getattr(config, "lambda_calib", 0.0))
                if _lambda_calib > 0.0:
                    _p_mean = torch.sigmoid(logits_A).mean()
                    _y_mean = labels.float().mean()
                    l_calib = (_p_mean - _y_mean) ** 2
                    l_total = l_total + _lambda_calib * l_calib
                else:
                    l_calib = torch.tensor(0.0)

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

            # ── Phase 33: self-blend + boundary-supervised attention (aux) ─────
            # Runs AFTER the main backward so the A/B/D graphs are already freed
            # -- the SBI forward graph never coexists with them, so peak VRAM
            # stays at the A+B level (no OOM cost on the T4).  Bounded to once per
            # optimizer step (batch_idx % stride); its gradient accumulates into
            # the SAME optimizer step (no /grad_accum, so lambda_localize lands at
            # full per-step weight).  The A/B/D detection regime is UNTOUCHED, so
            # the met detection/deletion numbers are structurally protected.
            # Generation is fp32 (autocast off) so grid_sample/affine_grid are
            # stable; the model forward re-enters autocast.
            loss_sbi_cls  = torch.zeros((), device=frames.device)
            loss_localize = torch.zeros((), device=frames.device)
            _sbi_ran = False
            _sbi_n   = 0
            if _sbi_active and (batch_idx % _sbi_stride == 0):
                _real_sel = (labels.reshape(-1) < 0.5)
                if bool(_real_sel.any()):
                    with autocast(_dev_str, enabled=False):
                        _sbi_frames, _sbi_boundary, _sbi_fmask = make_sbi_batch(
                            frames[_real_sel].float(),
                            blend_lo=_sbi_lo, blend_hi=_sbi_hi,
                            modes=_sbi_modes,                          # Phase 35: artifact families
                            partial_lo=_sbi_plo, partial_hi=_sbi_phi,  # Phase 35: temporally-partial
                            freq_mismatch=_sbi_freq,                   # Phase 37: resolution seam cue
                        )
                    with autocast(_dev_str, enabled=_use_amp, dtype=_amp_dtype):
                        out_sbi       = model(_sbi_frames)
                        _sbi_lbl      = torch.ones(
                            _sbi_frames.shape[0], device=frames.device
                        )
                        loss_sbi_cls  = cls_loss_fn(out_sbi.logit, _sbi_lbl)
                        # Phase 35: seam -> necessity lens (out_sbi.M_t = M_nec),
                        # weighted by frame_mask so clean frames of a partial fake
                        # are not pulled onto a seam they do not carry.
                        loss_localize = localization_loss(
                            out_sbi.M_t, _sbi_boundary, frame_weight=_sbi_fmask)
                        _lam_loc_eff  = _faith_warmup(
                            epoch, config.faith_warmup_epochs, _lambda_localize
                        )
                        sbi_total = (_lam_loc_eff * loss_localize
                                     + _lambda_sbi_cls * loss_sbi_cls)
                    if torch.isfinite(sbi_total):
                        scaler.scale(sbi_total).backward()
                        _sbi_ran = True
                        _sbi_n   = int(_sbi_frames.shape[0])
                    else:
                        print(f"[NaNGuard-P33] non-finite SBI loss at epoch={epoch} "
                              f"batch={batch_idx} — skipped.")

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
                      f"L_ins={loss_ins.item():.6f}  "
                      f"L_sparse={loss_sparse.item():.6f}  "
                      f"L_temp_sparse={loss_temp_sparse.item():.6f}  "       # Phase 25
                      f"L_peak_spread={l_peak_spread.item():.6f}  "
                      f"L_sharp={loss_sharp.item():.6f}")
                print(f"[DIAG] lam_faith_eff={lam_faith_eff:.4f}  "
                      f"lam_ins_eff={lam_ins_eff:.4f}  "
                      f"lam_temp_sparse_eff={lam_temp_sparse_eff:.4f}  "      # Phase 25
                      f"lambda_peak_spread={_lambda_peak_spread}  "
                      f"lambda_sharp={_lambda_sharp}  "
                      f"lambda_sparse={config.lambda_sparse}")
                # v4: log early_attn_tau (the tau that actually controls M_t sharpness)
                # NOT cross_attention.log_temp which is a dead legacy module
                print(f"[DIAG] early_attn_tau={out_A.early_attn_tau:.4f}  "
                      f"(log_tau={model.early_attn.log_tau.item():.4f})")
                # Phase 25: bi-directional refinement gate diagnostic
                _alpha_print = float(getattr(out_A, "refine_alpha", 0.0))
                _bidir_on    = bool(getattr(config, "bidirectional_enabled", True))
                _hard_topk   = float(getattr(config, "bottleneck_hard_topk_frac", 0.0))
                print(f"[DIAG-P25] bidirectional={_bidir_on}  refine_alpha={_alpha_print:.4f}  "
                      f"refine_gate={model.refine_gate.item():.4f}  "
                      f"hard_topk_frac={_hard_topk:.3f}  "
                      f"frame_attn_tau={out_A.frame_attn_tau:.4f}")
                # Phase 26: CBM diagnostic
                _cbm_on    = bool(getattr(config, "cbm_enabled", True))
                _cbm_blend = float(getattr(out_A, "cbm_blend", 0.0))
                _cbm_K     = int(getattr(config, "cbm_num_slots", 8))
                _cbm_serial = bool(getattr(out_A, "cbm_serial", False))
                print(f"[DIAG-P26] cbm_enabled={_cbm_on}  K={_cbm_K}  serial={_cbm_serial}  "
                      f"L_cbm_aux={loss_cbm_aux.item():.6f}  "
                      f"L_cbm_div={loss_cbm_div.item():.6f}  "
                      f"L_cbm_main_aux={loss_cbm_main_aux.item():.6f}  "
                      f"cbm_blend={_cbm_blend:.4f}  "
                      f"class_balanced_sampler={_use_cb_sampler}")
                # Phase 27: DANN diagnostic
                _dann_on    = bool(getattr(config, "dann_enabled", True))
                _num_dom    = int(getattr(config, "num_domains", 4))
                print(f"[DIAG-P27] dann_enabled={_dann_on}  num_domains={_num_dom}  "
                      f"L_domain={loss_domain.item():.6f}  "
                      f"domain_acc={_dom_acc:.4f}  "
                      f"lam_grl_eff={lam_grl_eff:.4f}  "
                      f"lam_dom_eff={lam_dom_eff:.4f}")
                # Phase 28: CBM-attention coupling diagnostic.  w = M_t ⊙ M_frame
                # (max-normalised) scales the CBM input tokens; eff_tokens =
                # 1/Σ(ŵ²) on the sum-normalised weights ≈ how many of the T*N
                # tokens the prediction can effectively see.  If this collapses
                # to ~1 the bottleneck is too brutal; if it stays ≈ T*N the
                # coupling is not biting.
                _coupled = bool(getattr(out_A, "cbm_coupled", False))
                with torch.no_grad():
                    _Bd, _Td = out_A.M_frame.shape
                    _w_diag = (out_A.M_t.reshape(_Bd, _Td, -1)
                               * out_A.M_frame.unsqueeze(-1)).reshape(_Bd, -1)
                    _w_sum  = _w_diag / _w_diag.sum(dim=1, keepdim=True).clamp(min=1e-12)
                    _eff_tokens = float((1.0 / (_w_sum.pow(2).sum(dim=1)
                                                .clamp(min=1e-12))).mean().item())
                    _w_peak = float(_w_diag.amax(dim=1).mean().item())
                print(f"[DIAG-P28] cbm_coupled={_coupled}  "
                      f"w_peak={_w_peak:.4f}  "
                      f"eff_tokens={_eff_tokens:.1f}/{_w_diag.shape[1]}  "
                      f"M_frame_peak={float(out_A.M_frame.amax(dim=-1).mean().item()):.4f}")
                # Phase 29: pooled-CBM coupling diagnostic.
                #   eff_frames  = 1/Herfindahl(M_frame) — frames the prediction
                #                 effectively sees (max T).
                #   eff_spatial = mean 1/Herfindahl(M_t per frame) — spatial
                #                 tokens per frame the pooling keeps (max N).
                #   slot_on_top = CBM slot-attention mass on the top-M_frame
                #                 frame (pooled mode) — prior-adherence probe.
                #   logit_std   = THE Phase 28 lesson: a healthy head varies
                #                 across samples; P28's starved head emitted a
                #                 constant (cls frozen at 0.1838 for 9 epochs).
                _pooled_diag = bool(getattr(out_A, "cbm_pooled", False))
                with torch.no_grad():
                    _eff_fr_d = float((1.0 / out_A.M_frame.pow(2).sum(dim=1)
                                       .clamp(min=1e-12)).mean().item())
                    _mt_d = out_A.M_t.reshape(_Bd, _Td, -1)
                    _mt_d = _mt_d / _mt_d.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                    _eff_sp_d = float((1.0 / _mt_d.pow(2).sum(dim=-1)
                                       .clamp(min=1e-12)).mean().item())
                    if (out_A.slot_attn is not None
                            and out_A.slot_attn.shape[-1] == _Td):
                        _topf_d = out_A.M_frame.argmax(dim=1)
                        _stop_d = float(out_A.slot_attn[
                            torch.arange(_Bd, device=_topf_d.device), :, _topf_d
                        ].mean().item())
                    else:
                        _stop_d = float("nan")
                    _lstd_d = float(out_A.logit.detach().float().std().item())
                print(f"[DIAG-P29] cbm_pooled={_pooled_diag}  "
                      f"eff_frames={_eff_fr_d:.1f}/{_Td}  "
                      f"eff_spatial={_eff_sp_d:.1f}/{_mt_d.shape[-1]}  "
                      f"slot_on_top={_stop_d:.3f} (uniform={1.0/_Td:.3f})  "
                      f"logit_std={_lstd_d:.2e}")
                if _lstd_d < 1e-4:
                    print("[DIAG-P29][WARNING] logit std < 1e-4 across the "
                          "batch — the prediction head is emitting a "
                          "near-constant output (Phase 28 failure signature); "
                          "training will not learn from this state.")
                # Phase 30: slot-scale repair + deletion pass + temporal band.
                #   slot_q_std  — must be ~1.0 (the P29 collapse cause was 0.02:
                #                 slot logits 100× below the log-prior).
                #   slot_logit_std — std of the CONTENT term of the slot-attn
                #                 logits across slots; must be same order as the
                #                 prior spread (~1) for slots to differentiate.
                #   L_cbm_div at init should be WELL below 1.0 now (random unit
                #                 queries → distinct rows). Pinned ≥0.95 = bug.
                _lam_del_cfg   = float(getattr(config, "lambda_del", 0.0))
                _lam_tband_cfg = float(getattr(config, "lambda_temp_band", 0.0))
                _tband_target  = float(getattr(config, "temp_band_target", 6.0))
                _sq_std = (float(model.cbm.slot_q.detach().float().std().item())
                           if getattr(model, "cbm", None) is not None else float("nan"))
                with torch.no_grad():
                    if (getattr(model, "cbm", None) is not None
                            and out_A.slot_attn is not None):
                        # std across slots of the attention logits' content term,
                        # reconstructed from slot_attn rows (post-prior). Row
                        # DISAGREEMENT is what we actually care about:
                        _row_std = float(out_A.slot_attn.std(dim=1).mean().item())
                    else:
                        _row_std = float("nan")
                print(f"[DIAG-P30] lambda_del={_lam_del_cfg}  "
                      f"lambda_temp_band={_lam_tband_cfg}  "
                      f"temp_band_target={_tband_target}  "
                      f"slot_q_std={_sq_std:.3f}  "
                      f"slot_attn_row_std={_row_std:.4f}  "
                      f"L_cbm_div={loss_cbm_div.item():.4f}  "
                      f"L_temp_band={loss_temp_band.item():.4f}")
                if _sq_std < 0.5:
                    print("[DIAG-P30][WARNING] slot_q std < 0.5 — slot queries "
                          "are too small to compete with the log(M_frame) "
                          "prior; all K slots will collapse onto the prior "
                          "shape (Phase 29 failure signature).")
                # Phase 31: D-pass detox + spatial band + class-conditional
                # focal alpha.
                #   del form     — fakes get hinge relu(logit−margin) gated on
                #                  A-pass detection; reals BCE-to-REAL (true
                #                  label); every del_anchor_every-th batch is a
                #                  full-blur anchor (target REAL, no M_t dep).
                #   L_spatial_band at init should be > 0 (init eff_sp ~17 > hi)
                #                  and fall toward 0 as eff_sp enters [lo, hi].
                #   alpha_pos/neg — fake/real error weights; -1 = legacy global.
                _lam_sband_cfg = float(getattr(config, "lambda_spatial_band", 0.0))
                _sband_lo      = float(getattr(config, "spatial_band_lo", 4.0))
                _sband_hi      = float(getattr(config, "spatial_band_hi", 10.0))
                _del_margin_d  = float(getattr(config, "del_margin_logit", 0.0))
                _anchor_every  = int(getattr(config, "del_anchor_every", 8))
                _ap = float(getattr(config, "focal_alpha_pos", -1.0))
                _an = float(getattr(config, "focal_alpha_neg", -1.0))
                print(f"[DIAG-P31] lambda_spatial_band={_lam_sband_cfg}  "
                      f"band=[{_sband_lo},{_sband_hi}]  "
                      f"L_spatial_band={loss_spatial_band.item():.4f}  "
                      f"del_margin_logit={_del_margin_d}  "
                      f"del_anchor_every={_anchor_every}  "
                      f"focal_alpha_pos={_ap}  focal_alpha_neg={_an}  "
                      f"lambda_sparse={config.lambda_sparse} (0.0 expected — "
                      f"superseded by spatial band)")
                if float(config.lambda_sparse) > 0.0 and _lam_sband_cfg > 0.0:
                    print("[DIAG-P31][WARNING] both lambda_sparse and "
                          "lambda_spatial_band are active — the -peak ratchet "
                          "will fight the band's lower hinge; set "
                          "--lambda_sparse 0.0.")
                # Phase 32: B-pass sufficiency hard-mask diagnostic.
                #   When hi>0 the B-pass keeps EXACTLY the top-k% pixels (hard STE
                #   mask) and blurs the rest — the same construction the insertion
                #   metric uses at eval.  L_ins should rise vs P31 first (the model
                #   now must call the class from a small region) then FALL across
                #   epochs as M_t learns to put SUFFICIENT evidence in its top
                #   cells.  The D-pass stays soft, so deletion (0.224 in P31) is
                #   protected.  Watch next run: ins_gain_over_random crossing 0,
                #   insertion_auc_fake_only rising from 0.388, faith corr from 0.318.
                _ins_lo_d = float(getattr(config, "ins_hard_topk_frac_lo", 0.0))
                _ins_hi_d = float(getattr(config, "ins_hard_topk_frac_hi", 0.0))
                _Hd, _Wd  = frames.shape[3], frames.shape[4]
                _kpx      = int(_ins_frac * _Hd * _Wd) if _ins_frac > 0 else 0
                print(f"[DIAG-P32] ins_hard_topk=[{_ins_lo_d},{_ins_hi_d}]  "
                      f"sampled_frac={_ins_frac:.3f}  "
                      f"K_px={_kpx}/{_Hd * _Wd}  "
                      f"L_ins={loss_ins.item():.4f}  "
                      f"(B-pass hard sufficiency; D-pass soft = deletion protected)")
                if _ins_hi_d <= 0.0:
                    print("[DIAG-P32] ins_hard_topk OFF (lo=hi=0) — soft B-pass "
                          "(P31 behaviour); insertion train/eval still misaligned.")
                # Phase 33: self-blend boundary-supervision diagnostic.  L_localize
                # is the soft cross-entropy of M_t against the blend seam: it
                # starts near log(49)=3.9 (uniform attention) and should FALL
                # across epochs as M_t learns to sit on the boundary -- that is
                # the mechanism that makes the attended region SUFFICIENT, so
                # watch ins_gain_over_random crossing 0 and faith corr rising.
                if _sbi_active:
                    print(f"[DIAG-P33] sbi=ON  lambda_localize={_lambda_localize}  "
                          f"lambda_sbi_cls={_lambda_sbi_cls}  stride={_sbi_stride}  "
                          f"blend=[{_sbi_lo},{_sbi_hi}]  sbi_n={_sbi_n}  "
                          f"L_sbi_cls={loss_sbi_cls.item():.4f}  "
                          f"L_localize={loss_localize.item():.4f}  "
                          f"(boundary-supervised attention; A/B/D regime untouched)")
                else:
                    print("[DIAG-P33] sbi=OFF (Phase 32 behaviour) -- insertion "
                          "stays holistic-limited (deletion necessary, not sufficient).")
                # Phase 34: spatial bottleneck diagnostic.  kept_mass = the soft
                # probability mass already sitting on the kept top-k cells; it
                # starts low (~k/N at uniform init) and should RISE toward 1.0 as
                # M_t concentrates real evidence into the cells the bottleneck
                # keeps -- that concentration is what makes the attended region
                # SUFFICIENT, so watch ins_gain_over_random crossing 0 and faith
                # corr rising while detection holds (funded by the P33 surplus).
                if _spatial_topk_active:
                    print(f"[DIAG-P34] spatial_topk=ON  frac={_spatial_topk}  "
                          f"keep_cells={_spatial_topk_k}/{model.N}  "
                          f"kept_mass={float(out_A.spatial_kept_mass):.4f}  "
                          f"(prediction funnelled through top-k cells; STE + convex renorm)")
                else:
                    print("[DIAG-P34] spatial_topk=OFF (frac=0) -- pooling over all "
                          "cells (Phase 33 behaviour); insertion stays holistic-limited.")
                if _dual_lens:
                    print(f"[DIAG-P35] dual_lens=ON  lens_gate={float(out_A.lens_gate):.4f}  "
                          f"suff_kept_mass={float(out_A.suff_kept_mass):.4f}  "
                          f"suff_keep_cells={_suff_k}/{model.N}  "
                          f"(M_nec=necessity/seam/D-pass; M_suff=sufficiency/B-pass/faith)")
                if _decomp_active and out_A.layer_weights is not None:
                    _lw0 = out_A.layer_weights.mean(dim=(0, 1)).detach().cpu().numpy()
                    print(f"[DIAG-P36] decomp=ON  mode={_decomp_mode}  layers={_decomp_L}  "
                          f"gate={float(out_A.decomp_gate):.4f}  "
                          f"overlap={float(out_A.decomp_overlap):.4f}  "
                          f"layer_w={np.round(_lw0, 3).tolist()}  "
                          f"(gate=decomp share vs single map; layer_w=per-layer evidence %)")

                # ── Phase 39/40: additive-head + concentration diagnostic ─────
                # top_cell_share = mean over batch of the single largest cell's
                # share of ReLU(M_t*e) mass.  Rising toward 1 = evidence is
                # concentrating (what insertion needs); ~1/N = still diffuse.
                if _aeh_on and (out_A.aeh_contrib is not None):
                    with torch.no_grad():
                        _c   = torch.relu(out_A.aeh_contrib)                 # (B,T,N)
                        _cn  = _c / _c.sum(dim=2, keepdim=True).clamp_min(1e-6)
                        _share = _cn.max(dim=2).values.mean().item()
                        # share of the top-k (~7/49) cells = what insertion needs
                        _km  = max(1, int(round(
                            float(getattr(config, "aeh_mass_topk_frac", 0.15))
                            * _cn.shape[2])))
                        _tkshare = _cn.topk(_km, dim=2).values.sum(dim=2).mean().item()
                    print(f"[DIAG-P39/40/41] aeh_gamma={float(out_A.aeh_gamma):.3f}  "
                          f"L_aeh_aux={loss_aeh_aux.item():.6f}  "
                          f"L_aeh_conc={loss_aeh_conc.item():.6f}  "
                          f"L_aeh_suff={loss_aeh_suff.item():.6f}  "
                          f"L_aeh_mass={loss_aeh_mass.item():.6f}  "
                          f"L_aeh_tconc={loss_aeh_tconc.item():.6f}  "
                          f"top_cell_share={_share:.3f}  "
                          f"top{_km}_share={_tkshare:.3f}")

            # ── Batch balance check ───────────────────────────────────────────
            if (batch_idx + 1) % 1000 == 0:
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
            _li  = loss_ins.item()                        # Phase 24
            _ld  = loss_del.item()                        # Phase 30
            _ltb = loss_temp_band.item()                  # Phase 30
            _lsb = loss_spatial_band.item()               # Phase 31
            _lts = loss_temp_sparse.item()                # Phase 25
            _lca = loss_cbm_aux.item()                    # Phase 26
            _lcd = loss_cbm_div.item()                    # Phase 26
            _lcm = loss_cbm_main_aux.item()               # Phase 27
            _ldm = loss_domain.item()                     # Phase 27
            _da  = (_dom_acc if not np.isnan(_dom_acc) else 0.0)  # Phase 27
            _ls  = loss_sparse.item()
            _lps = l_peak_spread.item()
            _lsh = loss_sharp.item()
            # Phase 29: coupling-health diagnostics (replaces the Phase 28
            # w-based eff_tokens, which measured a quantity that no longer
            # exists in pooled mode).
            #   eff_fr = 1/Herfindahl(M_frame)            — effective frames
            #            the prediction sees (max T; →1 = temporal one-hot,
            #            CBM slots starved of choice).
            #   eff_sp = mean 1/Herfindahl(M_t per frame) — effective spatial
            #            tokens each frame-vector pools (max N).
            #   s_top  = CBM slot-attention mass on the top-M_frame frame
            #            (pooled mode only).  Escape-hatch detector: if
            #            M_frame goes peaky while s_top stays ≈ 1/T, the
            #            slots' content term is overriding the log-prior and
            #            the temporal coupling is decoupling again.
            with torch.no_grad():
                _Bw, _Tw = out_A.M_frame.shape
                _eff_fr = float((1.0 / out_A.M_frame.pow(2).sum(dim=1)
                                 .clamp(min=1e-12)).mean().item())
                _mt_r = out_A.M_t.reshape(_Bw, _Tw, -1)
                _mt_r = _mt_r / _mt_r.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                _eff_sp = float((1.0 / _mt_r.pow(2).sum(dim=-1)
                                 .clamp(min=1e-12)).mean().item())
                if (out_A.slot_attn is not None
                        and out_A.slot_attn.shape[-1] == _Tw):
                    _topf = out_A.M_frame.argmax(dim=1)               # (B,)
                    _stop = float(out_A.slot_attn[
                        torch.arange(_Bw, device=_topf.device), :, _topf
                    ].mean().item())
                else:
                    _stop = 0.0

            run["total"]       += _lt;  run["cls"]    += _lc
            run["exp"]         += _le;  run["temp"]   += _lp
            run["cons"]        += _lco
            run["faith"]       += _lf;  run["ins"]    += _li     # Phase 24
            run["del"]         += _ld;  run["temp_band"] += _ltb # Phase 30
            run["spatial_band"] += _lsb                           # Phase 31
            run["temp_sparse"] += _lts                            # Phase 25
            run["cbm_aux"]     += _lca                            # Phase 26
            run["cbm_div"]     += _lcd                            # Phase 26
            run["cbm_main_aux"] += _lcm                           # Phase 27
            run["domain"]      += _ldm                            # Phase 27
            run["domain_acc"]  += _da                             # Phase 27
            run["eff_frames"]  += _eff_fr                         # Phase 29
            run["eff_spatial"] += _eff_sp                         # Phase 29
            run["slot_on_top"] += _stop                           # Phase 29
            run["sparse"]      += _ls
            run["peak_spread"] += _lps; run["sharp"]  += _lsh; run["n"] += 1
            # Phase 30: faith/ins only run on B-steps, del only on D-steps —
            # average each over its OWN pass count so the printed numbers stay
            # comparable with earlier phases.
            if _is_del_step:
                run["n_d"] += 1
            else:
                run["n_b"] += 1

            epoch_acc["total"]       += _lt;  epoch_acc["cls"]    += _lc
            epoch_acc["exp"]         += _le;  epoch_acc["temp"]   += _lp
            epoch_acc["faith"]       += _lf;  epoch_acc["ins"]    += _li   # Phase 24
            epoch_acc["del"]         += _ld                                 # Phase 30
            epoch_acc["temp_band"]   += _ltb                                # Phase 30
            epoch_acc["spatial_band"] += _lsb                               # Phase 31
            epoch_acc["temp_sparse"] += _lts                                # Phase 25
            epoch_acc["cbm_aux"]     += _lca                                # Phase 26
            epoch_acc["cbm_div"]     += _lcd                                # Phase 26
            epoch_acc["cbm_main_aux"] += _lcm                               # Phase 27
            epoch_acc["domain"]      += _ldm                                # Phase 27
            epoch_acc["domain_acc"]  += _da                                 # Phase 27
            epoch_acc["eff_frames"]  += _eff_fr                             # Phase 29
            epoch_acc["eff_spatial"] += _eff_sp                             # Phase 29
            epoch_acc["slot_on_top"] += _stop                               # Phase 29
            epoch_acc["kept_mass"]   += float(out_A.spatial_kept_mass)      # Phase 34
            epoch_acc["suff_kept"]   += float(out_A.suff_kept_mass)         # Phase 35
            epoch_acc["lens_gate"]   += float(out_A.lens_gate)              # Phase 35
            epoch_acc["decomp_div"]     += float(loss_decomp_div.item())    # Phase 36
            epoch_acc["decomp_overlap"] += float(out_A.decomp_overlap)      # Phase 36
            epoch_acc["decomp_gate"]    += float(out_A.decomp_gate)         # Phase 36
            if out_A.layer_weights is not None:                            # Phase 36
                epoch_acc["decomp_share"] += (
                    out_A.layer_weights.mean(dim=(0, 1)).detach().cpu().numpy())
            epoch_acc["sparse"]      += _ls
            epoch_acc["peak_spread"] += _lps; epoch_acc["sharp"]  += _lsh
            epoch_acc["n"]           += 1
            if _is_del_step:
                epoch_acc["n_d"] += 1
            else:
                epoch_acc["n_b"] += 1
            if _sbi_ran:                                                     # Phase 33
                epoch_acc["localize"] += loss_localize.item()
                epoch_acc["sbi_cls"]  += loss_sbi_cls.item()
                epoch_acc["n_sbi"]    += 1

            # ── Rolling log ───────────────────────────────────────────────────
            if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == total_batches:
                n = max(run["n"], 1)
                # Phase 30: faith/ins only run on B-steps, del on D-steps.
                n_b = max(run["n_b"], 1)
                n_d = max(run["n_d"], 1)
                _tau = out_A.early_attn_tau  # v4: actual M_t sharpening tau
                # Phase 25: also print refine_alpha so we can watch the
                # bi-directional gate open across batches/epochs.
                # Phase 26: also print cbm_blend so we can watch the CBM head
                # gain weight (or not) as training proceeds.
                # Phase 27: domain_acc tells us if DANN is engaging.
                _alpha_now = float(getattr(out_A, "refine_alpha", 0.0))
                _blend_now = float(getattr(out_A, "cbm_blend", 0.0))
                print(
                    f"[E{epoch:>{epoch_w}} {batch_idx+1:4d}/{total_batches}] "
                    f"total={run['total']/n:.4f}  cls={run['cls']/n:.4f}  "
                    f"exp={run['exp']/n:.4f}  temp={run['temp']/n:.4f}  "
                    f"faith={run['faith']/n_b:.4f}  ins={run['ins']/n_b:.4f}  "
                    f"del={run['del']/n_d:.4f}  "                          # Phase 30
                    f"tband={run['temp_band']/n:.4f}  "                    # Phase 30
                    f"sband={run['spatial_band']/n:.4f}  "                 # Phase 31
                    f"tsparse={run['temp_sparse']/n:.4f}  "                # Phase 25
                    f"cbm_aux={run['cbm_aux']/n:.4f}  "                    # Phase 26
                    f"cbm_div={run['cbm_div']/n:.4f}  "                    # Phase 26
                    f"cbm_main={run['cbm_main_aux']/n:.4f}  "              # Phase 27
                    f"domain={run['domain']/n:.4f}  "                      # Phase 27
                    f"dom_acc={run['domain_acc']/n:.3f}  "                 # Phase 27
                    f"eff_fr={run['eff_frames']/n:.1f}  "                  # Phase 29
                    f"eff_sp={run['eff_spatial']/n:.1f}  "                 # Phase 29
                    f"s_top={run['slot_on_top']/n:.3f}  "                  # Phase 29
                    f"sparse={run['sparse']/n:.4f}  "
                    f"sharp={run['sharp']/n:.4f}  "
                    f"peak_spread={run['peak_spread']/n:.4f}  "
                    f"cons={run['cons']/n:.4f}  "
                    f"alpha={_alpha_now:.3f}  "                            # Phase 25
                    f"blend={_blend_now:.3f}  "                            # Phase 26
                    f"tau={_tau:.3f}  sim={exp_out.inter_sample_sim:.2f}"
                )
                run = {
                    "total": 0.0, "cls": 0.0, "exp": 0.0, "temp": 0.0,
                    "cons": 0.0, "faith": 0.0, "ins": 0.0,
                    "del": 0.0, "temp_band": 0.0,                          # Phase 30
                    "spatial_band": 0.0,                                   # Phase 31
                    "temp_sparse": 0.0,                                    # Phase 25
                    "cbm_aux": 0.0, "cbm_div": 0.0,                        # Phase 26
                    "cbm_main_aux": 0.0,                                   # Phase 27
                    "domain": 0.0, "domain_acc": 0.0,                      # Phase 27
                    "eff_frames": 0.0, "eff_spatial": 0.0,                 # Phase 29
                    "slot_on_top": 0.0,                                    # Phase 29
                    "sparse": 0.0,
                    "peak_spread": 0.0, "sharp": 0.0, "n": 0,
                    "n_b": 0, "n_d": 0,                                    # Phase 30
                }

        scheduler.step()

        # ── Epoch-average train losses ─────────────────────────────────────────
        n = max(epoch_acc["n"], 1)
        # Phase 30: faith/ins are only computed on B-steps, del on D-steps —
        # average each over its own pass count.
        n_b = max(epoch_acc["n_b"], 1)
        n_d = max(epoch_acc["n_d"], 1)
        # Phase 33: report mean boundary-localization loss for the epoch.  It
        # should FALL across epochs (M_t learning to sit on the self-blend seam),
        # which is the mechanism behind insertion crossing random + faith rising.
        if _sbi_active:
            _nsbi = max(epoch_acc["n_sbi"], 1)
            print(f"[P33] epoch={epoch}  "
                  f"mean_L_localize={epoch_acc['localize'] / _nsbi:.4f}  "
                  f"mean_L_sbi_cls={epoch_acc['sbi_cls'] / _nsbi:.4f}  "
                  f"sbi_passes={epoch_acc['n_sbi']}")
        if _spatial_topk_active:
            print(f"[P34] epoch={epoch}  "
                  f"mean_kept_mass={epoch_acc['kept_mass'] / n:.4f}  "
                  f"keep_cells={_spatial_topk_k}/{model.N}  "
                  f"(rising mass = M_t concentrating into kept cells = locally sufficient)")
        if _dual_lens:
            print(f"[P35] epoch={epoch}  "
                  f"mean_lens_gate={epoch_acc['lens_gate'] / n:.4f}  "
                  f"mean_suff_kept_mass={epoch_acc['suff_kept'] / n:.4f}  "
                  f"suff_keep_cells={_suff_k}/{model.N}  "
                  f"(gate = sufficiency weight in the blend; suff_kept rising = M_suff concentrating)")
        if _decomp_active:
            _shares = epoch_acc["decomp_share"] / max(n, 1)             # (L,)
            _shares_str = "  ".join(
                f"L{i+1}={s * 100:.0f}%" for i, s in enumerate(_shares))
            print(f"[P36] epoch={epoch}  mode={_decomp_mode}  "
                  f"mean_gate={epoch_acc['decomp_gate'] / n:.4f}  "
                  f"mean_overlap={epoch_acc['decomp_overlap'] / n:.4f}  "
                  f"mean_div_loss={epoch_acc['decomp_div'] / n:.4f}  "
                  f"layer_shares: {_shares_str}  "
                  f"(gate=decomp vs single map; lower overlap=more complementary layers; "
                  f"shares=per-layer % of the evidence)")
        history["epoch"].append(epoch)
        history["train_total"].append(epoch_acc["total"]       / n)
        history["train_cls"].append(epoch_acc["cls"]           / n)
        history["train_exp"].append(epoch_acc["exp"]           / n)
        history["train_temp"].append(epoch_acc["temp"]         / n)
        history["train_faith"].append(epoch_acc["faith"]       / n_b)
        history["train_ins"].append(epoch_acc["ins"]           / n_b)   # Phase 24
        history["train_del"].append(epoch_acc["del"]           / n_d)   # Phase 30
        history["train_temp_band"].append(epoch_acc["temp_band"] / n)   # Phase 30
        history["train_spatial_band"].append(epoch_acc["spatial_band"] / n)  # Phase 31
        history["train_temp_sparse"].append(epoch_acc["temp_sparse"] / n)   # Phase 25
        history["train_cbm_aux"].append(epoch_acc["cbm_aux"]   / n)   # Phase 26
        history["train_cbm_div"].append(epoch_acc["cbm_div"]   / n)   # Phase 26
        history["train_cbm_main_aux"].append(epoch_acc["cbm_main_aux"] / n)  # Phase 27
        history["train_domain"].append(epoch_acc["domain"]     / n)   # Phase 27
        history["train_domain_acc"].append(epoch_acc["domain_acc"] / n)     # Phase 27
        history["train_eff_frames"].append(epoch_acc["eff_frames"] / n)     # Phase 29
        history["train_eff_spatial"].append(epoch_acc["eff_spatial"] / n)   # Phase 29
        history["train_slot_on_top"].append(epoch_acc["slot_on_top"] / n)   # Phase 29
        history["train_sparse"].append(epoch_acc["sparse"]     / n)
        history["train_peak_spread"].append(epoch_acc["peak_spread"] / n)
        history["train_sharp"].append(epoch_acc["sharp"] / n)

        # ── Validation (runs every _val_every epochs; default 1 = every epoch) ──
        if epoch % _val_every == 0:
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

            # ── Attention-diversity diagnostic (v4: full val-set) ─────────────
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

            # ── History ───────────────────────────────────────────────────────
            history["val_auc_roc"].append(float(metrics.get("auc_roc", float("nan"))))
            history["val_balanced_acc"].append(_val_bal_acc)
            history["val_real_acc"].append(_val_real_acc)
            history["val_fake_acc"].append(_val_fake_acc)
            history["val_inter_sample_cos"].append(diag_cosine)
            history["val_mt_std"].append(diag_std)
            history["val_peak_mode_share"].append(_peak_mode_share)

        # ── Clean-train sanity check (epochs 1, 2, then every _sanity_check_every) ─
        # Skipping saves ~1 full unaugmented forward pass per skipped epoch.
        _run_sanity = (epoch <= 2) or (epoch % _sanity_check_every == 0)
        if _run_sanity:
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
                "train_loss_del":       _avg("del"),          # Phase 30
                "train_loss_temp_band": _avg("temp_band"),    # Phase 30
                "train_loss_spatial_band": _avg("spatial_band"),  # Phase 31
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
        print("\n--- Pre-eval state cleanup ---")
        # Release training-only state so its CPU RAM + GPU VRAM is available
        # to the eval pipeline.  Without this, train_loader workers + optimizer
        # state + history + plot figures co-resided with the eval datasets
        # (HiDF + CelebDF + 5×FF++ + explanation suite) and OOM'd the process.
        try:
            del train_loader, val_loader
        except Exception:
            pass
        try:
            del optimizer, scheduler
        except Exception:
            pass
        try:
            del history
        except Exception:
            pass
        try:
            import matplotlib.pyplot as _plt_cleanup
            _plt_cleanup.close("all")
        except Exception:
            pass
        import gc as _gc_pre_eval
        _gc_pre_eval.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _cur = torch.cuda.memory_allocated(device) / 1e9
            print(f"[Pre-Eval] GPU allocated after cleanup: {_cur:.2f} GB")
        print("--- Starting evaluation ---")
        run_evaluation(config)


if __name__ == "__main__":
    import argparse as _ap_extra, sys as _sys_extra

    # ── New per-run speed knobs — not in EAHNConfig, parsed before main config ──
    _p_extra = _ap_extra.ArgumentParser(add_help=False)
    _p_extra.add_argument(
        "--sanity_check_every", type=int, default=5,
        help="Run clean-train sanity check every N epochs "
             "(always at epochs 1 and 2; default 5).")
    _p_extra.add_argument(
        "--val_every", type=int, default=1,
        help="Run validation every N epochs (default 1 = every epoch).")
    _ns_extra, _remaining = _p_extra.parse_known_args()

    # Strip new args from sys.argv so parse_args() doesn't choke on unknowns
    _sys_extra.argv = [_sys_extra.argv[0]] + _remaining

    args   = parse_args()
    config = EAHNConfig.from_args(args)

    # Inject fields that are not part of EAHNConfig dataclass
    config.sanity_check_every = _ns_extra.sanity_check_every
    config.val_every           = _ns_extra.val_every

    main(config)
