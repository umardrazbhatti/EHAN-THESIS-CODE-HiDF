"""
config.py — single source of truth for all EAHN hyperparameters.
CLI overrides via argparse; no hardcoded paths anywhere else.
"""

import argparse
import warnings
import torch
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EAHNConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: str = "/kaggle/input/"
    output_dir: str = "/kaggle/working/outputs/"
    cache_dir: str = "/kaggle/working/.face_cache/"
    resume_checkpoint: str = ""

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: Literal["synthetic", "ff++", "celeb_df", "dfdc", "hidf"] = "ff++"
    dataset_compression: str = "c23"
    num_frames: int = 16
    frame_size: int = 224
    train_split: float = 0.8
    val_split: float = 0.1

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = "efficientnet_b4"
    backbone_pretrained: bool = True
    transformer_layers: int = 4
    transformer_heads: int = 8
    d_model: int = 256
    dropout: float = 0.1

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda1: float = 0.02   # L_exp weight (reduced 0.3→0.1 phase20, 0.1→0.02 phase21: L_sparse takes over sparsity pressure)
    lambda2: float = 0.2   # L_temp weight (raised 0.1→0.2 phase6: loosen temporal grip)
    lambda_consistency: float = 0.3   # weight for consistency regularization loss (MSE between augmented and clean branch probs)
    alpha: float = 0.05    # entropy weight in weak supervision (phase20: alpha=0.3 was driving M_t to one-hot per frame; lowering frees M_t to form face-sized blobs)
    beta: float = 0.5      # TV weight in weak supervision
    gamma: float = 0.1     # gate decay rate in L_temp (was 10.0 — caused exp→0)
    attn_temp_init: float = 0.7    # start at τ=exp(0.7)≈2.0 (smoother softmax); log_temp remains learnable, this is initialization only (phase20)
    attn_diversity_weight: float = 5.0  # weight for JS diversity penalty in L_exp (raised 3.0→5.0 phase8: JS has smaller scale than cosine)
    cls_dropout_p: float = 0.0    # phase7: disabled — attn_pool now informative; joint gradient on every step
    label_smoothing: float = 0.05   # Task 3.2: maps 0→0.05, 1→0.95 to prevent logit saturation at 0.000/1.000
    max_per_class: int = 0         # if > 0, subsample train set to this many samples per class

    # ── Classification loss ───────────────────────────────────────────────────
    cls_loss_type: str = "focal"   # "bce" | "focal" — phase 19.8: activate focal to up-weight hard fakes
    focal_alpha: float = 0.75   # v4: raised 0.65→0.75 to penalise fake misses harder (fixes fake_acc collapse)
    focal_gamma: float = 2.5   # v4: raised 2.0→2.5 for stronger hard-example focus

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 4        # T4-safe with AMP+grad_ckpt: B*T=4*16=64 frames; grad_accum_steps=4 → effective 16
    grad_accum_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-2
    mixed_precision: bool = True   # kept for backward compat; use_amp is the authoritative flag
    num_workers: int = 0   # 0 = safe for Kaggle CUDA; increase locally if desired
    use_amp: bool = True           # FP16 automatic mixed precision (T4 supports FP16 not BF16)
    amp_dtype: str = "fp16"        # "fp16" | "bf16"
    grad_checkpoint: bool = True   # gradient checkpointing in TemporalStream to cut VRAM
    clip_grad_norm: float = 1.0    # max gradient norm for clipping

    # ── Evaluation / Visualisation ────────────────────────────────────────────
    eval_after_train: bool = True
    skip_eval: bool = False          # if True, suppress post-training evaluation entirely
    active_manipulation: str = ""           # REQUIRED at CLI; specialist-only mode
    celebdf_root: str = ""                  # path to Celeb-DF v2 dataset root
    celebdf_eval: bool = False              # run Celeb-DF test eval after FF++ test eval
    hidf_root: str = ""
    ffpp_cross_eval: bool = False
    ffpp_cross_root: str = ""
    hidf_split_seed: int = 42
    save_last_checkpoint: bool = False      # Phase 16 leftover; OFF by default
    explanation_suite: bool = True          # run new explanation metrics block after eval
    save_heatmaps: bool = True
    heatmap_samples: int = 20               # explanation-metric subset (faith corr, SSIM,
                                            # del/ins). Phase 30 runs pass 50 via CLI.
    xai_overlay_videos: int = 50            # Phase 30: XAI overlay PNG sample size
                                            # (was hardcoded 10 = 5 real + 5 fake).
                                            # Split per class 2:2:1 high:mid:low conf.
    random_test_n_samples: int = 30         # Task 1.7: n_random for model-randomization check (was 1)

    # ── Early stopping (Task 3.3) ─────────────────────────────────────────────
    early_stop_patience:  int   = 5                         # epochs without improvement before halt
    early_stop_metric:    str   = "val_balanced_accuracy"   # metric to monitor
    early_stop_min_delta: float = 0.001                     # minimum improvement to count
    no_early_stop:        bool  = False                     # v4: set True to disable ES entirely (run full epochs)

    # ── Phase 21: faithful attention bottleneck ───────────────────────────────
    phase21_enabled:      bool  = True    # master switch; False reverts to Phase 20 behaviour
    lambda_faith:         float = 0.3     # weight for faithfulness KL loss
    lambda_ins:           float = 0.5     # Phase 24: weight for insertion-AUC training loss (focal BCE on bottlenecked logits) — re-uses existing out_B forward; default 0.5 makes faithfulness ~35% of gradient magnitude vs cls 73% (was ~0.5% with KL alone)
    lambda_sparse:        float = 0.05    # weight for sparsity (negative peak) loss
    faith_warmup_epochs:  int   = 3       # linear ramp from 0 → lambda_faith over N epochs
    attn_floor:           float = 0.0     # Phase 23: gate floor reduced 0.05→0.0 to make M_t a true spatial bottleneck (was leaking 4.8% mass to every position regardless of M_t value)
    blur_kernel:          int   = 21      # Gaussian kernel size for bottlenecked input
    lambda_peak_spread:   float = 0.5     # v4: raised 0.3→0.5; weight for HardAttentionDiversityLoss
    lambda_sharp:         float = 0.15     # v4: raised 0.5→1.0; weight for sharpness loss on logits
    disk_guard_gb:        float = 3.0     # v4: min free GB before face-cache write is skipped
    blur_sigma:           float = 10.0    # Gaussian sigma for bottlenecked input
    bottleneck_peak_floor: float = 0.25   # Phase 22: fixed floor for bottleneck mask normalisation;
                                          # diffuse maps (peak < floor) get heavily blurred so L_faith bites
    bottleneck_hard_topk_frac: float = 0.20  # Phase 25: when > 0, build_bottlenecked_input uses HARD
                                             # top-K binary mask (straight-through estimator) instead of
                                             # soft blur.  K = frac * H * W (typical 0.20 → keep 20% of pixels).
                                             # Aligns loss_ins / loss_faith with the insertion AUC metric.
                                             # Set 0.0 to revert to Phase 22/24 soft blending.
    # ── Phase 32: B-pass sufficiency hard-mask (insertion alignment) ──────────
    # P31 verdict (run 6-13-26 0900hrs): detection FIXED (AUC 0.9115) and
    # necessity FIXED (deletion 0.224, del_gain +0.285 on EVERY sample), but the
    # insertion ordering still LOSES to random (ins_gain -0.216) and faith corr
    # stalled at 0.318.  Root cause: loss_ins trained on the SOFT bottleneck
    # (peak_floor path keeps the whole region at low contrast) while the
    # insertion metric does a HARD top-k pixel reveal on a blur canvas — the
    # model was never trained on what it is tested on.  These fields turn ON a
    # hard top-K binary keep-mask (straight-through estimator) for the B-pass
    # ONLY; the D-pass keeps using bottleneck_hard_topk_frac (0.0 = soft) so the
    # already-met deletion number is structurally untouched.  Each B-step samples
    # frac ~ U[lo, hi] so loss_ins trains the steep low-reveal span of the
    # insertion curve (the 10%/25% checkpoints) rather than a single point.
    # lo=hi=0.0 = P31 soft behaviour (full back-compat).
    ins_hard_topk_frac_lo: float = 0.0   # lower edge of the per-B-step keep fraction
    ins_hard_topk_frac_hi: float = 0.0   # upper edge; 0.0 = OFF (soft B-pass, P31)
    # ── Phase 33: self-blended images (SBI) + boundary-supervised attention ──
    # Root-cause fix for the insertion wall (proven runs 6-13/6-14): the
    # detector's fake-evidence is HOLISTIC -- deleting the attended region
    # crashes fake-conf (necessity) but revealing it on a blur canvas does NOT
    # restore conf (sufficiency), so insertion structurally loses to random for
    # any compact map.  Each optimizer step a small batch of REAL clips is
    # self-blended on-GPU (data.HiDF_self_blend) into pseudo-fakes with a KNOWN
    # blend boundary; the model is trained to classify them fake (lambda_sbi_cls)
    # AND to align its attention M_t to the boundary (lambda_localize).  The
    # boundary is a LOCAL sufficient artifact, so insertion in attention order
    # recovers conf fast; self-blends are also the SOTA cross-dataset cue.  The
    # A/B/D detection regime is UNTOUCHED (bounded additive aux pass once per
    # optimizer step), so the met detection/deletion numbers are protected.
    # sbi_enabled=False = exact Phase 32 behaviour (full back-compat).
    sbi_enabled:       bool  = False   # master switch (Phase 33)
    lambda_localize:   float = 0.0     # weight on the boundary-attention loss (SWEEP axis)
    lambda_sbi_cls:    float = 0.5     # weight on the SBI fake-classification aux loss
    sbi_blend_lo:      float = 0.25    # min ellipse semi-axis (fraction of half-extent)
    sbi_blend_hi:      float = 0.55    # max ellipse semi-axis
    sbi_stride:        int   = 8       # run the SBI aux pass every N micro-batches
                                       # (8 = once per optimizer step at grad_accum=8)
    # ── Phase 34: hard spatial top-k attention bottleneck (insertion fix) ─────
    # P33 verdict (run 6-15-26): SBI fake-classification gave best-ever detection
    # (0.971) but seam-localization did NOT make HiDF insertion beat random
    # (ins_gain still negative, WORSE as lambda_localize rose) — the SBI seam
    # location != the holistic HiDF evidence location.  The map is near-uniform
    # (m_t_std 0.03) and the model average-pools, so NO compact map can be
    # "sufficient" (insertion's exact test) by tuning.  Architectural fix: at the
    # M_t spatial-pool step keep ONLY the top `spatial_topk_frac` of the 49 cells
    # (straight-through estimator + convex renormalisation -> magnitude preserved,
    # the Phase 28 anti-starvation lesson).  The prediction is FORCED through a
    # compact region, so the model concentrates real evidence into the kept cells
    # -> revealing them (insertion) beats random and faithfulness rises.  Lives in
    # EAHN.forward, so it applies identically at train AND eval.  Funded by the
    # P33 detection surplus (0.971 vs 0.92 target).  0.0 or >=1.0 = OFF (exact
    # Phase 33, full back-compat).  This is the single Phase-34 sweep axis.
    spatial_topk_frac: float = 0.0     # keep fraction of the 49 spatial cells (SWEEP axis)
    bidirectional_enabled:  bool  = True  # Phase 25: re-wire CrossAttentionFusion as the refined M_t
                                          # path.  M_t_used = α * M_t_refined + (1-α) * M_t_early, with
                                          # α = sigmoid(refine_gate).  Phase 26: refine_gate init
                                          # raised from -2.0 to -0.5 (sigmoid -2.0 = 0.119 vs -0.5
                                          # = 0.378) so the bidirectional path actually engages from
                                          # epoch 1 — in Phase 25 alpha only crept 0.118 → 0.121
                                          # across all 8 epochs and the gate never opened.
    refine_gate_init:       float = -0.5   # Phase 26: initial value of refine_gate parameter; sigmoid
                                           # gives alpha. -0.5 → alpha=0.378 (engaged from start).
    lambda_temp_sparse:   float = 0.02    # Phase 25: weight for temporal_sparsity_loss on M_frame.
                                          # Phase 26: gentled from 0.05 → 0.02 (was driving M_frame
                                          # to one-hot on frame 0).  Still shares faith_warmup_epochs.
    # ── Phase 26: Concept Slot Bottleneck (CBM) ──────────────────────────────
    # Phase 27 NOTE: cbm_serial=True changes the wiring -- cbm_logit becomes
    # the SOLE prediction (no main/CBM blend), main_logit is supervised only
    # as a regulariser via lambda_cbm_main_aux.  K bumped 8 -> 12 because the
    # serial bottleneck must carry the full prediction.
    cbm_enabled:           bool  = True
    cbm_serial:            bool  = True   # Phase 27: serial vs Phase 26 parallel
    cbm_coupled:           bool  = True   # Phase 28 (DEAD — kept for ablation
                                          # history only): scale CBM input tokens
                                          # by w = M_t ⊙ M_frame (max-normalised).
                                          # Run 6-11-26 1300hrs: multiplicative
                                          # scaling starved slot attention (782 of
                                          # 784 keys ≈ 0 → softmax dilution →
                                          # constant logit), cls froze at 0.1838
                                          # from batch 1000 and val AUC never left
                                          # ~0.5 for 9 epochs.  Superseded by
                                          # cbm_pooled, which takes precedence.
    cbm_pooled:            bool  = True   # Phase 29: CBM reads the M_t-pooled
                                          # per-frame vectors attn_pool_per_frame
                                          # (B, T, d) — a CONVEX combination, so
                                          # magnitude is always preserved and M_t
                                          # is structurally load-bearing (tokens
                                          # it suppresses are absent, not small).
                                          # M_frame couples as an attention PRIOR:
                                          # log(M_frame) added to slot-attention
                                          # logits (renormalised by softmax — no
                                          # starvation possible).  P23 precedent:
                                          # the same pooled path classified at
                                          # 0.904 with k1 1.64 under the old
                                          # artifact-diluted protocol.
    cbm_num_slots:         int   = 12     # Phase 27: K = 12 (was 8 in Phase 26)
    lambda_cbm_aux:        float = 0.10   # weight on CBM auxiliary classification loss
    lambda_cbm_div:        float = 0.05   # weight on slot diversity loss.  Phase 30:
                                          # run with 0.15 — at 0.05 (and slot_q init
                                          # 0.02) cbm_div stayed pinned 0.98–1.0 for
                                          # 9 epochs (run 6-12-26 0020hrs): all 12
                                          # slots identical, effective K=1.
    # ── Phase 30: necessity (deletion) pass + bounded temporal band ──────────
    lambda_del:            float = 0.2    # weight for the deletion (necessity) loss
                                          # on the INVERSE-bottlenecked input (top-M_t
                                          # region blurred, rest visible).  Phase 31
                                          # form: reals → BCE-to-REAL (true label);
                                          # fakes → hinge relu(logit−del_margin_logit)
                                          # gated on A-pass detection; plus a full-blur
                                          # ANCHOR step every del_anchor_every batches.
                                          # P30 ran 0.3 with BCE-to-0 on ALL samples —
                                          # certainty-REAL on 96%-visible fakes — and
                                          # collapsed fake_acc (run 6-12-26 1300hrs).
                                          # NOTE the hinge is in LOGIT units (~1–3
                                          # early), so the logged `del=` value is much
                                          # larger than P30's focal units; per-sample
                                          # GRADIENT (lambda×1, vanishing at p≤0.5) is
                                          # comparable.  Alternates with the B-pass
                                          # (even/odd steps) so the per-step forward
                                          # count stays at 2.  Shares
                                          # faith_warmup_epochs.  0.0 = off.
    lambda_temp_band:      float = 0.05   # weight for temporal_band_loss: hinge
                                          # penalty when eff_fr = 1/Σp² exceeds
                                          # temp_band_target.  ZERO gradient below
                                          # the target — cannot ratchet to one-hot
                                          # (the P27 lambda_temp_sparse failure).
                                          # 0.0 = off.
    temp_band_target:      float = 6.0    # eff_fr the band allows before penalising
                                          # (of T=16). 6 effective frames keeps the
                                          # top frame at ~17-25% mass — enough for
                                          # measurable k1 drops without one-hot.
    # ── Phase 31: D-pass detox + spatial band ─────────────────────────────────
    # Run 6-12-26 1300hrs verdict: every P30 mechanism fired (cbm_div 0.94→0.10,
    # eff_fr 4.3–4.6, del falling) but detection fell 0.879→0.796 and fake_acc
    # collapsed (train_clean 0.40, val E4 0.082 at warmup completion).  Root
    # cause: loss_del demanded CERTAIN-REAL on fakes whose erased footprint was
    # ~2/49 cells (96% of the face still visible) for HALF of all steps —
    # systematic "visible fake evidence → REAL" supervision.  Secondary: the
    # open-ended -peak sparsity reward squeezed eff_sp 3.4 (P29) → 1.8 (P30).
    del_margin_logit:      float = 0.0    # Phase 31: hinge margin for the fake-side
                                          # D-loss, relu(logit_D − margin).  0.0 =
                                          # push p(fake|erased) to ≤0.5 then STOP.
                                          # The old BCE-to-0 kept pushing toward
                                          # certainty-REAL, which poisons detection.
    del_anchor_every:      int   = 8      # Phase 31: every Nth batch the D-pass uses
                                          # a FULL-blur input (no M_t dependence)
                                          # with target REAL for all samples — no
                                          # evidence visible ⇒ REAL is epistemically
                                          # correct for both classes.  Anchors the
                                          # blur end of the del/ins eval curves
                                          # (P30: blurred_conf 0.40 put a floor
                                          # under deletion AUC).  0 = off.
    lambda_spatial_band:   float = 0.05   # Phase 31: two-sided hinge on per-frame
                                          # eff_sp = 1/Σp² over the 49 cells.
                                          # Replaces lambda_sparse (-peak reward,
                                          # an open-ended ratchet — same pathology
                                          # as P27's temporal -max).  0.0 = off.
    spatial_band_lo:       float = 4.0    # lower edge: penalise eff_sp < 4 cells.
                                          # P30 collapsed to 1.8 cells → detection
                                          # starved + insertion ordering carried
                                          # ~2 cells of signal.  P29 ran at ~3.4.
    spatial_band_hi:       float = 10.0   # upper edge: penalise eff_sp > 10 cells
                                          # (keeps the map from re-diffusing to the
                                          # 17-cell init state; B-pass needs a
                                          # concentrated keep-region to work).
    focal_alpha_pos:       float = -1.0   # Phase 31: class-conditional focal alpha
                                          # for FAKE (label 1) samples.  Negative =
                                          # disabled → falls back to the legacy
                                          # global focal_alpha scale.  NOTE: the
                                          # legacy focal_alpha multiplies BOTH
                                          # classes equally (pure scale, no class
                                          # weighting) — fake recall never had a
                                          # loss-side counterweight.  Run with 1.0.
    focal_alpha_neg:       float = -1.0   # Phase 31: alpha for REAL (label 0).
                                          # Run with 0.5 → 2:1 fake:real error
                                          # weighting at the same mean scale as the
                                          # legacy 0.75 global multiplier.
    lambda_cbm_main_aux:   float = 0.05   # Phase 27: aux supervision on main_logit
                                          # (NOT in prediction path; just a regulariser
                                          # so we can diagnose whether attn_pool alone
                                          # could still classify).
    # ── Phase 26: Class-balanced sampler ─────────────────────────────────────
    # WeightedRandomSampler with weight = 1/n_class per sample. Combats the
    # majority-class collapse seen in Phase 25 (real_acc 0.142 = model
    # defaulted to always-fake). Default ON for HiDF (real:fake ≈ 1.10:1).
    class_balanced_sampler: bool = True
    # ── Phase 27: DANN (Domain Adversarial Neural Network) ────────────────────
    # GRL(attn_pool) -> DomainHead(d, num_domains) with per-sample synthetic
    # domain labels assigned in data/HiDF_datasets.py (random in [0, D-1])
    # via 4 augmentation pipelines defined in data/HiDF_transforms.py:
    #     0 = clean, 1 = heavy JPEG, 2 = noise, 3 = blur
    # Training adds:
    #     loss_domain = CE(domain_logits, domain_labels)
    # weighted by lam_domain_eff (linear warmup 0 -> lambda_domain over
    # domain_warmup_epochs).  The GRL passes -lambda_grl_eff * grad upstream,
    # so attn_pool is pushed toward domain-invariance.  Both warmups share
    # the same epoch count by default.
    dann_enabled:          bool  = True
    num_domains:           int   = 4
    lambda_domain:         float = 0.10   # weight on domain CE loss
    lambda_grl:            float = 1.0    # GRL scale (max after warmup)
    domain_warmup_epochs:  int   = 3      # linear ramp for both lambda_domain and lambda_grl
    snapshot_every:       int   = 2       # save snapshot every N epochs

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                warnings.warn("No GPU found. Switching to CPU with reduced settings.")
                self._apply_cpu_safe_overrides()

    def _apply_cpu_safe_overrides(self):
        self.num_frames = 4
        self.transformer_layers = 2
        self.transformer_heads = 2
        self.batch_size = 2
        self.mixed_precision = False
        self.use_amp = False
        self.grad_checkpoint = False
        self.num_workers = 0
        if "efficientnet_b4" in self.backbone:
            self.backbone = "efficientnet_b0"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EAHNConfig":
        cfg = cls()
        for key, val in vars(args).items():
            if hasattr(cfg, key) and val is not None:
                setattr(cfg, key, val)
        if cfg.dataset_name == "ff++" and not cfg.active_manipulation:
            raise ValueError(
                "--active_manipulation is required when --dataset_name ff++. "
                "Choose one of: Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures."
            )
        return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EAHN Training and Evaluation")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None,
                        choices=["synthetic", "ff++", "celeb_df", "dfdc", "hidf"])
    parser.add_argument("--dataset_compression", type=str, default=None,
                        help="FF++ compression level, e.g. c23 (default) or c40")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader worker processes. Use 0 on Kaggle to avoid fork errors.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=None,
                        help="Weight for consistency regularization loss (default 0.3). "
                             "MSE between augmented-branch and clean-branch probs.")
    parser.add_argument("--heatmap_samples", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--eval_after_train", action="store_true", default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--attn_temp_init", type=float, default=None)
    parser.add_argument("--attn_diversity_weight", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--cls_dropout_p", type=float, default=None)
    parser.add_argument("--cls_loss_type", type=str, default=None,
                        choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--focal_alpha_pos", type=float, default=None,
                        help="Phase 31: class-conditional focal alpha for FAKE "
                             "(label 1). Set together with --focal_alpha_neg; "
                             "negative/unset = legacy global focal_alpha scale. "
                             "Run with 1.0 (fake) / 0.5 (real) — 2:1 fake-error "
                             "weighting at the same mean scale as the legacy "
                             "0.75 global multiplier (which weighted BOTH "
                             "classes equally, i.e. no class counterweight).")
    parser.add_argument("--focal_alpha_neg", type=float, default=None,
                        help="Phase 31: class-conditional focal alpha for REAL "
                             "(label 0). See --focal_alpha_pos.")
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=None)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--amp_dtype", type=str, default=None, choices=["fp16", "bf16"])
    parser.add_argument("--grad_checkpoint", dest="grad_checkpoint", action="store_true", default=None)
    parser.add_argument("--no_grad_checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help="Label smoothing applied to BCE/focal loss target (0.05 = maps 0->0.05, 1->0.95)")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="If > 0, subsample train set to this many samples per class (balanced 1k/1k)")
    parser.add_argument("--skip_eval", action="store_true", default=False,
                        help="If set, skip post-training evaluation (useful for mid-run Kaggle sessions)")
    parser.add_argument("--active_manipulation", type=str, default=None,
                        choices=["Deepfakes", "Face2Face", "FaceShifter",
                                 "FaceSwap", "NeuralTextures"],
                        help="Required: specialist manipulation type to train on.")
    parser.add_argument("--celebdf_root", type=str, default=None,
                        help="Path to Celeb-DF v2 dataset root.")
    parser.add_argument("--celebdf_eval", action="store_true", default=None,
                        help="Run Celeb-DF v2 test evaluation after FF++ test eval.")
    parser.add_argument("--hidf_root", type=str, default=None,
                        help="HiDF dataset root (contains Real-vid/ and Fake-vid/)")
    parser.add_argument("--ffpp_cross_eval", action="store_true", default=None,
                        help="Run FF++ per-manipulation cross-evaluation after training")
    parser.add_argument("--ffpp_cross_root", type=str, default=None,
                        help="FF++ ffpp_data/ root for cross-evaluation")
    parser.add_argument("--hidf_split_seed", type=int, default=None,
                        help="Seed for HiDF source-grouped train/val/test split")
    parser.add_argument("--save_last_checkpoint", action="store_true", default=None,
                        help="Save last_checkpoint.pth after every epoch (for multi-session resume).")
    parser.add_argument("--explanation_suite", dest="explanation_suite",
                        action="store_true", default=None,
                        help="Run explanation metrics suite after evaluation.")
    parser.add_argument("--no_explanation_suite", dest="explanation_suite",
                        action="store_false")
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Epochs without improvement before early stopping (default 5).")
    parser.add_argument("--early_stop_metric", type=str, default=None,
                        help="Metric to monitor for early stopping (default val_balanced_accuracy).")
    parser.add_argument("--early_stop_min_delta", type=float, default=None,
                        help="Minimum improvement to count for early stopping (default 0.001).")
    parser.add_argument("--save_heatmaps", dest="save_heatmaps",
                        action="store_true", default=None,
                        help="Save heatmap PNGs and MP4 overlays after evaluation.")
    parser.add_argument("--no_save_heatmaps", dest="save_heatmaps",
                        action="store_false")
    parser.add_argument("--phase21_enabled", dest="phase21_enabled",
                        action="store_true", default=None,
                        help="Enable Phase 21 faithful attention bottleneck (default True).")
    parser.add_argument("--no_phase21_enabled", dest="phase21_enabled",
                        action="store_false")
    parser.add_argument("--lambda_faith", type=float, default=None,
                        help="Weight for Phase 21 faithfulness KL loss (default 0.3).")
    parser.add_argument("--lambda_ins", type=float, default=None,
                        help="Phase 24: weight for insertion-AUC training loss "
                             "(focal BCE on bottlenecked logits — re-uses out_B). "
                             "Default 0.5; shares faith_warmup_epochs ramp.")
    parser.add_argument("--lambda_sparse", type=float, default=None,
                        help="Weight for Phase 21 sparsity (negative peak) loss (default 0.05). "
                             "Phase 31: superseded by --lambda_spatial_band (bounded two-sided "
                             "hinge); the -peak form is an open-ended ratchet that squeezed "
                             "eff_sp to 1.8/49 cells in P30. Keep this at 0.0.")
    parser.add_argument("--faith_warmup_epochs", type=int, default=None,
                        help="Epochs to linearly ramp lambda_faith from 0 (default 3).")
    parser.add_argument("--attn_floor", type=float, default=None,
                        help="Gate floor for EarlyAttnHead (default 0.05).")
    parser.add_argument("--blur_kernel", type=int, default=None,
                        help="Gaussian kernel size for bottlenecked input (default 21).")
    parser.add_argument("--blur_sigma", type=float, default=None,
                        help="Gaussian sigma for bottlenecked input (default 10.0).")
    parser.add_argument("--bottleneck_peak_floor", type=float, default=None,
                        help="Phase 22: fixed absolute floor for bottleneck mask normalisation "
                             "(default 0.25). Diffuse maps whose peak < floor get blurred harder, "
                             "so the faithfulness loss bites even before sharpening converges.")
    parser.add_argument("--bottleneck_hard_topk_frac", type=float, default=None,
                        help="Phase 25: when > 0 (typical 0.20), bottleneck uses a HARD top-K binary "
                             "mask via straight-through estimator instead of the soft blur. Aligns "
                             "loss_ins / loss_faith training signal with the insertion-AUC metric. "
                             "Set 0.0 to revert to Phase 22/24 soft behaviour. "
                             "Phase 32: this controls the D-pass only; use "
                             "--ins_hard_topk_frac_lo/hi for the B-pass.")
    parser.add_argument("--ins_hard_topk_frac_lo", type=float, default=None,
                        help="Phase 32: lower edge of the B-pass sufficiency hard "
                             "top-K keep fraction (sampled U[lo,hi] per B-step). "
                             "Trains loss_ins/loss_faith on the insertion metric's "
                             "HARD pixel reveal. lo=hi=0.0 = soft B-pass (P31).")
    parser.add_argument("--ins_hard_topk_frac_hi", type=float, default=None,
                        help="Phase 32: upper edge of the B-pass sufficiency hard "
                             "top-K keep fraction. Typical 0.25. The D-pass "
                             "(deletion) stays soft and is unaffected.")
    # ── Phase 33: SBI self-blend + boundary-supervised attention ──────────────
    parser.add_argument("--sbi_enabled", dest="sbi_enabled",
                        action="store_true", default=None,
                        help="Phase 33: enable online self-blended pseudo-fakes with "
                             "blend-boundary attention supervision (local-evidence fix "
                             "for the insertion wall). Bounded additive aux pass; the "
                             "A/B/D detection regime is untouched. OFF = Phase 32.")
    parser.add_argument("--lambda_localize", type=float, default=None,
                        help="Phase 33: weight on the boundary-attention loss (pulls "
                             "M_t onto the self-blend seam). This is the sweep axis. "
                             "Typical 0.5-2.0. Requires --sbi_enabled.")
    parser.add_argument("--lambda_sbi_cls", type=float, default=None,
                        help="Phase 33: weight on the SBI fake-classification aux loss "
                             "(teaches the blend cue; helps cross-dataset). Default 0.5.")
    parser.add_argument("--sbi_blend_lo", type=float, default=None,
                        help="Phase 33: min self-blend ellipse semi-axis (fraction of "
                             "half-extent). Default 0.25.")
    parser.add_argument("--sbi_blend_hi", type=float, default=None,
                        help="Phase 33: max self-blend ellipse semi-axis. Default 0.55.")
    parser.add_argument("--sbi_stride", type=int, default=None,
                        help="Phase 33: run the SBI aux pass every N micro-batches "
                             "(8 = once per optimizer step at grad_accum=8). Lower = "
                             "stronger localization signal but more compute.")
    # ── Phase 34: hard spatial top-k attention bottleneck ─────────────────────
    parser.add_argument("--spatial_topk_frac", type=float, default=None,
                        help="Phase 34: hard spatial attention bottleneck. At the "
                             "M_t pooling step, keep ONLY the top fraction of the 49 "
                             "spatial cells (straight-through estimator + convex "
                             "renormalisation) so the prediction is forced through a "
                             "compact region and insertion can beat random. Typical "
                             "0.25-0.50. 0.0 or >=1.0 = OFF (Phase 33 behaviour). "
                             "Applies at train AND eval. Single Phase-34 sweep axis.")
    parser.add_argument("--bidirectional_enabled", dest="bidirectional_enabled",
                        action="store_true", default=None,
                        help="Phase 25: enable bi-directional refinement — CrossAttentionFusion "
                             "produces a refined M_t from transformer-context Q, blended with the "
                             "early M_t via learnable sigmoid gate.")
    parser.add_argument("--no_bidirectional", dest="bidirectional_enabled",
                        action="store_false",
                        help="Phase 25: disable bi-directional refinement (use only early M_t).")
    parser.add_argument("--lambda_temp_sparse", type=float, default=None,
                        help="Phase 25: weight for temporal_sparsity_loss on M_frame "
                             "(default 0.02). Pushes temporal_gate toward peaky M_frame so "
                             "k1/k2/k4 frame-drop ratios exceed 1.0x. Phase 30: superseded "
                             "by --lambda_temp_band (bounded); keep this at 0.0.")
    # ── Phase 30: necessity pass + bounded temporal band + overlay count ──
    parser.add_argument("--lambda_del", type=float, default=None,
                        help="Phase 30/31: weight for the deletion (necessity) loss on "
                             "the inverse-bottlenecked input. Phase 31 form: reals -> "
                             "BCE-to-REAL (true label); fakes -> hinge "
                             "relu(logit - del_margin_logit) gated on A-pass detection; "
                             "full-blur anchor every del_anchor_every batches. Alternates "
                             "with the B-pass per step; shares faith_warmup_epochs. "
                             "Default 0.2; 0.0 disables (ablation).")
    parser.add_argument("--lambda_temp_band", type=float, default=None,
                        help="Phase 30: weight for temporal_band_loss — hinge penalty "
                             "when eff_fr=1/sum(p^2) exceeds temp_band_target. Zero "
                             "gradient below target (no one-hot ratchet). Default 0.05; "
                             "0.0 disables (ablation).")
    parser.add_argument("--temp_band_target", type=float, default=None,
                        help="Phase 30: effective-frame count the temporal band allows "
                             "before penalising (default 6.0 of T=16).")
    parser.add_argument("--xai_overlay_videos", type=int, default=None,
                        help="Phase 30: number of XAI overlay videos saved at eval "
                             "(default 50 = 25 real + 25 fake; was 10).")
    # ── Phase 31: D-pass detox + spatial band ─────────────────────────────
    parser.add_argument("--del_margin_logit", type=float, default=None,
                        help="Phase 31: hinge margin for the fake-side D-loss "
                             "relu(logit_D - margin), applied only to fakes the "
                             "A-pass currently detects. 0.0 = push p(fake|erased) "
                             "to 0.5 then stop. (P30's BCE-to-0 on ALL samples "
                             "pushed toward certainty-REAL on 96%-visible fakes "
                             "and collapsed fake_acc to 0.40 train / 0.57 test.)")
    parser.add_argument("--del_anchor_every", type=int, default=None,
                        help="Phase 31: every Nth batch the D-pass uses a FULL-blur "
                             "input with target REAL for all samples (no evidence "
                             "visible => REAL). Anchors the blur end of the del/ins "
                             "curves. Default 8; 0 disables.")
    parser.add_argument("--lambda_spatial_band", type=float, default=None,
                        help="Phase 31: weight for spatial_band_loss — two-sided "
                             "hinge on per-frame eff_sp=1/sum(p^2) over 49 cells. "
                             "Replaces --lambda_sparse (open-ended -peak ratchet "
                             "that squeezed eff_sp to 1.8 cells in P30). Default "
                             "0.05; 0.0 disables.")
    parser.add_argument("--spatial_band_lo", type=float, default=None,
                        help="Phase 31: lower band edge — penalise eff_sp below "
                             "this many effective cells (default 4.0).")
    parser.add_argument("--spatial_band_hi", type=float, default=None,
                        help="Phase 31: upper band edge — penalise eff_sp above "
                             "this many effective cells (default 10.0).")
    parser.add_argument("--refine_gate_init", type=float, default=None,
                        help="Phase 26: initial value of bidirectional refine_gate parameter. "
                             "alpha = sigmoid(refine_gate_init). Default -0.5 → alpha=0.378 so "
                             "the bidirectional path engages from epoch 1.")
    # ── Phase 26: Concept Slot Bottleneck ────────────────────────────────
    parser.add_argument("--cbm_enabled", dest="cbm_enabled",
                        action="store_true", default=None,
                        help="Phase 26: enable Concept Slot Bottleneck head (K=8 learned slot "
                             "queries → concept scores → parallel classifier blended with main).")
    parser.add_argument("--no_cbm", dest="cbm_enabled", action="store_false",
                        help="Phase 26: disable CBM head; revert to Phase 25 architecture.")
    parser.add_argument("--cbm_num_slots", type=int, default=None,
                        help="Phase 26: number of CBM concept slots K (default 8).")
    parser.add_argument("--lambda_cbm_aux", type=float, default=None,
                        help="Phase 26: weight on CBM auxiliary classification loss (default 0.10).")
    parser.add_argument("--lambda_cbm_div", type=float, default=None,
                        help="Phase 26: weight on slot diversity loss (default 0.05).")
    # ── Phase 26: Class-balanced sampler ─────────────────────────────────
    parser.add_argument("--class_balanced_sampler", dest="class_balanced_sampler",
                        action="store_true", default=None,
                        help="Phase 26: use WeightedRandomSampler with inverse-class-frequency "
                             "weights to combat majority-class collapse (default True).")
    parser.add_argument("--no_class_balanced_sampler", dest="class_balanced_sampler",
                        action="store_false",
                        help="Phase 26: disable class-balanced sampler; use plain shuffle.")
    # ── Phase 27: serial CBM toggle + aux supervision weight ────────────
    parser.add_argument("--cbm_serial", dest="cbm_serial",
                        action="store_true", default=None,
                        help="Phase 27: serial CBM mode -- cbm_logit is the SOLE "
                             "prediction (no main/CBM blend). main_logit is "
                             "supervised separately as a regulariser.")
    parser.add_argument("--no_cbm_serial", dest="cbm_serial",
                        action="store_false",
                        help="Phase 27: disable serial mode (Phase 26 parallel blend).")
    parser.add_argument("--lambda_cbm_main_aux", type=float, default=None,
                        help="Phase 27: weight on the aux supervision of main_logit "
                             "in serial mode (default 0.05). Pure diagnostic — does "
                             "NOT participate in prediction.")
    # ── Phase 28: CBM-attention coupling toggle ─────────────────────────
    parser.add_argument("--cbm_coupled", dest="cbm_coupled",
                        action="store_true", default=None,
                        help="Phase 28: scale CBM input tokens by w = M_t * M_frame "
                             "(max-normalised) so the explanation maps are "
                             "load-bearing for the prediction (default True).")
    parser.add_argument("--no_cbm_coupled", dest="cbm_coupled",
                        action="store_false",
                        help="Phase 28: disable coupling (Phase 27 raw-Q CBM input; "
                             "ablation switch).")
    # ── Phase 29: pooled-frame CBM input (supersedes cbm_coupled) ───────
    parser.add_argument("--cbm_pooled", dest="cbm_pooled",
                        action="store_true", default=None,
                        help="Phase 29: CBM reads M_t-pooled per-frame vectors "
                             "(B, T, d) with log(M_frame) as a slot-attention "
                             "prior. Convex pooling — no magnitude starvation "
                             "(the Phase 28 failure mode). Takes precedence "
                             "over --cbm_coupled (default True).")
    parser.add_argument("--no_cbm_pooled", dest="cbm_pooled",
                        action="store_false",
                        help="Phase 29: disable pooled CBM input (falls back to "
                             "cbm_coupled / raw-Q behaviour; ablation switch).")
    # ── Phase 27: DANN flags ────────────────────────────────────────────
    parser.add_argument("--dann_enabled", dest="dann_enabled",
                        action="store_true", default=None,
                        help="Phase 27: enable Domain Adversarial Training. Adds "
                             "4 synthetic augmentation domains + DomainHead with "
                             "GRL on attn_pool to push features toward "
                             "domain-invariance.")
    parser.add_argument("--no_dann", dest="dann_enabled", action="store_false",
                        help="Phase 27: disable DANN entirely (single-domain training).")
    parser.add_argument("--num_domains", type=int, default=None,
                        help="Phase 27: number of synthetic DANN domains (default 4).")
    parser.add_argument("--lambda_domain", type=float, default=None,
                        help="Phase 27: weight on domain CE loss (default 0.10). "
                             "Warmed up linearly over domain_warmup_epochs.")
    parser.add_argument("--lambda_grl", type=float, default=None,
                        help="Phase 27: max GRL gradient scale (default 1.0). "
                             "Warmed up linearly over domain_warmup_epochs.")
    parser.add_argument("--domain_warmup_epochs", type=int, default=None,
                        help="Phase 27: epochs for linear warmup of lambda_domain "
                             "and lambda_grl (default 3).")
    parser.add_argument("--snapshot_every", type=int, default=None,
                        help="Save Phase 21 snapshot every N epochs (default 2).")
    parser.add_argument("--no_early_stop", dest="no_early_stop",
                        action="store_true", default=None,
                        help="Disable early stopping — run all epochs regardless of metric plateau.")
    parser.add_argument("--lambda_peak_spread", type=float, default=None,
                        help="Weight for HardAttentionDiversityLoss (default 0.5).")
    parser.add_argument("--lambda_sharp", type=float, default=None,
                        help="Weight for sharpness loss on M_t_logits (default 1.0).")
    parser.add_argument("--disk_guard_gb", type=float, default=None,
                        help="Min free disk GB before face-cache write is skipped (default 3.0).")
    return parser.parse_args()
