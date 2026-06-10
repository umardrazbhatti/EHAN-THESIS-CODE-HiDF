"""
models/HiDF_eahn.py — Explanation-Aware Hybrid Network (EAHN).

v4 patch — mt_std ceiling fix:
  ROOT CAUSE: M_t is a softmax over 49 cells (7×7). The theoretical maximum
  std of a softmax distribution over D=49 cells is sqrt((1-1/D)/D) ≈ 0.141.
  The diagnostic threshold is 0.15 — IMPOSSIBLE to reach with softmax values.

  FIX: EAHNOutput now carries M_t_logits (pre-softmax logits, unnormalised).
  loss_sharp in train_real.py is computed on M_t_logits instead of M_t.
  The diagnostic mt_std is also computed on M_t_logits (std of raw scores,
  range unbounded, easily exceeds 0.15 once the conv learns to peak).

  M_t (softmax) is kept for all gating, attention pooling, and loss_faith —
  it must remain a proper probability distribution for those paths.

  Also: EAHNOutput now exposes early_attn_tau so train_real.py can log the
  actual sharpening temperature (not cross_attention.log_temp which is dead).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from HiDF_config import EAHNConfig
from models.HiDF_spatial_stream import SpatialStream
from models.HiDF_temporal_stream import TemporalStream
from models.HiDF_cross_attention import CrossAttentionFusion
from models.HiDF_concept_bottleneck import ConceptSlotBottleneck   # Phase 26
from models.HiDF_grl import DomainHead, grad_reverse                # Phase 27


@dataclass
class EAHNOutput:
    logit:          torch.Tensor
    prob:           torch.Tensor
    M_t:            torch.Tensor   # (B, T, h, w) — softmax, sums to 1 per (b,t)
    M_t_logits:     torch.Tensor   # (B, T, h, w) — pre-softmax raw scores (for mt_std loss/diag)
    M_t_up:         torch.Tensor   # (B, T, H, W)
    S:              torch.Tensor   # (B, T, N, d_model)
    low_level:      torch.Tensor   # (B, T, C_low, Hl, Wl)
    attn_pool:      torch.Tensor   # (B, d_model)
    early_attn_tau: float          # exp(log_tau) at forward time — for logging
    # ── Phase 23: temporal-attention bottleneck (frame-level M_t) ─────────────
    # M_frame = softmax over T frames, computed AFTER the per-frame pool so the
    # classifier MUST depend on which frames are weighted high. Replaces the
    # uniform `.mean(dim=1)` that made all k1/k2/k4 frame-drop ratios ≈ 1.0x.
    # Shape (B, T). Sums to 1 per sample. Read by metrics.HiDF_explanation
    # .frame_attention_drop_test for ranking instead of M_t.amax.
    M_frame:        torch.Tensor   # (B, T)
    frame_attn_tau: float          # exp(frame_log_tau) — for logging
    # ── Phase 25: bi-directional refinement diagnostic ───────────────────────
    # α = sigmoid(refine_gate); M_t = α * M_t_refined + (1-α) * M_t_early.
    # 0.0 when bidirectional_enabled=False.  Logged in first-batch diagnostics
    # so we can confirm the refinement gate is opening across epochs.
    refine_alpha:   float = 0.0
    # ── Phase 26: CBM (Concept Slot Bottleneck) outputs ──────────────────────
    # Phase 27: in serial mode, out.logit IS cbm_logit.  main_logit is exposed
    # for the auxiliary-supervision loss term and as a diagnostic but does NOT
    # participate in prediction.
    cbm_logit:       torch.Tensor = None
    main_logit:      torch.Tensor = None
    concept_scores:  torch.Tensor = None
    slot_attn:       torch.Tensor = None
    cbm_blend:       float = 0.0
    # ── Phase 27: DANN domain classifier outputs ─────────────────────────────
    # domain_logits   : (B, num_domains)  predictions of the DANN domain head
    #                                     fed by GRL(attn_pool).  Cross-entropy
    #                                     with `domain` labels from the batch.
    # cbm_serial      : bool              True when out.logit = cbm_logit
    #                                     (no parallel-blend escape hatch).
    domain_logits:   torch.Tensor = None
    cbm_serial:      bool = False
    # ── Phase 28: True when CBM input tokens are scaled by M_t ⊙ M_frame ─────
    # (the coupling that makes the explanation maps load-bearing for the
    # prediction; False = Phase 27 raw-Q behaviour).
    cbm_coupled:     bool = False


class EarlyAttnHead(nn.Module):
    """Phase 21: produces M_t from CNN feature map BEFORE the transformer.

    v4: returns (M_softmax, logits_raw) so callers can use the proper quantity
    for loss_sharp and the mt_std diagnostic without hitting the softmax ceiling.

    Softmax ceiling problem: softmax over D=49 cells has max std ≈ sqrt((1-1/D)/D)
    ≈ 0.141, which is below the required threshold of 0.15. Using raw logits (std
    of unnormalised scores) has no such ceiling — the conv network just needs to
    learn to produce high-variance score maps, which is much easier to optimise.

    M_softmax is still used for gating and attention pooling (must sum to 1).
    logits_raw is used only for loss_sharp and mt_std diagnostic.
    """
    def __init__(self, d_model: int = 256, hidden: int = 64,
                 init_temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        # Init log_tau to log(0.5) so tau starts at 0.5 (sharper than exp(0)=1)
        self.log_tau = nn.Parameter(torch.tensor(-0.693))  # exp(-0.693) ≈ 0.5

    def forward(self, feats):  # feats: (B, T, C, H, W)
        B, T, C, H, W = feats.shape
        x = feats.reshape(B * T, C, H, W)
        logits = self.proj(x).reshape(B, T, H * W)          # (B, T, H*W) raw scores
        tau = self.log_tau.exp().clamp(min=0.1, max=3.0)
        M = F.softmax(logits / tau, dim=-1)                  # (B, T, H*W), sums to 1
        M_spatial   = M.reshape(B, T, H, W)
        logits_spatial = logits.reshape(B, T, H, W)
        return M_spatial, logits_spatial, tau.item()


class EAHN(nn.Module):
    def __init__(self, config: EAHNConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # ── Spatial Stream ────────────────────────────────────────────────────
        self.spatial_stream = SpatialStream(
            backbone_name=config.backbone,
            pretrained=config.backbone_pretrained,
            d_model=d,
            freeze_backbone=False,
        )

        dummy = torch.zeros(1, 3, config.frame_size, config.frame_size)
        with torch.no_grad():
            dummy_tokens = self.spatial_stream(dummy)
        N = dummy_tokens.shape[1]
        self.N      = N
        self.feat_h = self.spatial_stream.feat_h
        self.feat_w = self.spatial_stream.feat_w

        # ── Early Attention Head (Phase 21, v3) ───────────────────────────────
        self.early_attn = EarlyAttnHead(d_model=d, hidden=64)
        self.attn_floor = float(getattr(config, "attn_floor", 0.05))

        # ── Temporal Stream ───────────────────────────────────────────────────
        max_seq = config.num_frames * N + 1
        self.temporal_stream = TemporalStream(
            d_model=d,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            dropout=config.dropout,
            max_seq_len=max_seq,
        )

        # ── Cross-Attention Fusion ────────────────────────────────────────────
        self.cross_attention = CrossAttentionFusion(
            d_model=d,
            num_heads=config.transformer_heads,
            attn_temp_init=getattr(config, "attn_temp_init", 0.0),
        )

        # ── Temporal Attention Bottleneck (Phase 23) ──────────────────────────
        # Replaces the uniform `attn_pool_per_frame.mean(dim=1)` with a learned
        # per-frame weighting M_frame.  This is the critical fix for k1/k2/k4
        # frame-drop ratios (which were stuck at 1.0x because the classifier
        # consumed all frames democratically). With this gate, removing the
        # top-attended frame substantially harms prediction → k1 ratio >> 1.
        #
        # Architecture (~17k params, +0.1% of model size):
        #   LayerNorm(d) → Linear(d → d/4) → GELU → Linear(d/4 → 1) → softmax over T
        # Learnable temperature `frame_log_tau` (init exp(-0.693)=0.5) controls
        # how peaky M_frame can get; clamped to [0.1, 3.0] for stability.
        self.temporal_gate = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, max(d // 4, 32)),
            nn.GELU(),
            nn.Linear(max(d // 4, 32), 1),
        )
        self.frame_log_tau = nn.Parameter(torch.tensor(-0.693))   # tau ≈ 0.5 init

        # ── Phase 25: Bi-directional refinement gate ──────────────────────────
        # Pre-Phase-25 the CrossAttentionFusion module was wired into a discarded
        # `_legacy_M_t` variable.  Phase 25 plugs its output back in as a
        # *refined* M_t that has seen temporal context (Q comes from the
        # transformer).  M_t_used = α * M_t_refined + (1-α) * M_t_early where
        # α = sigmoid(refine_gate).  refine_gate is initialised at -2.0 so
        # α ≈ 0.12 at epoch 0 — Phase 24 behaviour is preserved at start, and
        # the model learns how much to trust the refined map as training
        # progresses.  Set config.bidirectional_enabled=False to disable.
        self.bidirectional_enabled = bool(
            getattr(config, "bidirectional_enabled", True)
        )
        # Phase 26: refine_gate init raised from -2.0 to -0.5 (sigmoid 0.119 →
        # 0.378) so the bidirectional path actually engages from epoch 1.
        # Phase 25 evidence: alpha only crept 0.118 → 0.121 over 8 epochs,
        # never exceeding the init value, because every other knob was
        # destabilising training before the gate could open.
        _refine_init = float(getattr(config, "refine_gate_init", -0.5))
        self.refine_gate = nn.Parameter(torch.tensor(_refine_init))

        # ── Classification Head ───────────────────────────────────────────────
        self.classifier = nn.Linear(d, 1)

        # ── Phase 26: Concept Slot Bottleneck (parallel interpretable head) ──
        # K learned slot queries → soft-attention over (T*N) transformer
        # features → K scalar concept scores → Linear(K → 1) parallel logit.
        # Combined with main logit via learnable sigmoid blend (init 0.5).
        # See models/HiDF_concept_bottleneck.py for full description.
        self.cbm_enabled = bool(getattr(config, "cbm_enabled", True))
        if self.cbm_enabled:
            self.cbm = ConceptSlotBottleneck(
                d_model=d,
                num_slots=int(getattr(config, "cbm_num_slots", 8)),
            )
        else:
            self.cbm = None

        # ── Phase 27: serial CBM flag ─────────────────────────────────────────
        # When True, out.logit = cbm_logit (no main/CBM blend).  main_logit
        # is still exposed and supervised via lambda_cbm_main_aux, but never
        # participates in prediction.  This removes the Phase 26 escape hatch
        # that decoupled M_t from prediction (and tanked faithfulness 0.225 ->
        # 0.079).  Default True; set --no_cbm_serial to revert to Phase 26
        # parallel-blend behaviour for ablation.
        self.cbm_serial = bool(getattr(config, "cbm_serial", True))

        # ── Phase 28: CBM-attention coupling ──────────────────────────────────
        # Phase 27 post-mortem: serial CBM read RAW transformer tokens Q_flat,
        # bypassing both the M_t-weighted spatial pooling and the M_frame
        # temporal gate.  Result: detection 0.914 (best ever) but k1 ratio
        # exactly 1.000 and faith_corr 0.071 — the run log proved M_frame was
        # near one-hot (99.998% on one frame) yet dropping that frame moved the
        # prediction no more than dropping a random frame.  The maps were
        # decoration.
        # Fix: scale each token fed to the CBM by its joint spatio-temporal
        # attention mass w = M_t[b,t,n] * M_frame[b,t], max-normalised per
        # sample.  A token the maps ignore now contributes ~nothing to the
        # prediction, so classification loss directly punishes dishonest maps.
        # Set --no_cbm_coupled to revert to Phase 27 raw-Q behaviour (ablation).
        self.cbm_coupled = bool(getattr(config, "cbm_coupled", True))

        # ── Phase 27: DANN domain classifier ──────────────────────────────────
        # GRL(attn_pool) -> DomainHead -> (B, D) logits.  In training the
        # batch carries a per-sample `domain` label in [0, D-1] (random
        # synthetic augmentation domain assigned at __getitem__).  Loss is
        # CE on domain_logits with sign-reversed gradient flowing back to
        # attn_pool, pushing it to be domain-invariant.
        self.dann_enabled = bool(getattr(config, "dann_enabled", True))
        if self.dann_enabled:
            self.domain_head = DomainHead(
                d_model=d,
                num_domains=int(getattr(config, "num_domains", 4)),
                dropout=float(getattr(config, "dropout", 0.1)),
            )
        else:
            self.domain_head = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def enable_gradient_checkpointing(self):
        if hasattr(self.temporal_stream, "enable_gradient_checkpointing"):
            self.temporal_stream.enable_gradient_checkpointing()
        if hasattr(self.spatial_stream, "set_grad_checkpointing"):
            self.spatial_stream.set_grad_checkpointing(True)

    def forward(self, frames: torch.Tensor,
                lambda_grl: float = 0.0) -> EAHNOutput:
        """Phase 27: optional `lambda_grl` controls the GRL strength on the
        DANN domain head.  Default 0.0 keeps Phase-26 behaviour (domain head
        still runs but does NOT pull attn_pool toward invariance).  Training
        script ramps lambda_grl 0 -> lambda_grl_max over domain_warmup_epochs.
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        spatial_tokens = self.spatial_stream(frames_flat)    # (B*T, N, d)
        low_feat = self.spatial_stream.low_level_features()  # (B*T, C_low, Hl, Wl)

        N = spatial_tokens.shape[1]
        d = self.config.d_model
        C_low, Hl, Wl = low_feat.shape[1], low_feat.shape[2], low_feat.shape[3]

        feats_5d = (
            spatial_tokens
            .permute(0, 2, 1)
            .reshape(B * T, d, self.feat_h, self.feat_w)
            .reshape(B, T, d, self.feat_h, self.feat_w)
        )
        # v4: unpack (softmax map, raw logits, tau scalar)
        M_t_early, M_t_logits, _tau_val = self.early_attn(feats_5d)
        gate = (M_t_early + self.attn_floor) / (1.0 + self.attn_floor)
        spatial_tokens = (
            feats_5d * gate.unsqueeze(2)
        ).reshape(B * T, d, self.feat_h * self.feat_w).permute(0, 2, 1)

        spatial_tokens = spatial_tokens.view(B, T, N, d)
        low_level      = low_feat.view(B, T, C_low, Hl, Wl)

        Q, cls_out = self.temporal_stream(spatial_tokens.reshape(B, T * N, d))
        Q = Q.reshape(B, T, N, d)

        # ── Phase 25: bi-directional refinement ──────────────────────────────
        # Run CrossAttentionFusion with transformer-context Q vs spatial keys.
        # Its M_t output (column-mean of attention) is now USED, not discarded.
        # M_t_used = α * M_t_refined + (1-α) * M_t_early, α = sigmoid(refine_gate).
        # α starts ≈ 0.12 so the first epoch matches Phase 24 behaviour; the
        # model learns how much to trust the refinement as it converges.
        if self.bidirectional_enabled:
            M_t_refined, _xa_pool = self.cross_attention(Q, spatial_tokens)
            alpha   = torch.sigmoid(self.refine_gate)
            M_t_use = alpha * M_t_refined + (1.0 - alpha) * M_t_early
            _alpha_val = float(alpha.item())
        else:
            M_t_use    = M_t_early
            _alpha_val = 0.0

        M_flat = M_t_use.reshape(B, T, N)
        attn_pool_per_frame = (Q * M_flat.unsqueeze(-1)).sum(dim=2)   # (B, T, d)

        # ── Phase 23: temporal attention bottleneck (replaces .mean(dim=1)) ──
        # Force the classifier to depend on which FRAMES carry attention mass,
        # not just average all T frames democratically.  This is the structural
        # fix for k1/k2/k4 ratios (previously ~1.0x because removing any frame
        # only removed 1/T of an unweighted mean).
        frame_logits = self.temporal_gate(attn_pool_per_frame).squeeze(-1)  # (B, T)
        _tau_frame   = self.frame_log_tau.exp().clamp(min=0.1, max=3.0)
        M_frame      = F.softmax(frame_logits / _tau_frame, dim=-1)         # (B, T)
        attn_pool    = (attn_pool_per_frame * M_frame.unsqueeze(-1)).sum(dim=1)  # (B, d)

        # Phase 25: M_t_up now reflects the REFINED map so the bottleneck
        # construction (loss_ins/loss_faith) and downstream metric code both
        # operate on the same M_t that the classifier consumes.
        M_t_up_use = F.interpolate(
            M_t_use.reshape(B * T, 1, self.feat_h, self.feat_w),
            size=(H, W), mode="bilinear", align_corners=False,
        ).reshape(B, T, H, W)

        main_logit = self.classifier(attn_pool).squeeze(-1)

        # ── Phase 26+27: Concept Slot Bottleneck ──────────────────────────────
        # The CBM reads transformer Q (B, T*N, d) — same features the standard
        # classifier consumes after M_t-gated pooling, but BEFORE the temporal
        # gate collapses time.  Slots can therefore attend to (frame, position)
        # pairs and discover spatio-temporal concepts.
        #
        # Phase 27 (serial): out.logit = cbm_logit only.  main_logit is exposed
        # for the auxiliary supervision loss (regulariser) but is NOT in the
        # prediction path -- this removes the Phase 26 escape hatch.
        if self.cbm is not None:
            Q_flat = Q.reshape(B, T * N, d)
            # ── Phase 28: couple CBM input to the explanation maps ────────────
            # w[b, t*N+n] = M_t_use[b,t,n] * M_frame[b,t]  (joint mass over the
            # T*N grid; sums to 1 per sample since both factors are softmaxes).
            # Max-normalise so the top token keeps full magnitude (mean-norm
            # would blow token scale up ~T*N-fold for peaky maps → fp16 risk).
            # Gradient w.r.t. M_frame/M_t stays alive even where w underflows
            # to 0 because slot_pool is linear in w.
            if self.cbm_coupled:
                w = (M_flat * M_frame.unsqueeze(-1)).reshape(B, T * N, 1)
                w = w / w.amax(dim=1, keepdim=True).clamp(min=1e-6)
                Q_cbm = Q_flat * w
            else:
                Q_cbm = Q_flat
            cbm_logit, concept_scores, slot_attn, cbm_blend = self.cbm(Q_cbm)
            if self.cbm_serial:
                # Phase 27: pure serial bottleneck — cbm_logit is the sole
                # prediction.  main_logit is only used by the aux loss.
                logit = cbm_logit
            else:
                # Phase 26 fallback (parallel blend, --no_cbm_serial)
                beta_main = torch.sigmoid(self.cbm.blend)
                logit = beta_main * main_logit + (1.0 - beta_main) * cbm_logit
        else:
            cbm_logit, concept_scores, slot_attn, cbm_blend = None, None, None, 0.0
            logit = main_logit

        prob = torch.sigmoid(logit)

        # ── Phase 27: DANN domain head ────────────────────────────────────────
        # Forward attn_pool through GRL(lambda) then domain head.  Training
        # script ramps lambda from 0 over domain_warmup_epochs so the domain
        # head learns a real signal before it starts trying to confuse the
        # backbone.  At eval we still emit domain_logits but the training loop
        # only uses them when labels are present.
        if self.domain_head is not None:
            pool_rev      = grad_reverse(attn_pool, float(lambda_grl))
            domain_logits = self.domain_head(pool_rev)
        else:
            domain_logits = None

        return EAHNOutput(
            logit=logit, prob=prob,
            M_t=M_t_use, M_t_logits=M_t_logits,
            M_t_up=M_t_up_use,
            S=spatial_tokens, low_level=low_level,
            attn_pool=attn_pool,
            early_attn_tau=_tau_val,
            M_frame=M_frame,
            frame_attn_tau=float(_tau_frame.item()),
            refine_alpha=_alpha_val,
            cbm_logit=cbm_logit,
            main_logit=main_logit,
            concept_scores=concept_scores,
            slot_attn=slot_attn,
            cbm_blend=cbm_blend,
            domain_logits=domain_logits,
            cbm_serial=self.cbm_serial,
            cbm_coupled=(self.cbm_coupled and self.cbm is not None),
        )
