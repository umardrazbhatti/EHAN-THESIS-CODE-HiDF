# EAHN-HiDF Phase-by-Phase Changelog

Single source of truth for every architectural change shipped to date.  Use
this when writing the ablation section of the paper — each row is one
self-contained increment with its rationale, the metric outcome it produced
on a real Kaggle run, and the commit hash that landed it.

> **How to use this file**
> - Add a new row whenever a numbered Phase ships to GitHub.
> - Update the "outcome" column only after the next Kaggle run completes.
> - Keep one paragraph of "what we learned" so the ablation story is honest.
> - Never edit a past row except to add the post-run outcome — history is
>   what makes ablations credible.

---

## Phase ledger

| Phase | Date | Commit | Headline change | HiDF AUC | CelebDF AUC | Insertion AUC | Faithfulness corr | k1 ratio | Status |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (origin) | — | EfficientNet-B4 + 4-layer Transformer + softmax M_t at 7×7, classifier = Linear(d→1) over mean-temporal-pool | — | — | — | — | — | baseline |
| 18 | 2026-05-?? | (pre-history) | Class-symmetric augmentation, focal loss, label smoothing 0.01 | — | — | — | — | — | pre-explanation work |
| 19 | 2026-05-?? | — | Balanced 100R+100F CelebDF/FF++ cross-eval; per-manipulation FF++ split | — | — | — | — | — | eval-only |
| 20 | 2026-05-?? | — | `attn_floor` raised so M_t can never be exactly zero; Gaussian-blur fill in insertion/deletion metric | — | — | — | — | — | metric hygiene |
| 21 | 2026-06-?? | — | **EarlyAttnHead** generates M_t from CNN features BEFORE transformer; M_t gates spatial tokens going into transformer; bottlenecked input + faithfulness KL loss | — | — | — | — | — | core attention path |
| 22 | 2026-06-?? | — | `bottleneck_peak_floor=0.25`: prevents diffuse M_t from making bottleneck identical to original; KL loss starts biting | — | — | — | — | — | bottleneck pressure |
| 23 | 2026-06-08 | `922b5cd` | **temporal_gate bottleneck** — Linear(d→d/4)→Linear(d/4→1)→softmax over T replaces `.mean(dim=1)`; `attn_floor 0.05→0.0` (true spatial bottleneck); moderated aug strengths | 0.904 | 0.66 | 0.249 | 0.18 | 1.64 | partial win (insertion regressed) |
| 24 | 2026-06-09 | `065129e` | **loss_ins** = focal-CE on bottleneck-forward logits, re-uses existing `out_B` (zero extra forward pass); `lambda_ins=0.5` with 3-epoch warmup; insertion-AUC training objective | 0.825 | 0.649 | 0.499 | 0.225 | 0.97 | detection -8pts; insertion +25pts |
| 25 | 2026-06-09 | `a6597bd` | **Bi-directional refinement**: cross_attention rewired (was discarded `_legacy_`), learnable `refine_gate` blend; **hard top-K binary bottleneck mask** (straight-through, K=20%); `lambda_ins 0.5→1.0`, `lambda_sparse 0.05→0.10`, `alpha→0`, `beta→0.1`; **temporal_sparsity_loss** on M_frame; `lambda_temp_sparse=0.05` | 0.708 | 0.535 | 0.554 | **0.117** | 0.66 | **REGRESSED** — too many knobs at once; M_t over-collapsed; real_acc fell to 0.14 |
| 26 | 2026-06-10 | TBD | **Revert** Phase 25 amplifications to Phase 24 values; **`refine_gate` init -2.0→-0.5** so α≈0.38 from epoch 1 (Phase 25 never engaged the bidirectional path); **WeightedRandomSampler** to combat the 14% real-class collapse seen in Phase 25; **Concept Slot Bottleneck (CBM)** — K=8 learned slot queries, parallel classifier head, slot-diversity loss; targets faithfulness via concept organisation | TBD | TBD | TBD | TBD | TBD | gamble |

---

## Phase 26 — implementation detail

### A. Reverts (low-risk, Phase-25-collapse rollback)

| Flag | Phase 25 value | Phase 26 value | Why |
|---|---|---|---|
| `lambda_ins` | 1.0 | **0.5** | Hard top-K + lambda=1.0 dominated the gradient, crushing classification signal |
| `lambda_sparse` | 0.10 | **0.05** | Over-sparsified M_t into one-hot per frame, classifier lost spatial context |
| `alpha` (entropy in L_exp) | 0.0 | **0.05** | Removing entropy regulariser let M_t collapse with nothing pushing back |
| `beta` (TV in L_exp) | 0.1 | **0.5** | Less TV smoothing made M_t pointillistic; restored to standard |
| `bottleneck_hard_topk_frac` | 0.20 | **0.0** | Hard mask + class imbalance pushed model to "always predict fake" |
| `lambda_temp_sparse` | 0.05 | **0.02** | Pushed M_frame to one-hot on frame 0 (sample 1: 79% mass on one frame) |

### B. Bi-directional improvement
- `refine_gate` init: `-2.0` → `-0.5`.  Sigmoid(-2.0)=0.119 vs sigmoid(-0.5)=0.378.
- Phase 25 evidence: `refine_alpha` only moved 0.118→0.121 across all 8 epochs — the gate never engaged because every other knob was destabilising training.
- New init means cross_attention contributes ~38% to M_t from epoch 1, giving the gate gradient something to actually optimise.

### C. Class-balanced sampling
- `WeightedRandomSampler` with weights = `1/n_class` per sample.
- Optional flag `--class_balanced_sampler` (default True for HiDF where real:fake = 3488:3175).
- Phase 25 evidence: real_acc collapsed to 0.142 (375/437 reals flagged as fake) — model defaulted to majority class.

### D. Concept Slot Bottleneck (CBM)
- New module: `models/HiDF_concept_bottleneck.py`
- K=8 learned slot queries `Q_k`, dot-product attention over transformer features
- Each slot produces (1) a spatial attention map over (T·N) positions, (2) a pooled vector, (3) a scalar concept score via `slot_v`
- Final CBM logit = `Linear(K → 1)(concept_scores)`
- Combined with main logit via learnable blend `β` = sigmoid(blend), init 0.0 (50/50 blend)
- **Slot diversity loss** = mean off-diagonal cosine similarity between slot attention vectors; minimised → slots attend to *different* regions
- **CBM auxiliary classification loss** = focal-CE on `cbm_logit` alone (regularises the CBM head independently)
- Combined loss term: `lambda_cbm * (loss_cbm_aux + 0.5 * loss_cbm_div)`, `lambda_cbm` default 0.1

### Compute budget
- CBM parameters: K * d * 2 + K = 8 * 256 * 2 + 8 = ~4.1k extra params (negligible vs B4 backbone's 17M)
- CBM forward: K dot products + K softmaxes over T·N=784 positions + K scalar projections ≈ 1-2% wall-clock overhead
- Slot diversity loss: K² cosine similarities ≈ negligible

---

## Phase 27+ planned (NOT in this commit)

- **Phase 27 — DIAT/DANN** (domain-invariant adversarial training): augmentation-shifted synthetic domains + gradient reversal layer to make features domain-invariant; targets CelebDF/FF++ cross-AUC.  Requires data-pipeline changes (per-sample domain labels in collate) — deferred to avoid Phase-25-style "too many knobs at once."
- **Phase 28 — SSL pretraining** on a mixed deepfake corpus (DFDC + FaceForensics++ unlabelled subsets) before fine-tuning on HiDF.
- **Phase 29 — Test-Time Adaptation (TTA)**: at eval time, run K=10 SGD steps on entropy minimisation per cross-dataset batch.

---

## Standing rule (lesson from Phase 25)

**Change one architectural axis per phase.**
Phase 25 bundled seven simultaneous amplifications and we couldn't attribute which knob caused the collapse.  Phase 26 changes only the CBM axis (plus reverts that *undo* Phase 25); future phases ship one new architectural direction at a time so the ablation table has clean attribution.
