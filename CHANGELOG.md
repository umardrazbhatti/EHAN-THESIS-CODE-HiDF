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
| 26 | 2026-06-10 | `dead521` | **Revert** Phase 25 amplifications to Phase 24 values; **`refine_gate` init -2.0→-0.5** so α≈0.38 from epoch 1 (Phase 25 never engaged the bidirectional path); **WeightedRandomSampler** to combat the 14% real-class collapse seen in Phase 25; **Concept Slot Bottleneck (CBM)** — K=8 learned slot queries, **parallel** classifier head, slot-diversity loss | **0.800** | 0.618 | 0.499 | **0.079** | 1.000 | partial — detection recovered, real_acc 0.689 (BEST), but **parallel CBM gave the model an escape hatch** → M_t decoupled from prediction → faithfulness regressed to worst-ever value, k_ratios flat-uniform. FF++ FaceSwap fell to 0.439 (below random). |
| 27 | 2026-06-11 | `a60dd31` | **Serial CBM** — out.logit IS cbm_logit; main_logit kept only as aux-supervision regulariser via `lambda_cbm_main_aux=0.05`; **K bumped 8 → 12** for serial expressiveness. **DANN domain adversarial training (DIAT variant)** — 4 synthetic domains via augmentation (clean / heavy-JPEG / noise / blur), per-sample random domain at `__getitem__`, GRL on attn_pool, DomainHead MLP (~34k params, ~0.7% of model). Linear warmup of `lambda_grl` 0→1.0 and `lambda_domain` 0→0.10 over 3 epochs. Detailed rolling-log + first-batch DIAG-P27 prints + 3 new history columns (`train_domain`, `train_domain_acc`, `train_cbm_main_aux`) for attribution. | **0.914** | 0.593 | 0.541 | 0.071 | 1.000 | split verdict — **detection best-ever (0.914, +0.114 vs P26)**, training stable for the first time since P23, DANN engaged exactly as designed (dom_acc → chance 0.25, L_domain pinned at ln 4). But explanations stayed dead because the serial CBM reads **RAW transformer Q**, bypassing both the M_t-weighted pooling and the M_frame temporal gate — run log proved M_frame was 99.998% one-hot on one frame yet dropping that frame ≡ dropping a random frame. The escape hatch wasn't removed in P27; it moved. FF++ FaceSwap fell further to 0.381 — DANN over processing domains cannot teach manipulation-method generalisation. Two metric confounds also identified: zero-fill frame drop = +0.37 fake-prob artifact for ANY masked frame; blurred clips score 0.82 fake vs 0.47 baseline so del/ins curves invert regardless of map quality. |
| 28 | 2026-06-11 | `77b0c69` | **CBM-attention coupling** — CBM input tokens scaled by `w = M_t ⊙ M_frame` (max-normalised per sample, `cbm_coupled=True`, `--no_cbm_coupled` ablation): the explanation maps become load-bearing for the prediction, so cls loss directly punishes dishonest maps (P23 precedent: coupled path gave 0.904 AUC + k1 1.64 + faith 0.18). **Metric protocol fixes (eval-only)**: frame-drop test fill changed zero → nearest-frame **replication** (legacy zero-fill kept under `*_zerofill` keys); del/ins gains a **random-saliency control** (`ins/del_gain_over_random` — robust to the blur-direction artifact), fake-only aggregates, chunked forwards, and bigger samples (evaluate 5→20 clips, suite 10→50, steps 10→20). `--lambda_cbm_aux 0.0` (in serial mode it duplicated loss_cls → cls double-counted at 1.10). New diagnostics: `[DIAG-P28]` first-batch print + rolling `eff_tok=` + history column `train_eff_tokens` (effective token count of w; watch: collapse → ~1 means bottleneck too brutal, stuck at T·N means coupling not biting). | 0.541 | 0.470 | 0.500 | (noise) | (noise) | **CATASTROPHIC — training never started** (run 6-11-26 1300hrs). Output constant at prob ≈0.4996 (= sigmoid of the CBM fc bias) for all 9 epochs: `cls` frozen at 0.1838 from batch 1000, val AUC never left ~0.5, every test clip predicted "real" (tp=0). Root cause: **scaling token MAGNITUDES is not pooling** — once the sparsity regularisers made the maps peaky, 782/784 CBM keys were near-zero, the slot softmax diluted the survivors (~780 dead keys at e⁰ each soak up the mass), `slot_pool` collapsed to ~0, the logit froze at the fc bias, the cls gradient died, and the regularisers ground `eff_tok` from 56 → 2 unopposed (one-way ratchet; `tsparse` reward hit −0.98 by end of epoch 1, `cbm_div` pinned at 0.9995 = all 12 slots identical). The metric protocol fixes (replicate fill, random-saliency control) are KEPT — they are eval-only and orthogonal to the collapse. The k1=2.35/faith=0.33 numbers in that run's files are noise on a constant function — never quote them. |
| 29 | 2026-06-11 | TBD | **Pooled-frame CBM — repair the coupling** (`cbm_pooled=True`, `--no_cbm_pooled` ablation; supersedes `cbm_coupled`). The CBM reads `attn_pool_per_frame` (B, T, d) — the M_t-pooled per-frame vectors. A **convex combination**: full magnitude however peaky M_t gets, and tokens M_t suppresses are structurally ABSENT from the prediction (spatial coupling cannot be bypassed). M_frame couples as a **log-space attention prior** inside the CBM (`attn_logits += log(M_frame)`, renormalised by the softmax — the temporal coupling cannot starve magnitudes either). Both couplings are renormalised forms, so the P28 starvation mode is structurally impossible. **`lambda_temp_sparse 0.02 → 0.0`** — it drove M_frame to 99.998% one-hot in P27 and was the P28 collapse accelerant; P23 had no temporal sparsity reward and still hit k1 1.64 because a load-bearing M_frame concentrates itself. New diagnostics: `[DIAG-P29]` (incl. `logit_std` constant-output warning — the P28 signature), rolling `eff_fr=`/`eff_sp=`/`s_top=` (slot mass on top-M_frame frame = prior-adherence / escape-hatch detector), history columns `train_eff_frames`/`train_eff_spatial`/`train_slot_on_top`. Smoke gate now includes a 30-step mini-overfit (loss must decrease — would have caught P28 pre-push). P23 precedent for this path: 0.904 AUC / k1 1.64 / faith 0.18 under the old confounded protocol; P29 adds the K=12 slot readout + sampler + DANN on top. | TBD | TBD | TBD | TBD | TBD | **repair the loop** |

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

## Phase 30+ planned (NOT in this commit)

- **Phase 30 — SBI self-blending augmentation** (Face X-ray / SBI style): synthesise pseudo-fakes from HiDF REAL clips by blending a transformed copy of the face back onto itself → teaches generic blending-boundary artifacts. This is the literature-standard fix for cross-dataset transfer and the only credible path to CelebDF/FF++ ≥ 0.75 (P27 evidence: DANN over processing domains left FaceSwap at 0.381). Real build: `data/HiDF_synthetic_generator.py` is currently a toy rectangle generator.
- **Phase 31 — SSL pretraining** on a mixed deepfake corpus (DFDC + FaceForensics++ unlabelled subsets) before fine-tuning on HiDF.
- **Phase 32 — Test-Time Adaptation (TTA)**: at eval time, run K=10 SGD steps on entropy minimisation per cross-dataset batch.

## Metric protocol note (Phase 28)

The k1/k2/k4 frame-drop columns are NOT directly comparable across the P28 boundary: P≤27 numbers used zero-fill (gray frame — carries a +0.37 fake-prob artifact), P28+ headline numbers use nearest-frame replication. The legacy protocol is still computed every run under `k{K}_*_zerofill` keys, so a same-protocol comparison is always available. Similarly, del/ins gains `ins_gain_over_random` / `del_gain_over_random` (P28+) are the artifact-robust quantities to quote; the absolute del/ins AUCs remain confounded by the model's blur→fake response (blurred clip scores 0.82 vs 0.47 baseline).

---

## Standing rule (lesson from Phase 25)

**Change one architectural axis per phase.**
Phase 25 bundled seven simultaneous amplifications and we couldn't attribute which knob caused the collapse.  Phase 26 changes only the CBM axis (plus reverts that *undo* Phase 25); future phases ship one new architectural direction at a time so the ablation table has clean attribution.

## Standing rule (lesson from Phase 28)

**Couple through renormalised forms only, and smoke-test trainability, not just gradients.**
1. Any mechanism that makes attention maps load-bearing must preserve feature
   MAGNITUDE no matter how peaky the maps get: convex pooling (`Σ wᵢ·xᵢ` with
   `Σ wᵢ = 1`) or softmax-renormalised log-priors. Multiplicative masking of
   token magnitudes is banned — peaky maps + softmax attention over the masked
   tokens = dilution → constant output → dead cls gradient → the sparsity
   regularisers collapse the maps unopposed (the P28 one-way ratchet).
2. A pre-push smoke test must include (a) per-sample logit VARIANCE relative
   to a known-good mode, and (b) a ~30-step mini-overfit asserting the loss
   actually decreases. P28's smoke verified gradient existence at init — true
   but insufficient; either check above would have caught the death before it
   cost an 8-hour Kaggle run.
