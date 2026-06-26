# EAHN-HiDF — Experiment Registry (frozen baseline)

**Purpose.** Single authoritative record of the current best models and their exact
state, so that every future amendment has a known "last good" to compare against and
**roll back to**. Update this file whenever a run is promoted to a new baseline.

---

## Rollback anchors (how to get back to "now")

| What | Anchor |
|---|---|
| **Code (models trained on)** | git commit `ee6e9ca` (Phase 45 notebooks + metric + sigconc) |
| **Code (reporting fix, eval-only)** | git commit `c3acbc2` — does NOT change models |
| **Git tag for this baseline** | `p45-baseline` (points at `c3acbc2`) |
| **Notebooks** | `Exp_A/B/C/D_p45_*` (repo root, committed at `ee6e9ca`) |
| **Trained checkpoints (247 MB each)** | `HiDF-Results-Claude-Code/6-26-26 1200hrs/EXP{1,2,3,4}/eahn_hidf_best.pth` |
| **Full run archive (805 MB each)** | same folders / `eahn_hidf_complete.zip` |
| **Results + corrected reports** | `HiDF-Results-Claude-Code/6-26-26 1200hrs/` (incl. `P45_PUBLISHABLE_SUMMARY.md`, `report_CORRECTED.txt`) |

To restore code: `git checkout p45-baseline`. To restore a model: load the matching
`eahn_hidf_best.pth` (strict load → architecture flags must match that notebook).

---

## Shared configuration (all 4 P45 runs)

`EfficientNet-B4 → EarlyAttnHead M_t (7×7=49) → 4-layer transformer → temporal gate
M_frame → additive evidence head (AEH)`. Trained 8 epochs, batch 2 × grad-accum 8,
16 frames, focal loss (`alpha_pos 1.5`, `gamma 1.5`), `aeh_enabled, gamma_max 0.8,
lambda_aeh_aux 0.5`, **`aeh_freq_enabled` (SRM→e), `lambda_calib 1.0`**, moderate SBI
(`sbi_modes blend,warp,color, sbi_freq_mismatch 0.5, lambda_sbi_cls 1.0, sbi_stride 6`).
Authoritative full command = the committed notebook + each run's `*.log`.

**Per-experiment differentiator:**

| Exp | Notebook | Differentiator |
|---|---|---|
| EXP1 | `Exp_A_p45_anchor_g80` | none (anchor / control) |
| EXP2 | `Exp_B_p45_sigconc_mod_g80` | `--lambda_aeh_sigconc 0.15` |
| EXP3 | `Exp_C_p45_sigconc_strong_g80` | `--lambda_aeh_sigconc 0.40` |
| EXP4 | `Exp_D_p45_sigconc_multiband_g80` | `--lambda_aeh_sigconc 0.15 --aeh_freq_mode multiband` |

---

## Frozen results (6-26-26 1200hrs run)

### Detection — HiDF in-distribution (437 real / 392 fake)

| Metric | EXP1 | **EXP2 ⭐** | EXP3 | EXP4 |
|---|---|---|---|---|
| AUC-ROC | 0.959 | **0.972** | 0.969 | 0.954 |
| Real acc @0.5 | 0.975 | 0.954 | 0.982 | 0.952 |
| Fake acc @0.5 | 0.699 | **0.867** | 0.724 | 0.793 |
| Balanced @0.5 | 0.837 | **0.911** | 0.853 | 0.873 |
| Balanced @opt | 0.905 | 0.923 | 0.914 | 0.893 |
| Optimal threshold | 0.282 | 0.346 | 0.237 | 0.345 |

### Cross-dataset — real / fake acc @0.5 ; AUC

| Dataset | EXP1 | EXP2 | EXP3 | EXP4 |
|---|---|---|---|---|
| FF++ Deepfakes | .97/.14 ; .766 | .95/.46 ; **.820** | .93/.30 ; .760 | .85/.41 ; .714 |
| FF++ Face2Face | .97/.09 ; .548 | .94/.10 ; .555 | .93/.08 ; .506 | .85/.14 ; .499 |
| FF++ FaceShifter | .97/.08 ; .611 | .93/.15 ; .651 | .93/.09 ; .641 | .83/.17 ; .644 |
| FF++ FaceSwap | .98/.02 ; .447 | .94/.04 ; .457 | .95/.02 ; .373 | .85/.03 ; .391 |
| FF++ NeuralTextures | .98/.07 ; .653 | .96/.09 ; .670 | .94/.09 ; .630 | .84/.20 ; .607 |
| Celeb-DF v2 | .97/.05 ; .575 | .97/.09 ; .563 | .93/.04 ; .544 | .90/.11 ; .544 |

### Explanation — contribution space (analytic, all 392 fakes) — **the correct metric**

| Metric | EXP1 | EXP2 | EXP3 | EXP4 |
|---|---|---|---|---|
| Insertion fake (head) ↑ | 0.730 | **0.836** | 0.782 | 0.765 |
| Deletion fake (head) ↓ | 0.388 | 0.385 | **0.328** | 0.406 |
| Ins/Del gain (head) | +.156/+.182 | +.189/+.261 | +.197/+.252 | +.162/+.198 |
| Insertion fake (blended) ↑ | 0.720 | **0.832** | 0.758 | 0.740 |
| Deletion fake (blended) ↓ | 0.471 | 0.513 | **0.404** | 0.458 |
| **Faithful?** | ✅ | ✅ | ✅ | ✅ |

### Explanation — diagnostics

| Metric | EXP1 | EXP2 | EXP3 | EXP4 |
|---|---|---|---|---|
| Faithfulness corr | 0.321 | **0.433** | 0.366 | 0.369 |
| Temporal SSIM | 0.936 | 0.937 | 0.943 | 0.945 |
| Peak mode share | 0.139 | 0.195 | 0.176 | 0.163 |
| M_t std (uniform≈0.0204) | 0.0100 | 0.0168 | 0.0151 | 0.0113 |
| Inter-sample cosine ↓ | 0.561 | 0.645 | 0.573 | 0.677 |
| k-drop (zerofill) k1/k2/k4 | 1.18/1.23/1.18 | 2.28/2.06/1.78 | 0.22/0.98/1.77 | 0.42/0.84/1.03 |
| Pixel-occlusion ins/del *(confounded — do not headline)* | .41/.52 | .36/.47 | .37/.48 | .44/.58 |

---

## Per-experiment record: strengths → improvement targets

Tags: **[free]** = eval-only, no retrain · **[1-knob]** = config-only retrain · **[hard]** = modeling work.

### EXP2 ⭐ (winner, deployed model)
- **Strong:** best detection (AUC .972 / fake .867 / bal .911), best insertion (.832), best faith_corr (.433), best cross (Deepfakes .82).
- **Improve:** (1) **cross-dataset fake recall** — collapses off Deepfakes (Face2Face .10, FaceSwap .04, Celeb-DF .09) **[hard]**; (2) **deletion sharpness** — blended del .513 vs EXP3's .404, evidence could be tighter **[1-knob: sigconc 0.15→~0.25]**; (3) heatmaps must stay correct/faithful through any generalization aug **[verify]**.

### EXP3 (explanation/concentration showcase)
- **Strong:** sharpest faithful evidence (del .328/.404, highest gains), high real acc .982, AUC .969.
- **Improve:** (1) **fake acc @0.5 = .724** (real-biased, opt_thr .237) **[free: report @opt or fix checkpoint pick]**; (2) cross-dataset weakest of the sigconc runs (FaceSwap AUC .373) **[hard]**; (3) over-concentration cost insertion (.758 vs .832) **[1-knob: ease sigconc]**.

### EXP1 (anchor / control)
- **Strong:** clean baseline; proves the head is faithful **without** the lever (head ins .730); high real acc .975.
- **Improve:** (1) **worst fake acc @0.5 = .699** **[free: threshold]**; (2) weakest faith_corr (.321) + least concentrated (peak .139) — by design, no sigconc; (3) cross-dataset fake recall poor. Role = before/after reference; superseded by EXP2.

### EXP4 (multiband variant — droppable)
- **Strong:** 2nd-best fake acc @0.5 (.793), faithful (ins .765).
- **Improve:** (1) **lowest in-dist AUC (.954)** — multiband didn't help detection; (2) **worst cross-dataset AUC** + cross real acc dropped to .83–.90 (more false positives on real) **[hard]**; (3) insertion below EXP2. Recommendation: **drop** unless multiband earns its keep elsewhere.

**Shared hard problem (all four):** cross-dataset fake recall — the model is conservative
off-distribution (high real acc, low fake recall). This is the thesis's honest
limitation and the primary target of the next runs.

---

## Paper-assets status (so we never re-run just to get a figure)

### Already saved per run (in `analysis_essentials/` + root)
- Detection: ROC, PR, confusion, score-distribution (in-dist) ✓
- Cross-dataset: AUC bar chart + per-dataset score plots (21–26) + per-class JSON ✓
- Training: loss / val-accuracy / metric curves + `training_history.csv` + `logs.csv` ✓
- Explanation: heatmap overlays (intrinsic/gradcam/rollout) + heatmap strips ✓
- All metrics JSON incl. `contribution_space` block ✓
- Corrected reporting (this run, post-hoc): `report_CORRECTED.txt`, `P45_PUBLISHABLE_SUMMARY.md`, `P45_results_table.csv` ✓

### Shipped in P46 (every run from 6-26-26 2300hrs onward carries these)
1. ✅ **Contribution-space del/ins CURVE figure** — `contribution_del_ins_curve.png` (curve arrays now exported in the JSON too).
2. ✅ **ROAD curve figure** — `road_curve.png`.
3. ✅ **Corrected faithfulness CSV + authoritative report** — `faithfulness_summary.csv` + `faithfulness_report.txt` (contribution-space headline).

### Still nice-to-have (not blocking)
4. **Cross-experiment comparison figure** (the multi-model table as a plotted panel) — currently the local `P45_results_table.csv`; can be generated post-hoc from any set of result folders (no re-run needed).
5. (Nice-to-have) qualitative heatmap grid: real/fake × correct/incorrect, multiple samples (partially present in `heatmaps/`).

---

## Improvement roadmap (staged, one axis per run)

1. **EXP2 → generalization (`Exp_A_p46_generalize_g80`, Phase 46) — ✅ RAN 6-26-26 2300hrs.**
   `--generalize_aug` fired (verified: `RandomDownscale` active in training). **Partial win:**
   in-dist IMPROVED (AUC 0.972→**0.977**, fake@0.5 0.867→**0.901**, bal 0.911→**0.917**) and
   explanation IMPROVED + faithful (blended ins 0.832→**0.878**, del 0.513→**0.424**, head ins
   **0.902**/del **0.342**, faithful YES; m_t_std 0.0168→0.0255 = sharper). **BUT cross-dataset
   AUC DROPPED across the board** (Deepfakes 0.820→0.743, NeuralTextures 0.670→0.613, Celeb-DF
   0.563→0.534). Insight: the weakly-transferring cross signal is HIGH-FREQUENCY; downscale
   destroyed it → degradation is the WRONG lever for cross-dataset. Paper figures all generated.
   **P46 is the new best in-dist + explanation model** (candidate headline; promotion pending).
2. **Phase 47 — two parallel runs (built 6-26-26):**
   - **`Exp_A_p47_sbi_seams_g80`** = cross-dataset, DIFFERENT lever: EXP2 base (NO downscale) +
     stronger generic blend-seams (`sbi_cls 1.0→1.5, stride 6→4, freq_mismatch 0.5→0.7`) with
     calibration holding the operating point (the protection P43-sbi-primary lacked). Honest
     long shot. **Watch:** cross AUC/fake recall up (esp. blend-based FaceShifter/FaceSwap);
     in-dist holds; faithful YES.
   - **`Exp_B_p47_exp3_sharp_g80`** = EXP3 lineage (staged): strong `sigconc 0.4` + `--generalize_aug`
     (proven operating-point + explanation booster) = the definitive explanation-showcase model
     (sharpest deletion + fixed operating point). Cross-dataset is NOT its job.
3. **EXP1 → control**, re-scored honestly (threshold) — kept as the no-lever reference.
4. **Final run → best-of**, all fixes folded in, complete paper-ready bundle → write-up.
