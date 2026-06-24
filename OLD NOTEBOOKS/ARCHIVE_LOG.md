# OLD NOTEBOOKS - archive log

Superseded Kaggle launcher notebooks live here. **Convention:** whenever a new
phase's notebooks are created at the repo root, the previous phase's notebooks are
moved into this folder and a row is added below with the archive **date + time**,
so any past config can be recovered and compared.

- Newest archival at the **top**.
- The actual RESULTS for each run are saved separately under
  `C:\Users\Admin\Desktop\HiDF-Results-Claude-Code\<date>` (not here).
- History is also preserved in git (`git mv`), and every config is summarised in
  `CHANGELOG.md`.

| Archived (date / time) | Notebook | Phase / config | Superseded by | Why |
|---|---|---|---|---|
| 2026-06-24 10:30 PST | `Exp_A_p41_concentrate_strong_g80.ipynb` | P41 EXP-A: soft concentration STRONG (`--lambda_aeh_topk_mass 0.5 --lambda_aeh_temporal_conc 0.1 --lambda_aeh_suff 0.3`), gamma 0.8 | `Exp_A_p42_clean_rebalance_g80.ipynb` | RAN 6-24-26 1000hrs (EXP1) and **broke detection 0.982 -> 0.920** while deletion/insertion got WORSE (ins_fake 0.704 -> 0.595). The lever FIRED ([Phase41] banner, L_aeh_mass=0.466) but m_t_std stayed 0.003 (M_t uniform): the model games the mass loss through `e`, so the displayed map concentrates while the prediction stays diffuse -> map<->logit alignment (del/ins) degrades. Third independent proof that concentration is the wrong lever. P42 drops it. |
| 2026-06-24 10:30 PST | `Exp_B_p41_concentrate_clean_g80.ipynb` | P41 EXP-B: soft concentration CLEAN (`--lambda_aeh_topk_mass 0.3 --lambda_aeh_temporal_conc 0.05`), gamma 0.8 | `Exp_B_p42_evalonly_reframe.ipynb` | RAN 6-24-26 1000hrs (EXP2): detection held (0.974) and faithfulness hit a record 0.464, but insertion still fell (fake-only 0.704 -> 0.626) and ROAD random beat intrinsic. Confirms concentration trades insertion for faithfulness rather than winning both. P42-B reframes the eval (multi-baseline del/ins + ROAD headline) on the existing best checkpoint instead of retraining. |
| 2026-06-23 18:01 PST | `Exp_A_p40_topk_g80.ipynb` | P40 EXP-1: HARD top-k STE bottleneck on the additive logit (`--aeh_topk_frac 0.15`), gamma 0.8 | `Exp_A_p41_concentrate_strong_g80.ipynb` | RAN 6-23-26 1600hrs and **BACKFIRED**: the dense STE backward let the model equalise cells to game the mask -> evidence MORE diffuse (peak_mode_share 0.14->0.066), deletion 0.33->0.52, faithfulness 0.32->0.14. Hard bottleneck on the forward is the wrong tool; P41 uses pure-aux concentration instead. |
| 2026-06-23 18:01 PST | `Exp_B_p40_concentrate_g80.ipynb` | P40 EXP-2: entropy concentration + soft sufficiency (`--lambda_aeh_concentrate 0.05 --lambda_aeh_suff 0.3 --aeh_suff_topk_frac 0.15`), gamma 0.8 | `Exp_A/B_p41_concentrate_*_g80.ipynb` | The 6-23-26 1600hrs EXP2 run **never executed these flags** (launched the pre-P40 g80 command; metrics bit-identical to 6-22 g80). Concentration was never actually tested. P41 re-implements it as a stronger, correctly-wired pure-aux loss and verifies the flags reach the config + print in the log. |
| 2026-06-23 18:01 PST | `Exp_C_p40_consolidate_g80.ipynb` | P40 EXP-3: hard top-k + decomposition layered figure (`--aeh_topk_frac 0.15 --decomp_enabled`), gamma 0.8, 7 ep (never run) | (deferred) | Depended on EXP-1/EXP-2; carries the backfiring hard top-k. The layered "delete-a-layer" figure can be revived later on top of a working P41 concentration. |
| 2026-06-23 06:34 PST | `Exp_A_p39_additive_g60.ipynb` | P39.1 additive head, gamma_max **0.6** (low bracket of the gamma ablation) | `Exp_A_p40_topk_g80.ipynb` | Gamma ablation done (ran 6-22-26 0800hrs, results saved). P40 moves on to evidence **concentration** to fix the insertion gap. |
| 2026-06-23 06:34 PST | `Exp_B_p39_additive_g80.ipynb` | P39.1 additive head, gamma_max **0.8** (the P39 winner: det 0.982, ROAD intrinsic beat random+gradient) | `Exp_B_p40_concentrate_g80.ipynb`, `Exp_C_p40_consolidate_g80.ipynb` | g80 was the winning operating point; all three P40 notebooks build on it (gamma=0.8 base) to push insertion >=0.7. |
