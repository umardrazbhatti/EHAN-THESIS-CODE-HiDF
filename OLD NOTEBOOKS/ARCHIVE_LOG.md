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
| 2026-06-23 06:34 PST | `Exp_A_p39_additive_g60.ipynb` | P39.1 additive head, gamma_max **0.6** (low bracket of the gamma ablation) | `Exp_A_p40_topk_g80.ipynb` | Gamma ablation done (ran 6-22-26 0800hrs, results saved). P40 moves on to evidence **concentration** to fix the insertion gap. |
| 2026-06-23 06:34 PST | `Exp_B_p39_additive_g80.ipynb` | P39.1 additive head, gamma_max **0.8** (the P39 winner: det 0.982, ROAD intrinsic beat random+gradient) | `Exp_B_p40_concentrate_g80.ipynb`, `Exp_C_p40_consolidate_g80.ipynb` | g80 was the winning operating point; all three P40 notebooks build on it (gamma=0.8 base) to push insertion >=0.7. |
