"""
losses/temporal.py — Gated Temporal Consistency loss L_temp.

w_t = exp(-γ · ||φ(f_t) − φ(f_{t+1})||₂)   where φ is L2-normalised
L_temp = Σ_t w_t · ||M_t − M_{t+1}||²_F  (mean over batch and pairs)

Key fix (Task 3.1 Cause A resolved): low_level features are L2-normalised to
unit sphere before computing pairwise distances, bounding dist ∈ [0, 2] and
making γ=0.1 meaningful:
  diff=0.5 → w=exp(-0.05)≈0.95  (similar frames weighted high)
  diff=2.0 → w=exp(-0.20)≈0.82  (very different frames still contribute)
At the old γ=10.0 these gates would be exp(-5) and exp(-20) → ≈ 0.

Task 3.1 additions:
  - First-batch diagnostic prints (dumped once per training run).
  - Training-start gate assertion: FAIL FAST if mean(w_t) < 0.01 or > 0.99
    (indicates degenerate gate — wrong γ or features not L2-normalised).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self._diag_printed = False   # print diagnostics once per training run

    def forward(
        self,
        M_t:       torch.Tensor,   # (B, T, h, w)
        low_level: torch.Tensor,   # (B, T, C_low, Hl, Wl)
    ) -> torch.Tensor:
        B, T = M_t.shape[:2]
        if T < 2:
            return torch.tensor(0.0, device=M_t.device)

        # Detach, flatten, and L2-normalise to unit sphere
        phi = low_level.detach().reshape(B, T, -1)   # (B, T, C·H'·W')
        phi = F.normalize(phi, p=2, dim=-1)           # unit vectors on d-sphere

        total_loss = torch.tensor(0.0, device=M_t.device)
        n_pairs    = 0

        # Collect gate stats for diagnostics
        all_diff_norms = []
        all_w_t        = []
        all_map_diffs  = []

        for t in range(T - 1):
            # L2 distance between consecutive normalised feature vectors; in [0, 2]
            diff_norm = (phi[:, t] - phi[:, t + 1]).norm(dim=-1)   # (B,)

            # Gate: upweights pairs where consecutive frames look similar
            w_t = torch.exp(-self.gamma * diff_norm)               # (B,)

            # Penalise explanation map change weighted by frame similarity
            map_diff = (M_t[:, t] - M_t[:, t + 1]).pow(2).mean(dim=(-1, -2))  # (B,)

            total_loss = total_loss + (w_t * map_diff).mean()
            n_pairs   += 1

            all_diff_norms.append(diff_norm.detach())
            all_w_t.append(w_t.detach())
            all_map_diffs.append(map_diff.detach())

        # ── Task 3.1: first-batch diagnostics ────────────────────────────────
        if not self._diag_printed and all_w_t:
            dn_cat = torch.cat(all_diff_norms)   # (n_pairs * B,)
            wt_cat = torch.cat(all_w_t)
            md_cat = torch.cat(all_map_diffs)
            print(
                f"[L_temp DIAG] γ={self.gamma}  "
                f"||φ_t - φ_t+1||: mean={dn_cat.mean():.4f} std={dn_cat.std():.4f}  "
                f"w_t: mean={wt_cat.mean():.4f} std={wt_cat.std():.4f}  "
                f"||M_t - M_t+1||²: mean={md_cat.mean():.6f} std={md_cat.std():.6f}"
            )
            # Task 3.1: FAIL FAST if gate is degenerate
            wt_mean = float(wt_cat.mean())
            if wt_mean < 0.01:
                raise RuntimeError(
                    f"[L_temp] DEGENERATE GATE: mean(w_t)={wt_mean:.4f} < 0.01. "
                    f"γ={self.gamma} is too large — exp(-γ·dist) saturates to 0. "
                    f"Reduce γ to ≤ 1.0 or verify L2-normalisation of low_level features."
                )
            if wt_mean > 0.99:
                raise RuntimeError(
                    f"[L_temp] DEGENERATE GATE: mean(w_t)={wt_mean:.4f} > 0.99. "
                    f"γ={self.gamma} is too small — gate is always 1 and provides no "
                    f"frame-pair discrimination.  Increase γ or check feature quality."
                )
            print(f"[L_temp] Gate sanity PASSED (mean_w_t={wt_mean:.4f} ∈ [0.01, 0.99])")
            self._diag_printed = True

        return total_loss / n_pairs
