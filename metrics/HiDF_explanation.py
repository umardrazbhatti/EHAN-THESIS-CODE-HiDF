"""
metrics/explanation.py — Explanation quality metrics.

FIX: faithfulness_correlation received M_t (subset, 49) and grad_maps (subset, T, 49)
     of mismatched shapes. Both are now averaged over time before reshaping, giving
     (subset, 49) for each, so Spearman correlation is well-defined.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
from typing import Dict


class ExplanationMetrics:

    @staticmethod
    def temporal_ssim(M_t_up: torch.Tensor) -> float:
        """
        Mean SSIM between consecutive explanation frames.
        M_t_up: (N, T, H, W) subset.
        """
        values = []
        N, T, H, W = M_t_up.shape
        for b in range(N):
            for t in range(T - 1):
                a = M_t_up[b, t].cpu().numpy().astype(np.float32)
                b_ = M_t_up[b, t + 1].cpu().numpy().astype(np.float32)
                val = ssim(a, b_, data_range=1.0)
                values.append(val)
        return float(np.mean(values)) if values else 1.0

    @staticmethod
    def faithfulness_correlation(
        M_flat: torch.Tensor,     # (subset, K) — intrinsic maps flattened
        grad_flat: torch.Tensor,  # (subset, K) — gradient maps flattened
    ) -> float:
        """
        Spearman rank correlation between intrinsic attention and gradient attribution.
        Both tensors must already be (subset, K) with the same K.
        """
        m = M_flat.detach().cpu().numpy().flatten()
        g = grad_flat.detach().cpu().numpy().flatten()
        if len(m) < 3 or np.std(m) < 1e-8 or np.std(g) < 1e-8:
            return 0.0
        corr, _ = spearmanr(m, g)
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def deletion_insertion_auc(model, frames, saliency,
                               steps: int = 10) -> dict:
        """
        Deletion/Insertion AUC: simplified implementation.
        Steps are coarse for speed; increase for publication-quality numbers.
        """
        device = next(model.parameters()).device
        B, T, C, H, W = frames.shape
        total_pixels  = H * W

        import torch.nn.functional as _F

        # Phase 20: switch to Gaussian-blur fill (matches baseline benchmark
        # protocol). Zero-fill in normalized space is technically ImageNet mean,
        # but blur preserves low-frequency content (face shape, lighting) while
        # removing high-frequency content (manipulation cues), giving a much
        # cleaner faithfulness signal.
        def _gauss_blur(x, k=21, sigma=10.0):
            # x: (B, T, C, H, W)
            Bx, Tx, Cx, Hx, Wx = x.shape
            flat = x.reshape(Bx * Tx, Cx, Hx, Wx)
            ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
            g = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
            g = g / g.sum()
            kx = g.view(1, 1, 1, k).expand(Cx, 1, 1, k)
            ky = g.view(1, 1, k, 1).expand(Cx, 1, k, 1)
            pad = k // 2
            blurred = _F.conv2d(flat, kx, padding=(0, pad), groups=Cx)
            blurred = _F.conv2d(blurred, ky, padding=(pad, 0), groups=Cx)
            return blurred.reshape(Bx, Tx, Cx, Hx, Wx)

        with torch.no_grad():
            baseline_logit = model(frames.to(device)).prob.mean().item()
            blurred_full = _gauss_blur(frames)            # (B, T, C, H, W)

        del_scores = []
        ins_scores = []

        # Use mean explanation over time
        sal = saliency.mean(1)   # (B, H, W)

        for step in range(steps + 1):
            frac = step / steps
            k    = max(1, int(frac * total_pixels))

            # Deletion: replace top-k pixels with blurred version
            # Insertion: start blurred, reveal top-k from original
            del_frames = frames.clone()
            ins_frames = blurred_full.clone()

            for b in range(B):
                flat_sal = sal[b].reshape(-1)
                top_k_idx = np.argsort(flat_sal)[-k:].copy()
                mask = np.zeros(H * W, dtype=bool)
                mask[top_k_idx] = True
                mask_2d = mask.reshape(H, W)

                del_frames[b, :, :, mask_2d] = blurred_full[b, :, :, mask_2d]
                ins_frames[b, :, :, mask_2d] = frames[b, :, :, mask_2d]

            with torch.no_grad():
                del_score = model(del_frames.to(device)).prob.mean().item()
                ins_score = model(ins_frames.to(device)).prob.mean().item()

            del_scores.append(del_score)
            ins_scores.append(ins_score)

        _trapz = getattr(np, "trapezoid", np.trapz)
        del_auc = float(_trapz(del_scores) / steps)
        ins_auc = float(_trapz(ins_scores) / steps)
        return {"deletion_auc": del_auc, "insertion_auc": ins_auc}

    @staticmethod
    def collapse_diagnostics(all_M_t: torch.Tensor) -> Dict[str, float]:
        """
        Compute three collapse diagnostic metrics on the full test-set M_t tensor.

        Parameters
        ----------
        all_M_t : (N, T, H, W)  — explanation maps for all test samples

        Returns
        -------
        dict with keys:
            inter_sample_cosine_mean  — mean pairwise cosine sim; < 0.5 healthy
            peak_mode_share           — fraction of samples whose argmax lands at
                                        the most common (row, col); < 0.2 healthy
            m_t_std_mean              — mean M_t std across samples; > 0.13 = one-hot
            m_t_std_max               — max  M_t std across samples
        """
        N, T, H, W = all_M_t.shape

        # --- inter-sample cosine similarity ---
        flat = all_M_t.mean(dim=1).reshape(N, H * W).float()   # (N, H*W) — time-averaged
        flat_norm = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = flat_norm @ flat_norm.T                    # (N, N)
        eye = torch.eye(N, dtype=torch.bool, device=all_M_t.device)
        n_pairs = N * (N - 1)
        inter_cosine = float(
            sim_matrix.masked_fill(eye, 0.0).sum().item() / max(n_pairs, 1)
        )

        # --- peak-coordinate mode share ---
        mean_maps = all_M_t.mean(dim=1)                         # (N, H, W)
        peak_indices = mean_maps.reshape(N, -1).argmax(dim=-1)  # (N,)
        peak_rc = [(int(idx) // W, int(idx) % W) for idx in peak_indices.tolist()]
        from collections import Counter
        most_common_count = Counter(peak_rc).most_common(1)[0][1]
        peak_mode_share = float(most_common_count) / N

        # --- M_t std (per-sample, time-and-space) ---
        stds = all_M_t.std(dim=(-1, -2)).mean(dim=-1)           # (N,) mean over T
        m_t_std_mean = float(stds.mean().item())
        m_t_std_max  = float(stds.max().item())

        return {
            "inter_sample_cosine_mean": inter_cosine,
            "peak_mode_share":          peak_mode_share,
            "m_t_std_mean":             m_t_std_mean,
            "m_t_std_max":              m_t_std_max,
        }

    @staticmethod
    def frame_attention_drop_test(
        model, loader, device, k_values=(1, 2, 4), seed: int = 42
    ) -> dict:
        """
        Intrinsic faithfulness test.

        For each video in the loader:
        1. Forward pass (eval, no_grad) → get M_t (B, T, h, w).
        2. Per-frame attention score = M_t[b, t, :, :].mean() over (h, w).
        3. Rank frames by score (descending).
        4. For each K in k_values:
           a. Zero out top-K frames (at normalised input level) → re-forward → record prob.
           b. Zero out K random frames (seeded) → re-forward → record prob.
        5. Aggregate: conf_drop = original_prob - masked_prob.

        Returns dict with keys:
            k{K}_top_conf_drop, k{K}_random_conf_drop, k{K}_ratio  for each K.
        A faithful explanation shows top_conf_drop >> random_conf_drop.
        """
        import numpy as np

        model.eval()
        rng = np.random.default_rng(seed)

        accum = {k: {"top": [], "rand": []} for k in k_values}

        _debug_printed = False   # print diagnostics once for the first batch

        with torch.no_grad():
            for batch in loader:
                frames = batch["frames"].to(device)          # (B, T, C, H, W)
                B, T, C, H, W = frames.shape

                out     = model(frames)
                orig_p  = out.prob.cpu()                      # (B,)
                M_t     = out.M_t.cpu()                       # (B, T, h, w)

                # Task 1.3 diagnostic: dump raw M_t statistics once (first batch only)
                if not _debug_printed:
                    print(f"[DIAG frame_attn_drop] M_t shape: {M_t.shape}")
                    print(f"[DIAG frame_attn_drop] M_t mean per frame:\n"
                          f"  {M_t.mean(dim=(-1,-2))}")
                    print(f"[DIAG frame_attn_drop] M_t max  per frame:\n"
                          f"  {M_t.amax(dim=(-1,-2))}")
                    print(f"[DIAG frame_attn_drop] M_t min  per frame:\n"
                          f"  {M_t.amin(dim=(-1,-2))}")
                    print(f"[DIAG frame_attn_drop] M_t std  per frame:\n"
                          f"  {M_t.std(dim=(-1,-2))}")
                    _debug_printed = True

                # Task 1.3 fix (cause c): use peak intensity (amax) for frame ranking
                # rather than mean — peak reflects the heatmap focus point, while
                # mean collapses to a near-constant when maps are spatially diffuse.
                frame_scores = M_t.amax(dim=(-1, -2))         # (B, T) — peak per frame

                for b in range(B):
                    scores_b  = frame_scores[b].numpy()              # (T,)
                    # .copy() prevents negative-stride error when indexing a tensor
                    ranked    = np.argsort(scores_b)[::-1].copy()    # desc, contiguous
                    orig_prob = float(orig_p[b])

                    for k in k_values:
                        k = min(k, T)

                        # — Top-K drop —
                        top_k_idx = ranked[:k]
                        f_top     = frames[b:b+1].clone()      # (1, T, C, H, W)
                        f_top[0, top_k_idx] = 0.0
                        drop_top  = orig_prob - float(model(f_top.to(device)).prob.cpu())

                        # — Random-K drop —
                        rand_k_idx = rng.choice(T, size=k, replace=False)
                        f_rand     = frames[b:b+1].clone()
                        f_rand[0, rand_k_idx] = 0.0
                        drop_rand  = orig_prob - float(model(f_rand.to(device)).prob.cpu())

                        accum[k]["top"].append(drop_top)
                        accum[k]["rand"].append(drop_rand)

        result = {}
        for k in k_values:
            tops   = accum[k]["top"]
            rands  = accum[k]["rand"]
            t_mean = float(np.mean(tops))  if tops  else 0.0
            r_mean = float(np.mean(rands)) if rands else 0.0
            ratio  = t_mean / (r_mean + 1e-8)
            result[f"k{k}_top_conf_drop"]    = t_mean
            result[f"k{k}_random_conf_drop"] = r_mean
            result[f"k{k}_ratio"]            = ratio
        return result

    @staticmethod
    def stability_check(model, loader, device, n_batches: int = 5) -> dict:
        """
        Determinism check: run forward pass twice on the same batches (model.eval(),
        no dropout). Compute mean cosine similarity between the two M_t maps per video.

        Returns:
            {"stability_cosine_mean": float, "stability_cosine_min": float}

        Expected ~1.0 for a deterministic intrinsic attention mechanism.
        Low values indicate stochasticity (dropout not disabled, or non-deterministic ops).
        """
        import numpy as np
        import torch.nn.functional as F

        model.eval()
        cos_sims = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                frames = batch["frames"].to(device)
                B      = frames.shape[0]

                out1 = model(frames)
                out2 = model(frames)

                M1 = out1.M_t.cpu().reshape(B, -1)  # (B, T*h*w)
                M2 = out2.M_t.cpu().reshape(B, -1)

                M1 = F.normalize(M1, dim=-1)
                M2 = F.normalize(M2, dim=-1)

                cos = (M1 * M2).sum(dim=-1).tolist()   # (B,)
                cos_sims.extend(cos)

        if not cos_sims:
            return {"stability_cosine_mean": 1.0, "stability_cosine_min": 1.0}

        return {
            "stability_cosine_mean": float(np.mean(cos_sims)),
            "stability_cosine_min":  float(np.min(cos_sims)),
        }
