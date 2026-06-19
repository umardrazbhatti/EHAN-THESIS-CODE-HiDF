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
        Mean PER-SAMPLE Spearman rank correlation between intrinsic attention
        and gradient attribution.

        Phase 30 fix: the previous implementation flattened all samples into
        one long vector before correlating.  M_t maps are softmax-normalised
        (same scale every sample) but raw |gradient| magnitudes differ by
        orders of magnitude between samples, so the pooled ranking was
        dominated by BETWEEN-sample gradient scale — a quantity that says
        nothing about whether the map ranks cells correctly WITHIN a video.
        The standard formulation (Quantus / saliency literature) is rank
        correlation per explanation, averaged over the evaluation set; rank
        correlation is scale-invariant per sample, so no normalisation is
        needed.  Samples with degenerate (constant) maps are skipped.
        """
        m_all = M_flat.detach().cpu().numpy()
        g_all = grad_flat.detach().cpu().numpy()
        corrs = []
        for i in range(m_all.shape[0]):
            m, g = m_all[i], g_all[i]
            if len(m) < 3 or np.std(m) < 1e-12 or np.std(g) < 1e-12:
                continue
            corr, _ = spearmanr(m, g)
            if not np.isnan(corr):
                corrs.append(float(corr))
        return float(np.mean(corrs)) if corrs else 0.0

    @staticmethod
    def deletion_insertion_auc(model, frames, saliency,
                               steps: int = 10,
                               n_samples: int = 100,
                               labels=None,
                               chunk: int = 4,
                               random_control: bool = True,
                               seed: int = 123,
                               verbose: bool = True,
                               saliency_ins=None,
                               baseline: str = "blur") -> dict:
        # Phase 35 (dual-lens / baseline):
        #   saliency_ins : separate saliency for the INSERTION ordering (the
        #     sufficiency lens M_suff); deletion keeps using `saliency` (the
        #     necessity lens M_nec).  None = use `saliency` for both (single-lens,
        #     exact P34 behaviour).
        #   baseline : fill for the del/ins canvas -- "blur" (P34 headline; floors
        #     insertion AUC at ~blurred_conf), "mean" (ImageNet mean = 0 in
        #     normalised space), or "black".  Alternate baselines give a cleaner
        #     sufficiency readout.  Eval-only, zero training risk.
        """
        Deletion/Insertion AUC with random-saliency control (Phase 28).

        n_samples      : maximum number of video clips to process.
        labels         : optional (B,) array of class labels; when provided,
                         fake-only aggregates are added (*_fake_only keys).
        chunk          : forward-pass micro-batch — allows n_samples >> 5
                         without OOM (Phase ≤27 forwarded all clips at once).
        random_control : also run the identical procedure with a per-sample
                         RANDOM pixel ranking and report the gain of the real
                         saliency over it.  Phase 27 evidence made this
                         mandatory: the model scores a fully-blurred clip at
                         0.82 fake vs 0.47 baseline, so the absolute del/ins
                         curves are dominated by the blur-direction artifact
                         and deletion > insertion is guaranteed REGARDLESS of
                         map quality.  The random control cancels the
                         artifact: it affects both orderings equally, so the
                         difference isolates the ordering quality of the map.

        Returns
        -------
        dict with keys:
          deletion_auc, insertion_auc      — AUC of confidence-vs-fraction curves
          del_at_{0,10,25,50,100}pct       — deletion confidence at those levels
          ins_at_{0,10,25,50,100}pct       — insertion confidence at those levels
          faithfulness_ok                  — True when insertion_auc > deletion_auc
          deletion_auc_random, insertion_auc_random   (when random_control)
          ins_gain_over_random             — insertion_auc - insertion_auc_random
                                             (> 0 ⇒ map beats random ordering)
          del_gain_over_random             — deletion_auc_random - deletion_auc
                                             (> 0 ⇒ map beats random ordering)
          deletion_auc_fake_only, insertion_auc_fake_only  (when labels given)
        """
        device = next(model.parameters()).device
        B_full, T, C, H, W = frames.shape
        B = min(n_samples, B_full)
        if verbose:
            print(f"[del_ins] Running on {B} samples, steps={steps}, "
                  f"chunk={chunk}, random_control={random_control}")
        frames   = frames[:B]
        saliency = np.asarray(saliency)[:B]
        if labels is not None:
            labels = np.asarray(labels)[:B]
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
            # FIX: .expand() returns a non-contiguous view; F.conv2d (grouped)
            # requires contiguous weight tensors on some PyTorch versions.
            kx = g.view(1, 1, 1, k).expand(Cx, 1, 1, k).contiguous()
            ky = g.view(1, 1, k, 1).expand(Cx, 1, k, 1).contiguous()
            pad = k // 2
            blurred = _F.conv2d(flat, kx, padding=(0, pad), groups=Cx)
            blurred = _F.conv2d(blurred, ky, padding=(pad, 0), groups=Cx)
            return blurred.reshape(Bx, Tx, Cx, Hx, Wx)

        def _fwd_probs(x):
            """Chunked no-grad forward → (B,) numpy of fake-probs."""
            outs = []
            with torch.no_grad():
                for i in range(0, x.shape[0], chunk):
                    outs.append(model(x[i:i+chunk].to(device)).prob.detach().cpu())
            return torch.cat(outs).numpy()

        with torch.no_grad():
            baseline_probs = _fwd_probs(frames)
            # Phase 35: choose the del/ins canvas.  "blur" preserves low-freq
            # content (P34 headline); "mean"/"black" are flat baselines that lift
            # the absolute insertion number the blur floor caps.
            if baseline == "mean":
                blurred_full = torch.zeros_like(frames)     # ImageNet mean = 0 (normalised)
            elif baseline == "black":
                _mn = torch.tensor((0.485, 0.456, 0.406),
                                   device=frames.device, dtype=frames.dtype).view(1, 1, 3, 1, 1)
                _sd = torch.tensor((0.229, 0.224, 0.225),
                                   device=frames.device, dtype=frames.dtype).view(1, 1, 3, 1, 1)
                blurred_full = (torch.zeros_like(frames) - _mn) / _sd   # black in normalised space
            else:
                blurred_full = _gauss_blur(frames)          # (B, T, C, H, W)
            blurred_probs  = _fwd_probs(blurred_full)
        baseline_conf = float(baseline_probs.mean())
        blurred_conf  = float(blurred_probs.mean())

        # Use mean explanation over time; per-sample descending pixel ranking
        sal       = saliency.mean(1).reshape(B, -1)             # (B, H*W)
        order_sal = np.argsort(sal, axis=1)[:, ::-1]            # (B, H*W) desc
        # Phase 35: separate INSERTION ordering from the sufficiency lens (M_suff).
        # None -> use the deletion ordering for both (single-lens, exact P34).
        if saliency_ins is not None:
            sal_i     = np.asarray(saliency_ins)[:B].mean(1).reshape(B, -1)
            order_ins = np.argsort(sal_i, axis=1)[:, ::-1]
        else:
            order_ins = order_sal
        _rng       = np.random.default_rng(seed)
        order_rand = np.stack([_rng.permutation(total_pixels) for _ in range(B)])

        # Checkpoint steps for per-percentage reporting (0%, 10%, 25%, 50%, 100%)
        _pct_to_step = {
            0:   0,
            10:  max(0, round(0.10 * steps)),
            25:  max(0, round(0.25 * steps)),
            50:  max(0, round(0.50 * steps)),
            100: steps,
        }

        def _curves(order, do_del=True, do_ins=True):
            """Run the del/ins sweep for a given (B, H*W) pixel ordering.
            Returns (steps+1, B) per-sample prob matrices.  Phase 35: do_del/
            do_ins let the caller compute deletion from the necessity ordering and
            insertion from the sufficiency ordering at the SAME total forward cost
            as one combined sweep."""
            del_mat = np.zeros((steps + 1, B), dtype=np.float64)
            ins_mat = np.zeros((steps + 1, B), dtype=np.float64)
            for step in range(steps + 1):
                frac = step / steps
                k    = max(1, int(frac * total_pixels))

                # Deletion: replace top-k pixels with the baseline
                # Insertion: start from the baseline, reveal top-k from original
                if do_del:
                    del_frames = frames.clone()
                if do_ins:
                    ins_frames = blurred_full.clone()

                for b in range(B):
                    top_k_idx = order[b, :k].copy()
                    mask = np.zeros(total_pixels, dtype=bool)
                    mask[top_k_idx] = True
                    mask_2d = mask.reshape(H, W)

                    if do_del:
                        del_frames[b, :, :, mask_2d] = blurred_full[b, :, :, mask_2d]
                    if do_ins:
                        ins_frames[b, :, :, mask_2d] = frames[b, :, :, mask_2d]

                if do_del:
                    del_mat[step] = _fwd_probs(del_frames)
                if do_ins:
                    ins_mat[step] = _fwd_probs(ins_frames)
            return del_mat, ins_mat

        # Phase 35: when an insertion saliency is supplied, measure deletion on the
        # necessity ordering and insertion on the sufficiency ordering (same cost).
        if saliency_ins is None:
            del_mat, ins_mat = _curves(order_sal)
        else:
            del_mat, _ = _curves(order_sal, do_ins=False)
            _, ins_mat = _curves(order_ins, do_del=False)

        # NumPy 2.x removed np.trapz (renamed to np.trapezoid).
        # getattr(np, "trapezoid", np.trapz) looks safe but Python evaluates
        # ALL THREE arguments before calling getattr, so np.trapz raises
        # AttributeError on NumPy 2.x even when "trapezoid" would have been found.
        # try/except is the only safe way to handle this.
        try:
            _trapz = np.trapezoid   # NumPy ≥ 2.0
        except AttributeError:
            _trapz = np.trapz       # NumPy < 2.0

        del_auc_per = _trapz(del_mat, axis=0) / steps           # (B,)
        ins_auc_per = _trapz(ins_mat, axis=0) / steps           # (B,)
        del_auc = float(del_auc_per.mean())
        ins_auc = float(ins_auc_per.mean())
        del_scores = del_mat.mean(axis=1)                        # mean curve for table
        ins_scores = ins_mat.mean(axis=1)

        # Assemble result dict — scalar values for CSV; per-checkpoint for reporting
        result = {
            "deletion_auc":    del_auc,
            "insertion_auc":   ins_auc,
            "faithfulness_ok": ins_auc > del_auc,
        }
        for pct, sidx in _pct_to_step.items():
            if 0 <= sidx < len(del_scores):
                result[f"del_at_{pct}pct"] = float(del_scores[sidx])
                result[f"ins_at_{pct}pct"] = float(ins_scores[sidx])

        # ── Phase 28: random-saliency control ─────────────────────────────────
        if random_control:
            del_mat_r, ins_mat_r = _curves(order_rand)
            del_auc_r = float((_trapz(del_mat_r, axis=0) / steps).mean())
            ins_auc_r = float((_trapz(ins_mat_r, axis=0) / steps).mean())
            result["deletion_auc_random"]   = del_auc_r
            result["insertion_auc_random"]  = ins_auc_r
            result["ins_gain_over_random"]  = ins_auc - ins_auc_r
            result["del_gain_over_random"]  = del_auc_r - del_auc

        # ── Phase 28: fake-only aggregates (artifact localisation is only
        # well-defined on manipulated samples) ────────────────────────────────
        if labels is not None and (labels == 1).any():
            _fk = labels == 1
            result["deletion_auc_fake_only"]  = float(del_auc_per[_fk].mean())
            result["insertion_auc_fake_only"] = float(ins_auc_per[_fk].mean())

        # ── Print formatted curve table ──────────────────────────────────────
        if verbose:
            print(f"\n  [Del/Ins] baseline_conf={baseline_conf:.4f}  "
                  f"blurred_conf={blurred_conf:.4f}  n_clips={B}  steps={steps}")
            print(f"  {'%removed':>8}  {'del_conf':>9}  {'ins_conf':>9}  "
                  f"{'del_drop':>10}  {'ins_gain':>10}")
            for pct in [0, 10, 25, 50, 100]:
                dk = f"del_at_{pct}pct"
                ik = f"ins_at_{pct}pct"
                if dk in result and ik in result:
                    drop = baseline_conf - result[dk]
                    gain = result[ik] - blurred_conf
                    print(f"  {pct:>7}%  {result[dk]:>9.4f}  {result[ik]:>9.4f}  "
                          f"{drop:>+10.4f}  {gain:>+10.4f}")
            faithful_tag = "✓ faithful" if ins_auc > del_auc else "✗ NOT faithful"
            print(f"  AUC → deletion={del_auc:.4f}  insertion={ins_auc:.4f}  [{faithful_tag}]")
            if random_control:
                _ig = result["ins_gain_over_random"]
                _dg = result["del_gain_over_random"]
                _ctl_tag = ("✓ beats random" if (_ig > 0 and _dg > 0)
                            else "✗ does NOT beat random")
                print(f"  Random control → del_rand={del_auc_r:.4f}  "
                      f"ins_rand={ins_auc_r:.4f}  "
                      f"ins_gain={_ig:+.4f}  del_gain={_dg:+.4f}  [{_ctl_tag}]")
                print("  (gains cancel the blur-direction artifact: a map with "
                      "real ordering information shows BOTH gains > 0 even when "
                      "the absolute curves are inverted by the artifact)")

        return result

    @staticmethod
    def layer_ablation_causal(model, frames, labels=None, n_samples: int = 50,
                              chunk: int = 4, verbose: bool = True) -> dict:
        """Phase 36 causal self-consistency check for the intrinsic decomposition.

        The model DECLARES each evidence layer's contribution as a % via
        layer_weights (e.g. "Layer 4 = 48%").  This test removes one layer at a
        time -- model(frames, ablate_layer=k) zeroes that layer's weight and
        renormalises the rest so the mixture stays convex -- and measures the
        resulting fake-confidence drop.  A FAITHFUL decomposition shows the
        declared %-share PREDICTS the drop: removing the layer the model says is
        48% should cost far more confidence than removing a 5% layer.  This is
        the intrinsic-safe validation (it probes the model's OWN declared parts,
        not an external saliency map).

        Returns per-layer declared share + mean drop, the across-sample rank
        correlation between share and drop, and the rate at which the
        largest-share layer is also the largest-drop layer (chance = 1/L).
        Returns {} when the model has no decomposition (decomp_enabled False).
        """
        if not getattr(model, "decomp_enabled", False):
            return {}
        L = int(getattr(model, "decomp_layers", 0))
        if L < 2:
            return {}
        device = next(model.parameters()).device
        B = min(n_samples, frames.shape[0])
        frames = frames[:B]
        if labels is not None:
            labels = np.asarray(labels)[:B]

        # Full forward: fake-prob + declared per-layer share (mean over time).
        prob_full_chunks, share_chunks = [], []
        with torch.no_grad():
            for i in range(0, B, chunk):
                o = model(frames[i:i + chunk].to(device))
                prob_full_chunks.append(o.prob.detach().cpu())
                share_chunks.append(o.layer_weights.mean(dim=1).detach().cpu())
        prob_full = torch.cat(prob_full_chunks).numpy()          # (B,)
        share     = torch.cat(share_chunks).numpy()              # (B, L)

        # Per-layer ablation: drop = prob_full - prob_with_layer_k_removed.
        drop = np.zeros((B, L), dtype=np.float64)
        for k in range(L):
            pk_chunks = []
            with torch.no_grad():
                for i in range(0, B, chunk):
                    pk_chunks.append(
                        model(frames[i:i + chunk].to(device),
                              ablate_layer=k).prob.detach().cpu())
            drop[:, k] = prob_full - torch.cat(pk_chunks).numpy()

        # Artifact attribution is only well-defined on fakes.
        if labels is not None and (labels == 1).any():
            sel = labels == 1
        else:
            sel = np.ones(B, dtype=bool)
        share_s, drop_s = share[sel], drop[sel]

        share_mean = share_s.mean(axis=0)                        # (L,)
        drop_mean  = drop_s.mean(axis=0)                         # (L,)
        fs, fd = share_s.flatten(), drop_s.flatten()
        if np.std(fs) > 1e-9 and np.std(fd) > 1e-9:
            corr, _ = spearmanr(fs, fd)
            corr = float(corr) if not np.isnan(corr) else 0.0
        else:
            corr = 0.0
        top_match = float(np.mean(share_s.argmax(axis=1) == drop_s.argmax(axis=1)))

        result = {
            "n_samples":              int(sel.sum()),
            "n_layers":               L,
            "share_per_layer":        [float(x) for x in share_mean],
            "drop_per_layer":         [float(x) for x in drop_mean],
            "share_vs_drop_spearman": corr,
            "top_layer_match_rate":   top_match,
            "top_layer_match_chance": 1.0 / L,
        }
        if verbose:
            print(f"\n  [Layer-ablation causal check] N={int(sel.sum())} fakes, L={L}")
            for k in range(L):
                print(f"    Layer {k + 1}: declared={share_mean[k] * 100:5.1f}%   "
                      f"conf_drop_when_removed={drop_mean[k]:+.4f}")
            print(f"    share->drop rank corr = {corr:+.3f}  "
                  f"(want > 0: bigger declared share => bigger drop)")
            print(f"    top-share == top-drop layer: {top_match:.2f}  "
                  f"(chance = {1.0 / L:.2f})")
        return result

    @staticmethod
    def road_faithfulness(model, frames, orderings, labels=None, steps: int = 20,
                          chunk: int = 4, seed: int = 123, noise_scale: float = 1.0,
                          verbose: bool = True) -> dict:
        """ROAD-style debiased MoRF deletion faithfulness (Rong et al., ICML 2022).

        WHY THIS EXISTS: the standard deletion/insertion fill (blur / zero / mean)
        pushes the clip OFF the data manifold.  For this detector a fully-blurred
        clip reads as MORE fake (measured blurred_conf ~0.82 >> the real
        baseline), so the blur deletion curve is dominated by that artifact, NOT
        by evidence removal -- it comes out flat or non-monotonic no matter how
        good the map is.  ROAD removes pixels with a NOISY imputation instead: the
        additive noise restores the high-frequency content whose ABSENCE the model
        was reading as 'blurry => fake', so the perturbed clip stays near the data
        manifold and the curve becomes MONOTONIC -- removing more evidence lowers
        fake-confidence proportionally (the 'relative' behaviour we want to read
        out).  MoRF = Most-Relevant-First: the highest-saliency pixels are removed
        first, so a faithful ordering produces a STEEPER, LOWER-AUC curve than a
        random ordering.

        frames    : (B, T, C, H, W) clips (already the eval subset).
        orderings : dict {name: saliency or None}.  saliency is (B,T,H,W) or
                    (B,H,W) array/tensor; None = per-sample RANDOM ordering (the
                    floor every real ordering must beat).
        labels    : optional (B,) 0/1; when given, a fake-only curve is added.

        Returns dict:
          baseline_conf, fully_removed_conf{name}  (sanity: fully-removed should
            fall toward 'real' -- if it RISES the fill is still read as fake),
          per ordering name -> {"auc","curve","drop_at":{pct:drop},
                                "auc_fake","drop_at_fake":{pct:drop}}.
        Lower AUC = more faithful.  gain = auc(random) - auc(name) > 0 => beats
        random.
        """
        device = next(model.parameters()).device
        B, T, C, H, W = frames.shape
        frames = frames.to(device)
        total = H * W
        if labels is not None:
            labels = np.asarray(labels)[:B]
            fake_sel = labels == 1
        else:
            fake_sel = np.zeros(B, dtype=bool)

        def _prob1(x):
            with torch.no_grad():
                return float(model(x).prob.detach().cpu().reshape(-1)[0])

        baseline_probs = np.zeros(B, dtype=np.float64)
        with torch.no_grad():
            for i in range(0, B, chunk):
                baseline_probs[i:i + chunk] = (
                    model(frames[i:i + chunk]).prob.detach().cpu().numpy())
        baseline_conf = float(baseline_probs.mean())

        # Noisy imputation canvas: per-sample, per-channel mean (over T,H,W) plus
        # Gaussian noise at the per-channel std.  The noise restores the
        # high-frequency content that a flat/blur fill removes (the thing the
        # detector mis-reads as 'fake'), keeping the perturbed clip near-manifold.
        mean_c = frames.mean(dim=(1, 3, 4), keepdim=True)          # (B,1,C,1,1)
        std_c  = frames.std(dim=(1, 3, 4), keepdim=True)           # (B,1,C,1,1)
        _g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(frames.shape, generator=_g).to(device)
        fill_full = mean_c + noise * std_c * float(noise_scale)    # (B,T,C,H,W)

        _rng = np.random.default_rng(seed)
        try:
            _trapz = np.trapezoid
        except AttributeError:
            _trapz = np.trapz

        def _order_from(sal):
            s = np.asarray(sal)
            if s.ndim == 4:            # (B,T,H,W) -> time-average
                s = s.mean(1)
            s = s.reshape(B, -1)
            return np.argsort(s, axis=1)[:, ::-1]                  # descending

        pct_steps = {p: int(round(p / 100.0 * steps)) for p in (0, 10, 25, 50, 75, 100)}
        result = {"baseline_conf": baseline_conf, "fully_removed_conf": {}}

        for name, sal in orderings.items():
            if sal is None:
                order = np.stack([_rng.permutation(total) for _ in range(B)])
            else:
                order = _order_from(sal)
            curve = np.zeros((steps + 1, B), dtype=np.float64)     # per-sample
            for b in range(B):
                fb   = frames[b:b + 1]
                fillb = fill_full[b:b + 1]
                for step in range(steps + 1):
                    k = int((step / steps) * total)
                    if k <= 0:
                        curve[step, b] = baseline_probs[b]
                        continue
                    db = fb.clone()
                    idx = order[b, :k].copy()
                    mask = np.zeros(total, dtype=bool)
                    mask[idx] = True
                    m2 = mask.reshape(H, W)
                    db[0, :, :, m2] = fillb[0, :, :, m2]
                    curve[step, b] = _prob1(db)
                    del db
            mean_curve = curve.mean(axis=1)
            auc = float(_trapz(mean_curve, dx=1.0 / steps))
            drop_at = {p: float(baseline_conf - mean_curve[s])
                       for p, s in pct_steps.items() if s < len(mean_curve)}
            entry = {"auc": auc,
                     "curve": [float(x) for x in mean_curve],
                     "drop_at": drop_at}
            if fake_sel.any():
                fcurve = curve[:, fake_sel].mean(axis=1)
                entry["auc_fake"] = float(_trapz(fcurve, dx=1.0 / steps))
                fbase = float(baseline_probs[fake_sel].mean())
                entry["drop_at_fake"] = {p: float(fbase - fcurve[s])
                                         for p, s in pct_steps.items()
                                         if s < len(fcurve)}
            result[name] = entry
            result["fully_removed_conf"][name] = float(mean_curve[-1])

        if verbose:
            rand_auc = result.get("random", {}).get("auc", None)
            print(f"\n  [ROAD debiased deletion] N={B} clips, steps={steps}, "
                  f"baseline_conf={baseline_conf:.4f}")
            print(f"  {'ordering':>10}  {'AUC':>7}  {'gain_vs_rand':>12}  "
                  f"{'drop@10%':>9}  {'drop@50%':>9}  {'fully_rmvd':>10}")
            for name in orderings:
                e = result[name]
                gain = (rand_auc - e["auc"]) if rand_auc is not None else 0.0
                d10 = e["drop_at"].get(10, 0.0)
                d50 = e["drop_at"].get(50, 0.0)
                fr  = result["fully_removed_conf"][name]
                print(f"  {name:>10}  {e['auc']:>7.4f}  {gain:>+12.4f}  "
                      f"{d10:>+9.4f}  {d50:>+9.4f}  {fr:>10.4f}")
            print("  (lower AUC = more faithful; gain>0 = beats random; if "
                  "fully_rmvd does NOT fall below baseline the fill is still "
                  "read as fake -- raise noise_scale)")
            print("  (a MONOTONIC drop@10% < drop@50% is the 'relative' "
                  "behaviour: removing more evidence costs more confidence)")
        return result

    @staticmethod
    def layer_ablation_cumulative(model, frames, labels=None, n_samples: int = 50,
                                  chunk: int = 4, verbose: bool = True) -> dict:
        """Cumulative version of the layer-ablation causal check (Phase 38).

        layer_ablation_causal removes ONE declared layer at a time.  This removes
        them CUMULATIVELY in decreasing declared-share order -- top-1, then
        top-1+2, then top-1+2+3 -- and records the running fake-confidence.  A
        faithful decomposition shows a MONOTONIC cumulative drop: 'remove the
        biggest reason -> confidence drops; remove the next -> drops more', which
        is exactly the layered/relative readout requested ('delete 1 layer ->
        drop, delete 2 -> more, delete 3 -> more').  Intrinsic-safe (probes the
        model's OWN declared parts -- no pixel perturbation, so no OOD confound).
        Returns {} when the model has no decomposition.
        """
        if not getattr(model, "decomp_enabled", False):
            return {}
        L = int(getattr(model, "decomp_layers", 0))
        if L < 2:
            return {}
        device = next(model.parameters()).device
        B = min(n_samples, frames.shape[0])
        frames = frames[:B].to(device)
        if labels is not None:
            labels = np.asarray(labels)[:B]

        prob_full_c, share_c = [], []
        with torch.no_grad():
            for i in range(0, B, chunk):
                o = model(frames[i:i + chunk])
                prob_full_c.append(o.prob.detach().cpu())
                share_c.append(o.layer_weights.mean(dim=1).detach().cpu())
        prob_full = torch.cat(prob_full_c).numpy()                # (B,)
        share     = torch.cat(share_c).numpy()                    # (B, L)

        if labels is not None and (labels == 1).any():
            sel = labels == 1
        else:
            sel = np.ones(B, dtype=bool)
        mean_share = share[sel].mean(axis=0)                      # (L,)
        order = np.argsort(mean_share)[::-1]                      # biggest share first

        cum_conf  = [float(prob_full[sel].mean())]
        cum_share = [0.0]
        for j in range(1, L):                                     # cannot remove all L
            abl = [int(x) for x in order[:j]]
            pj = []
            with torch.no_grad():
                for i in range(0, B, chunk):
                    pj.append(model(frames[i:i + chunk],
                                    ablate_layer=abl).prob.detach().cpu())
            pj = torch.cat(pj).numpy()
            cum_conf.append(float(pj[sel].mean()))
            cum_share.append(float(mean_share[order[:j]].sum()))
        drops = [cum_conf[0] - c for c in cum_conf]
        monotonic = all(drops[i + 1] >= drops[i] - 1e-6 for i in range(len(drops) - 1))

        result = {
            "n_samples":               int(sel.sum()),
            "n_layers":                L,
            "layer_order_by_share":    [int(x) for x in order],
            "cumulative_removed_share": cum_share,
            "cumulative_conf":         cum_conf,
            "cumulative_drop":         drops,
            "monotonic":               bool(monotonic),
        }
        if verbose:
            print(f"\n  [Layer-ablation CUMULATIVE] N={int(sel.sum())} fakes, L={L}, "
                  f"removal order (by declared share) = "
                  f"{[int(x) + 1 for x in order]}")
            for j in range(L):
                tag = "full map" if j == 0 else f"top-{j} layers removed"
                print(f"    {tag:>22}: conf={cum_conf[j]:.4f}  "
                      f"cum_drop={drops[j]:+.4f}  "
                      f"(declared share removed={cum_share[j] * 100:4.1f}%)")
            print(f"    monotonic cumulative drop: {monotonic}  "
                  f"(want True: each removed layer costs more confidence)")
        return result

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
    def _gauss_blur_frames(x: torch.Tensor, k: int = 21,
                           sigma: float = 10.0) -> torch.Tensor:
        """Separable Gaussian blur for a stack of frames (n, C, H, W).

        Same kernel as the deletion/insertion protocol (k=21, sigma=10) so
        the blur-fill frame-drop numbers live on the same evidence-removal
        scale as the del/ins curves.
        """
        import torch.nn.functional as _F
        n, C, H, W = x.shape
        ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
        g = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
        g = g / g.sum()
        kx = g.view(1, 1, 1, k).expand(C, 1, 1, k).contiguous()
        ky = g.view(1, 1, k, 1).expand(C, 1, k, 1).contiguous()
        pad = k // 2
        out = _F.conv2d(x, kx, padding=(0, pad), groups=C)
        out = _F.conv2d(out, ky, padding=(pad, 0), groups=C)
        return out

    @staticmethod
    def _mask_frames(clip: torch.Tensor, drop_idx, mode: str) -> torch.Tensor:
        """Remove frames at drop_idx from clip (1, T, C, H, W), in place.

        mode="replicate" (Phase 28 primary): each dropped frame is replaced by
            its nearest non-dropped neighbour (freeze-frame).  Stays
            in-distribution — removes the dropped frame's unique evidence
            without injecting an out-of-distribution artifact.
            Phase 31 caveat (run 6-12-26 1300hrs): on a FULLY-FAKE clip with
            16 temporally-redundant frames the neighbour carries the SAME
            class evidence, so replicate-fill measures a quantity that is
            physically ≈ 0 for any map — top-vs-random ratios from this fill
            are noise/noise (P29's +5.12 and P30's −19.2 are the same coin
            flip).  Reported for cross-phase continuity only.
        mode="blur" (Phase 31 primary): dropped frames are Gaussian-blurred
            in place (same kernel as the del/ins protocol).  DESTROYS the
            evidence in those frames instead of copying it back in from a
            neighbour, while staying closer to the input manifold than gray
            fill — the only fill of the three that can show a top-vs-random
            differential on fully-fake clips.
        mode="zero" (legacy, Phase ≤27): frames set to 0 in normalised space
            (≈ ImageNet-mean gray).  Phase 27 measured a +0.37 fake-prob shift
            for ANY masked frame — the gray-frame artifact swamped the
            top-vs-random differential and pinned all k-ratios at 1.000.
        """
        T = clip.shape[1]
        drop_set = {int(i) for i in drop_idx}
        if mode == "blur":
            idx = list(drop_set)
            clip[0, idx] = ExplanationMetrics._gauss_blur_frames(clip[0, idx])
            return clip
        if mode == "zero" or len(drop_set) >= T:
            clip[0, list(drop_set)] = 0.0
            return clip
        keep = np.asarray([t for t in range(T) if t not in drop_set])
        for t in drop_set:
            nearest = int(keep[np.argmin(np.abs(keep - t))])
            clip[0, t] = clip[0, nearest]
        return clip

    @staticmethod
    def frame_attention_drop_test(
        model, loader, device, k_values=(1, 2, 4), seed: int = 42
    ) -> dict:
        """
        Intrinsic faithfulness test.

        For each video in the loader:
        1. Forward pass (eval, no_grad) → get M_frame (B, T) from temporal_gate.
        2. Rank frames by attention score (descending).
        3. For each K in k_values, and for each fill protocol:
           a. Drop top-K frames → re-forward → record prob.
           b. Drop K random frames (seeded, same indices for both fills)
              → re-forward → record prob.
        4. Aggregate: conf_drop = original_prob - masked_prob.

        Phase 28 protocol change: the PRIMARY fill is nearest-frame
        replication (freeze-frame, in-distribution).  The legacy zero-fill
        numbers are still computed in the same pass and reported under
        *_zerofill keys for cross-phase comparability.

        Phase 31 protocol change: a BLUR fill is added (*_blurfill keys)
        and becomes the headline protocol.  Replicate fill copies the
        nearest kept frame back in — on fully-fake, temporally-redundant
        clips the replacement frame carries the same class evidence, so
        single-frame "necessity" is physically ≈ 0 for ANY map and the
        replicate ratios are noise/noise.  Blur fill destroys the evidence
        in the dropped frames (same Gaussian as the del/ins protocol), so a
        temporal map that ranks evidence-bearing frames higher CAN show
        top > random here.

        Returns dict with keys (per K):
            k{K}_top_conf_drop, k{K}_random_conf_drop, k{K}_ratio          (replicate fill)
            k{K}_top_conf_drop_blurfill, k{K}_random_conf_drop_blurfill,
            k{K}_ratio_blurfill                                            (Phase 31 headline)
            k{K}_top_conf_drop_zerofill, k{K}_random_conf_drop_zerofill,
            k{K}_ratio_zerofill                                            (legacy fill)
        A faithful explanation shows top_conf_drop >> random_conf_drop.
        """
        import numpy as np

        model.eval()
        rng = np.random.default_rng(seed)

        _fills = ("replicate", "blur", "zero")
        accum = {(k, f): {"top": [], "rand": []} for k in k_values for f in _fills}

        _debug_printed = False   # print diagnostics once for the first batch

        with torch.no_grad():
            for batch in loader:
                frames = batch["frames"].to(device)          # (B, T, C, H, W)
                B, T, C, H, W = frames.shape

                out     = model(frames)
                orig_p  = out.prob.cpu()                      # (B,)
                M_t     = out.M_t.cpu()                       # (B, T, h, w)
                # Phase 23: use M_frame if present (proper per-frame attention
                # produced by the temporal_gate bottleneck) — that is the
                # quantity the classifier actually depends on.  Fall back to
                # M_t.amax() ranking for older checkpoints without M_frame.
                _has_M_frame = hasattr(out, "M_frame") and out.M_frame is not None
                M_frame_cpu  = out.M_frame.cpu() if _has_M_frame else None

                # Task 1.3 diagnostic: dump raw M_t statistics once (first batch only)
                if not _debug_printed:
                    print("[DIAG frame_attn_drop] fill protocols: headline=blur "
                          "(Phase 31, evidence-destroying, *_blurfill keys), "
                          "replicate (Phase 28 freeze-frame -- physically ~0 signal "
                          "on fully-fake redundant clips, continuity only), "
                          "zero (legacy *_zerofill keys)")
                    print(f"[DIAG frame_attn_drop] M_t shape: {M_t.shape}")
                    print(f"[DIAG frame_attn_drop] M_t mean per frame:\n"
                          f"  {M_t.mean(dim=(-1,-2))}")
                    print(f"[DIAG frame_attn_drop] M_t max  per frame:\n"
                          f"  {M_t.amax(dim=(-1,-2))}")
                    print(f"[DIAG frame_attn_drop] M_t min  per frame:\n"
                          f"  {M_t.amin(dim=(-1,-2))}")
                    print(f"[DIAG frame_attn_drop] M_t std  per frame:\n"
                          f"  {M_t.std(dim=(-1,-2))}")
                    if _has_M_frame:
                        print(f"[DIAG frame_attn_drop] M_frame source: temporal_gate (Phase 23)")
                        print(f"[DIAG frame_attn_drop] M_frame per sample:\n  {M_frame_cpu}")
                    else:
                        print(f"[DIAG frame_attn_drop] M_frame source: M_t.amax fallback")
                    _debug_printed = True

                if _has_M_frame:
                    frame_scores = M_frame_cpu                # (B, T) — direct attention
                else:
                    # Legacy fallback: peak intensity per frame
                    frame_scores = M_t.amax(dim=(-1, -2))     # (B, T)

                for b in range(B):
                    scores_b  = frame_scores[b].numpy()              # (T,)
                    # .copy() prevents negative-stride error when indexing a tensor
                    ranked    = np.argsort(scores_b)[::-1].copy()    # desc, contiguous
                    orig_prob = float(orig_p[b])

                    for k in k_values:
                        k = min(k, T)
                        top_k_idx  = ranked[:k]
                        # Same random indices for both fill protocols so the
                        # replicate-vs-zero comparison is apples-to-apples.
                        rand_k_idx = rng.choice(T, size=k, replace=False)

                        for fill in _fills:
                            f_top = ExplanationMetrics._mask_frames(
                                frames[b:b+1].clone(), top_k_idx, fill)
                            drop_top = orig_prob - float(
                                model(f_top.to(device)).prob.cpu())

                            f_rand = ExplanationMetrics._mask_frames(
                                frames[b:b+1].clone(), rand_k_idx, fill)
                            drop_rand = orig_prob - float(
                                model(f_rand.to(device)).prob.cpu())

                            accum[(k, fill)]["top"].append(drop_top)
                            accum[(k, fill)]["rand"].append(drop_rand)

        result = {}
        _suffix = {"replicate": "", "blur": "_blurfill", "zero": "_zerofill"}
        for k in k_values:
            for fill in _fills:
                tops   = accum[(k, fill)]["top"]
                rands  = accum[(k, fill)]["rand"]
                t_mean = float(np.mean(tops))  if tops  else 0.0
                r_mean = float(np.mean(rands)) if rands else 0.0
                ratio  = t_mean / (r_mean + 1e-8)
                _sfx = _suffix[fill]
                result[f"k{k}_top_conf_drop{_sfx}"]    = t_mean
                result[f"k{k}_random_conf_drop{_sfx}"] = r_mean
                result[f"k{k}_ratio{_sfx}"]            = ratio
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
