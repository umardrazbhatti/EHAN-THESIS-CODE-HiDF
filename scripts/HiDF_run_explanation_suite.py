"""
scripts/run_explanation_suite.py — Orchestrator for all explanation quality metrics.

Runs all intrinsic metrics + new frame_attention_drop_test + stability_check
on the given model and test loader. Saves a unified JSON to output_path.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from metrics.HiDF_explanation import ExplanationMetrics


def run_explanation_suite(model, test_loader, config, output_path: Path) -> dict:
    """
    Run all explanation metrics on the trained model + test loader.
    Save unified JSON to output_path. Print summary table.
    Returns the metrics dict.

    Args:
        model       : trained EAHN model (eval mode will be set internally)
        test_loader : DataLoader for test set (no shuffle)
        config      : EAHNConfig
        output_path : Path where explanation_metrics.json will be written
    """
    device = torch.device(config.device)
    model.eval()

    print("\n[ExplanationSuite] Collecting M_t across test set...")

    # ── 1. Collect all M_t + probs + gradient maps ─────────────────────────────
    all_M_t_up  = []
    all_probs   = []
    all_frames  = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Suite pass", leave=False):
            frames = batch["frames"].to(device)
            out    = model(frames)
            all_M_t_up.append(out.M_t_up.cpu())
            all_probs.extend(out.prob.cpu().tolist())
            all_frames.append(frames.cpu())

    all_M_t_up = torch.cat(all_M_t_up, dim=0)   # (N, T, H, W)
    all_frames  = torch.cat(all_frames, dim=0)   # (N, T, C, H, W)
    N = len(all_M_t_up)

    subset_size = min(getattr(config, "heatmap_samples", 20), N)
    rng         = np.random.default_rng(42)
    indices     = rng.choice(N, subset_size, replace=False)

    # ── 2. Temporal SSIM ───────────────────────────────────────────────────────
    print("[ExplanationSuite] Computing temporal SSIM...")
    ssim_val = ExplanationMetrics.temporal_ssim(all_M_t_up[indices])

    # ── 3. Gradient maps for faithfulness correlation ─────────────────────────
    print("[ExplanationSuite] Computing faithfulness correlation (gradient)...")
    grad_maps = []
    model.eval()
    for idx in tqdm(indices, desc="Grad maps", leave=False):
        frames_t = all_frames[idx:idx+1].to(device).requires_grad_(True)
        out      = model(frames_t)
        out.logit.backward()
        grads    = frames_t.grad.abs().mean(dim=2)        # (1, T, H, W)
        grads_7  = torch.nn.functional.interpolate(
            grads.reshape(grads.shape[1], 1, *grads.shape[2:]),
            size=(7, 7), mode="bilinear", align_corners=False,
        ).squeeze(1)                                       # (T, 7, 7)
        grad_maps.append(grads_7.detach().cpu())
        frames_t.requires_grad_(False)
    model.eval()

    grad_maps  = torch.stack(grad_maps)                   # (subset, T, 7, 7)
    M_sub      = all_M_t_up[indices].mean(dim=1)          # (subset, H, W)
    M_sub_7    = torch.nn.functional.interpolate(
        M_sub.unsqueeze(1), size=(7, 7), mode="bilinear", align_corners=False,
    ).squeeze(1)                                           # (subset, 7, 7)
    grad_7_avg = grad_maps.mean(dim=1)                    # (subset, 7, 7)

    faithful_corr = ExplanationMetrics.faithfulness_correlation(
        M_sub_7.reshape(subset_size, -1),
        grad_7_avg.reshape(subset_size, -1),
    )

    # ── 4. Deletion / Insertion AUC ───────────────────────────────────────────
    print("[ExplanationSuite] Computing deletion/insertion AUC...")
    del_ins = {"deletion_auc": 0.0, "insertion_auc": 0.0}
    try:
        # Phase 20.1: use 8 samples instead of 1 for statistically reliable
        # deletion/insertion AUC. Baseline (Xception+GradCAM) used 40 frames;
        # we use 8 videos × 16 frames = 128 effective frames. T4 memory budget
        # allows N=8 comfortably (~4 GB peak); larger N risks OOM during the
        # batched forward pass inside deletion_insertion_auc.
        N_DEL_INS = min(8, N)
        rng_di = np.random.default_rng(42)
        di_indices = rng_di.choice(N, size=N_DEL_INS, replace=False)
        frames_sample = all_frames[di_indices]              # (N_DEL_INS, T, C, H, W)
        sal_sample    = all_M_t_up[di_indices].numpy()      # (N_DEL_INS, T, H, W)
        del_ins = ExplanationMetrics.deletion_insertion_auc(
            model, frames_sample, sal_sample, steps=10
        )
        print(f"  [del/ins AUC on N={N_DEL_INS} samples]  "
              f"del={del_ins.get('deletion_auc', 0):.4f}  "
              f"ins={del_ins.get('insertion_auc', 0):.4f}")
    except Exception as e:
        print(f"  [del/ins AUC skipped: {e}]")

    # ── 5. Collapse diagnostics ───────────────────────────────────────────────
    print("[ExplanationSuite] Computing collapse diagnostics...")
    collapse_diag = ExplanationMetrics.collapse_diagnostics(all_M_t_up)

    # ── 6. Model randomization sanity check ───────────────────────────────────
    # Task 1.7: fixed seed + n_random=30 for deterministic, statistically stable result.
    # Compute ONCE here; evaluate.py reads this value from the returned dict.
    mt_vs_random_cosine = 1.0
    _RANDOM_TEST_N = int(getattr(config, "random_test_n_samples", 30))
    try:
        import torch as _torch_sanity
        import numpy as _np_sanity
        _torch_sanity.manual_seed(42)
        _np_sanity.random.seed(42)
        from xai.HiDF_sanity_checks import model_randomization_check
        # Use a fixed sample (index 0 of the subset) so both runs pick the same frame
        _frames_s = all_frames[int(indices[0]):int(indices[0])+1].to(device)
        mt_vs_random_cosine = model_randomization_check(
            model, _frames_s, n_random=_RANDOM_TEST_N
        )
        print(f"[ExplanationSuite] model_randomization cosine = {mt_vs_random_cosine:.4f} "
              f"(n_random={_RANDOM_TEST_N}, seed=42)")
    except Exception as e:
        print(f"  [model_randomization skipped: {e}]")

    # ── 7. Frame attention drop test ──────────────────────────────────────────
    print("[ExplanationSuite] Computing frame_attention_drop_test...")
    drop_results = {}
    try:
        drop_results = ExplanationMetrics.frame_attention_drop_test(
            model, test_loader, device, k_values=(1, 2, 4), seed=42
        )
    except Exception as e:
        print(f"  [frame_attention_drop_test skipped: {e}]")

    # ── 8. Stability check ────────────────────────────────────────────────────
    print("[ExplanationSuite] Computing stability check...")
    stability = {}
    try:
        stability = ExplanationMetrics.stability_check(
            model, test_loader, device, n_batches=5
        )
    except Exception as e:
        print(f"  [stability_check skipped: {e}]")

    # ── Assemble result ───────────────────────────────────────────────────────
    result = {
        "active_manipulation": getattr(config, "active_manipulation", ""),
        "intrinsic": {
            "deletion_auc":              del_ins.get("deletion_auc", 0.0),
            "insertion_auc":             del_ins.get("insertion_auc", 0.0),
            "temporal_ssim":             float(ssim_val),
            "faithfulness_corr":         float(faithful_corr),
            "inter_sample_cos_mean":     float(collapse_diag.get("inter_sample_cosine_mean", 0.0)),
            "peak_mode_share":           float(collapse_diag.get("peak_mode_share", 0.0)),
            "m_t_std_mean":              float(collapse_diag.get("m_t_std_mean", 0.0)),
            "mt_vs_random_model_cosine": float(mt_vs_random_cosine),
        },
        "frame_attention_drop": drop_results,
        "stability":            stability,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n[ExplanationSuite] === Summary ===")
    print(f"  Temporal SSIM            : {result['intrinsic']['temporal_ssim']:.3f}")
    print(f"  Faithfulness corr        : {result['intrinsic']['faithfulness_corr']:.3f}")
    print(f"  Deletion AUC             : {result['intrinsic']['deletion_auc']:.3f}")
    print(f"  Insertion AUC            : {result['intrinsic']['insertion_auc']:.3f}")
    print(f"  Inter-sample cosine      : {result['intrinsic']['inter_sample_cos_mean']:.3f}")
    print(f"  Peak mode share          : {result['intrinsic']['peak_mode_share']:.3f}")
    print(f"  M_t std mean             : {result['intrinsic']['m_t_std_mean']:.4f}")
    print(f"  Mt vs random cosine      : {result['intrinsic']['mt_vs_random_model_cosine']:.3f}")
    for k in (1, 2, 4):
        if f"k{k}_ratio" in drop_results:
            print(f"  Drop ratio K={k}           : {drop_results[f'k{k}_ratio']:.3f} "
                  f"(top={drop_results[f'k{k}_top_conf_drop']:.3f} "
                  f"rand={drop_results[f'k{k}_random_conf_drop']:.3f})")
    if stability:
        print(f"  Stability cosine (mean)  : {stability.get('stability_cosine_mean', 0):.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ExplanationSuite] metrics saved → {output_path}")

    return result
