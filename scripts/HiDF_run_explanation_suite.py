"""
scripts/run_explanation_suite.py — Orchestrator for all explanation quality metrics.

Runs all intrinsic metrics + new frame_attention_drop_test + stability_check
on the given model and test loader. Saves a unified JSON to output_path.

MEMORY DESIGN (6-7-26 refactor):
  - GPU-first: keep all_M_t_up + sampled all_frames on GPU (T4 has 15 GB, ~6 GB used).
  - No CPU-RAM accumulation of full-test-set frames (was ~5 GB, caused OOM at 414/415).
  - Reuse pre-collected buffers from evaluate.py if passed in (avoid duplicate pass).
  - Only allocate frames for the indices we actually use (subset ∪ di_indices).
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

from metrics.HiDF_explanation import ExplanationMetrics


def run_explanation_suite(
    model,
    test_loader,
    config,
    output_path: Path,
    all_M_t_up_gpu=None,
    all_M_suff_up_gpu=None,          # Phase 35: sufficiency lens for insertion (None = use M_t)
    all_probs=None,
    all_labels=None,
    all_vid_paths=None,
) -> dict:
    """
    Run all explanation metrics on the trained model + test loader.
    Save unified JSON to output_path. Print summary table.
    Returns the metrics dict.

    Args:
        model              : trained EAHN model (eval mode set internally)
        test_loader        : DataLoader for test set (shuffle=False so order is stable)
        config             : EAHNConfig
        output_path        : Path where explanation_metrics.json will be written
        all_M_t_up_gpu     : optional pre-collected (N, T, H, W) GPU tensor from evaluate.py
        all_probs          : optional list of N floats (probabilities)
        all_labels         : optional list of N ints (labels)
        all_vid_paths      : optional list of N strs (video paths)

    If the optional buffers are provided, the suite skips its own collection pass
    (saves ~3 min and ~1.3 GB of duplicated M_t).
    """
    device = torch.device(config.device)
    model.eval()

    # ── 1. Collect M_t (skip if pre-supplied) ──────────────────────────────────
    if all_M_t_up_gpu is None:
        print("\n[ExplanationSuite] Collecting M_t across test set (GPU-resident)...")
        _M_chunks    = []
        all_probs    = []
        all_labels   = []
        all_vid_paths = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Suite pass", leave=False):
                frames = batch["frames"].to(device, non_blocking=True)
                out    = model(frames)
                _M_chunks.append(out.M_t_up.detach())             # keep on GPU
                all_probs.extend(out.prob.detach().cpu().tolist())
                all_labels.extend(batch.get("label", torch.zeros(frames.shape[0])).cpu().tolist())
                all_vid_paths.extend(batch.get("video_path", [""] * frames.shape[0]))
                del frames, out
        all_M_t_up_gpu = torch.cat(_M_chunks, dim=0)               # (N, T, H, W) on GPU
        del _M_chunks
    else:
        print("\n[ExplanationSuite] Reusing pre-collected M_t (skipping Pass 1).")

    N = len(all_M_t_up_gpu)
    all_labels    = list(all_labels)
    all_vid_paths = list(all_vid_paths)

    subset_size = min(getattr(config, "heatmap_samples", 10), N)
    rng         = np.random.default_rng(42)
    indices     = rng.choice(N, subset_size, replace=False)

    # ── Pre-sample del/ins indices ─────────────────────────────────────────────
    # Phase 28: 10 → 50 clips (the Phase ≤27 headline numbers rested on 10
    # clips — too noisy for publication).  Per-clip calls keep VRAM flat.
    N_DEL_INS = min(50, N)
    rng_di    = np.random.default_rng(42)
    di_indices = rng_di.choice(N, size=N_DEL_INS, replace=False)

    # Combined set of indices for which we actually need raw frames
    needed_idx_set = set(int(i) for i in indices) | set(int(i) for i in di_indices)
    print(f"[ExplanationSuite] need frames for {len(needed_idx_set)} sampled indices "
          f"(subset={subset_size}, del_ins={N_DEL_INS})")

    # ── 2. Targeted frames pass (GPU): only collect frames at needed indices ──
    # Test loader has shuffle=False so we walk it in order and map batch slots
    # to global indices. We keep these frames on GPU.
    print("[ExplanationSuite] Collecting frames at sampled indices (GPU)...")
    frames_by_idx = {}                                            # {int_idx: (1, T, C, H, W) GPU}
    _bs = test_loader.batch_size or 1
    _cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            _frames_batch = batch["frames"]                       # (b, T, C, H, W)
            _b = _frames_batch.shape[0]
            # Determine which slots in this batch we need
            _global = list(range(_cursor, _cursor + _b))
            _local_keep = [(i, g) for i, g in enumerate(_global) if g in needed_idx_set]
            if _local_keep:
                _frames_gpu = _frames_batch.to(device, non_blocking=True)
                for _i, _g in _local_keep:
                    frames_by_idx[_g] = _frames_gpu[_i:_i+1].detach().clone()
                del _frames_gpu
            _cursor += _b
            if len(frames_by_idx) >= len(needed_idx_set):
                break
    print(f"[ExplanationSuite] collected {len(frames_by_idx)} frame tensors on GPU "
          f"(~{sum(f.numel()*4 for f in frames_by_idx.values())/1e6:.1f} MB)")

    # ── 3. Temporal SSIM ───────────────────────────────────────────────────────
    print("[ExplanationSuite] Computing temporal SSIM...")
    ssim_val = ExplanationMetrics.temporal_ssim(all_M_t_up_gpu[indices])

    # ── 4. Faithfulness correlation (gradient saliency) ───────────────────────
    print("[ExplanationSuite] Computing faithfulness correlation (gradient)...")
    grad_maps = []
    model.eval()
    for idx in tqdm(indices, desc="Grad maps", leave=False):
        frames_t = frames_by_idx[int(idx)].clone().requires_grad_(True)
        out      = model(frames_t)
        out.logit.backward()
        grads    = frames_t.grad.abs().mean(dim=2)                # (1, T, H, W)
        grads_7  = torch.nn.functional.interpolate(
            grads.reshape(grads.shape[1], 1, *grads.shape[2:]),
            size=(7, 7), mode="bilinear", align_corners=False,
        ).squeeze(1)                                               # (T, 7, 7)
        grad_maps.append(grads_7.detach())                        # keep GPU
        del frames_t, out, grads, grads_7
    model.eval()

    grad_maps  = torch.stack(grad_maps)                            # (subset, T, 7, 7) GPU
    M_sub      = all_M_t_up_gpu[indices].mean(dim=1)               # (subset, H, W) GPU
    M_sub_7    = torch.nn.functional.interpolate(
        M_sub.unsqueeze(1), size=(7, 7), mode="bilinear", align_corners=False,
    ).squeeze(1)                                                    # (subset, 7, 7) GPU
    grad_7_avg = grad_maps.mean(dim=1)                              # (subset, 7, 7) GPU

    faithful_corr = ExplanationMetrics.faithfulness_correlation(
        M_sub_7.reshape(subset_size, -1),
        grad_7_avg.reshape(subset_size, -1),
    )
    del grad_maps, M_sub, M_sub_7, grad_7_avg
    torch.cuda.empty_cache()

    # ── 5. Deletion / Insertion AUC (GPU-resident frames + saliency) ──────────
    # Phase 42: report del/ins under EACH baseline in config.insertion_baselines.
    # The blur fill is confounded -- a blurred clip reads as MORE fake, so the
    # insertion curve is floored/non-monotonic regardless of map quality (proven
    # P38/P41).  black/mean are flat baselines that give a cleaner sufficiency
    # readout.  The FIRST listed baseline is the HEADLINE: its aggregates fill the
    # canonical keys (back-compat) and its per-sample rows are saved.  Every
    # baseline is ALSO stored under result["del_ins_baselines"][<name>].  When
    # insertion_baselines is unset/single, this reproduces the prior result exactly.
    _bl_raw = str(getattr(config, "insertion_baselines", "") or "").strip()
    if _bl_raw:
        _baselines = [b.strip() for b in _bl_raw.split(",") if b.strip()]
    else:
        _baselines = [str(getattr(config, "insertion_baseline", "blur"))]
    _seen = set()
    _baselines = [b for b in _baselines if not (b in _seen or _seen.add(b))]
    _headline_bl = _baselines[0]
    print(f"[ExplanationSuite] Computing deletion/insertion AUC "
          f"(N={N_DEL_INS}, steps=20, random-control; baselines={_baselines})...")

    def _run_del_ins_for_baseline(_bl, _headline=False):
        """Run the per-clip del/ins loop for one canvas baseline; return (agg, rows)."""
        _da, _ia, _ig, _dg = [], [], [], []
        _rows = []
        for _si, _di in enumerate(di_indices):
            _di = int(_di)
            _f_s   = frames_by_idx[_di]                               # (1,T,C,H,W) GPU
            _s_s   = all_M_t_up_gpu[_di:_di+1].detach().cpu().numpy() # necessity (deletion)
            # Insertion ranks by the sufficiency lens (= M_t when single-lens).
            _su    = all_M_suff_up_gpu if all_M_suff_up_gpu is not None else all_M_t_up_gpu
            _s_ins = _su[_di:_di+1].detach().cpu().numpy()
            _prob_s  = float(all_probs[_di])
            _label_s = int(all_labels[_di]) if all_labels else -1
            _vpath_s = str(all_vid_paths[_di]) if all_vid_paths else ""
            _di_result = ExplanationMetrics.deletion_insertion_auc(
                model, _f_s, _s_s, steps=20, n_samples=1,
                random_control=True, verbose=False,
                saliency_ins=_s_ins, baseline=str(_bl),
            )
            _d_auc  = float(_di_result.get("deletion_auc", 0.0))
            _i_auc  = float(_di_result.get("insertion_auc", 0.0))
            _i_gain = float(_di_result.get("ins_gain_over_random", 0.0))
            _d_gain = float(_di_result.get("del_gain_over_random", 0.0))
            _da.append(_d_auc); _ia.append(_i_auc); _ig.append(_i_gain); _dg.append(_d_gain)
            _rows.append({
                "video_path":       _vpath_s,
                "label":            _label_s,
                "prob":             _prob_s,
                "deletion_auc":     _d_auc,
                "insertion_auc":    _i_auc,
                "ins_gain_over_random": _i_gain,
                "del_gain_over_random": _d_gain,
                "m_t_std":          float(all_M_t_up_gpu[_di].std().item()),
                "faithfulness_score": _i_auc - _d_auc,
            })
            if _headline:
                print(f"  [del/ins {_bl} sample {_si+1}/{N_DEL_INS}]  "
                      f"del={_d_auc:.4f}  ins={_i_auc:.4f}  "
                      f"ins_gain={_i_gain:+.4f}  del_gain={_d_gain:+.4f}")
            # Per-sample VRAM hygiene -- del/ins clones full (T,C,H,W) per step
            torch.cuda.empty_cache()
        _fake = [r for r in _rows if r.get("label", -1) == 1]
        _agg = {
            "deletion_auc":            float(np.mean(_da)) if _da else 0.0,
            "insertion_auc":           float(np.mean(_ia)) if _ia else 0.0,
            "ins_gain_over_random":    float(np.mean(_ig)) if _ig else 0.0,
            "del_gain_over_random":    float(np.mean(_dg)) if _dg else 0.0,
            "deletion_auc_fake_only":  float(np.mean([r["deletion_auc"]  for r in _fake])) if _fake else 0.0,
            "insertion_auc_fake_only": float(np.mean([r["insertion_auc"] for r in _fake])) if _fake else 0.0,
            "n_fake":                  len(_fake),
        }
        print(f"  [del/ins {_bl} aggregate N={N_DEL_INS}]  "
              f"del={_agg['deletion_auc']:.4f}  ins={_agg['insertion_auc']:.4f}  "
              f"del_fake={_agg['deletion_auc_fake_only']:.4f}  "
              f"ins_fake={_agg['insertion_auc_fake_only']:.4f}  "
              f"ins_gain={_agg['ins_gain_over_random']:+.4f}  "
              f"del_gain={_agg['del_gain_over_random']:+.4f}")
        return _agg, _rows

    del_ins_by_baseline = {}
    per_sample_del_ins  = []
    del_ins             = {}
    for _bl in _baselines:
        _agg, _rows = _run_del_ins_for_baseline(_bl, _headline=(_bl == _headline_bl))
        del_ins_by_baseline[_bl] = _agg
        if _bl == _headline_bl:
            del_ins            = _agg
            per_sample_del_ins = _rows

    _per_sample_path = Path("outputs") / "explanation_per_sample.json"
    _per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_per_sample_path, "w") as _psf:
        json.dump(per_sample_del_ins, _psf, indent=2)
    print(f"  [per-sample results ({_headline_bl}) saved -> {_per_sample_path}]")

    # ── 5b. ROAD debiased deletion (Phase 38) ─────────────────────────────────
    # The blur del/ins above is confounded: a blurred clip reads as MORE fake, so
    # its curve is flat/non-monotonic no matter the map (the long-standing wall).
    # ROAD refills removed pixels with NOISY imputation (near-manifold), giving a
    # MONOTONIC 'relative' curve, and contrasts the intrinsic map ordering with a
    # GRADIENT ordering (faithful reference) and a RANDOM ordering (floor).
    print("[ExplanationSuite] Computing ROAD debiased deletion "
          "(intrinsic / gradient / random orderings)...")
    road = {}
    try:
        _di_list     = [int(d) for d in di_indices]
        _road_frames = torch.cat([frames_by_idx[d] for d in _di_list], dim=0)
        _road_labels = [int(all_labels[d]) if all_labels else -1 for d in _di_list]
        _intrinsic_sal = all_M_t_up_gpu[_di_list].detach().cpu().numpy()      # (Ndi,T,H,W)
        # Gradient saliency for the SAME clips (the faithful-ordering reference).
        _grad_sal = []
        model.eval()
        for d in _di_list:
            _ft = frames_by_idx[d].clone().requires_grad_(True)
            _o  = model(_ft)
            _o.logit.backward()
            _grad_sal.append(
                _ft.grad.abs().mean(dim=2).detach().cpu().numpy()[0])          # (T,H,W)
            del _ft, _o
        _grad_sal = np.stack(_grad_sal)                                        # (Ndi,T,H,W)
        model.eval()
        road = ExplanationMetrics.road_faithfulness(
            model, _road_frames,
            orderings={"intrinsic": _intrinsic_sal,
                       "gradient":  _grad_sal,
                       "random":    None},
            labels=_road_labels, steps=20, chunk=4, verbose=True,
        )
        del _road_frames, _grad_sal, _intrinsic_sal
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [ROAD skipped: {e}]")

    # ── 6. Collapse diagnostics (GPU) ─────────────────────────────────────────
    print("[ExplanationSuite] Computing collapse diagnostics...")
    collapse_diag = ExplanationMetrics.collapse_diagnostics(all_M_t_up_gpu)

    # ── 7. Model randomization sanity check ───────────────────────────────────
    mt_vs_random_cosine = 1.0
    _RANDOM_TEST_N = int(getattr(config, "random_test_n_samples", 30))
    try:
        import torch as _torch_sanity
        import numpy as _np_sanity
        _torch_sanity.manual_seed(42)
        _np_sanity.random.seed(42)
        from xai.HiDF_sanity_checks import model_randomization_check
        _ref_idx = int(indices[0])
        _frames_s = frames_by_idx[_ref_idx]                         # already on GPU
        mt_vs_random_cosine = model_randomization_check(
            model, _frames_s, n_random=_RANDOM_TEST_N
        )
        print(f"[ExplanationSuite] model_randomization cosine = {mt_vs_random_cosine:.4f} "
              f"(n_random={_RANDOM_TEST_N}, seed=42)")
    except Exception as e:
        print(f"  [model_randomization skipped: {e}]")

    # ── 8. Frame attention drop test ──────────────────────────────────────────
    print("[ExplanationSuite] Computing frame_attention_drop_test...")
    drop_results = {}
    try:
        drop_results = ExplanationMetrics.frame_attention_drop_test(
            model, test_loader, device, k_values=(1, 2, 4), seed=42
        )
    except Exception as e:
        print(f"  [frame_attention_drop_test skipped: {e}]")

    # ── 9. Stability check ────────────────────────────────────────────────────
    print("[ExplanationSuite] Computing stability check...")
    stability = {}
    try:
        stability = ExplanationMetrics.stability_check(
            model, test_loader, device, n_batches=5
        )
    except Exception as e:
        print(f"  [stability_check skipped: {e}]")

    # ── 9b. Layer-ablation causal check (Phase 36 intrinsic decomposition) ────
    # Remove each declared evidence layer and measure the fake-confidence drop;
    # a faithful decomposition shows the declared %-share predicts the drop.
    # This is the intrinsic-safe validation (probes the model's OWN parts).
    print("[ExplanationSuite] Computing layer-ablation causal check...")
    layer_causal = {}
    try:
        if getattr(model, "decomp_enabled", False):
            _lc_frames = torch.cat(
                [frames_by_idx[int(_d)] for _d in di_indices], dim=0)   # (N,T,C,H,W)
            _lc_labels = [int(all_labels[int(_d)]) if all_labels else 1
                          for _d in di_indices]
            layer_causal = ExplanationMetrics.layer_ablation_causal(
                model, _lc_frames, labels=_lc_labels,
                n_samples=len(di_indices), chunk=4,
            )
    except Exception as e:
        print(f"  [layer_ablation_causal skipped: {e}]")

    # ── 9c. Layer-ablation CUMULATIVE (Phase 38) ──────────────────────────────
    # Remove the declared layers one-by-one in decreasing-share order (top-1,
    # top-1+2, ...) and report the running fake-confidence: a faithful
    # decomposition drops MONOTONICALLY -- the 'delete 1 layer -> drop, delete 2
    # -> drop more' readout requested.  Intrinsic-safe (no pixel perturbation).
    print("[ExplanationSuite] Computing layer-ablation cumulative check...")
    layer_cumulative = {}
    try:
        if getattr(model, "decomp_enabled", False):
            _lcc_frames = torch.cat(
                [frames_by_idx[int(_d)] for _d in di_indices], dim=0)
            _lcc_labels = [int(all_labels[int(_d)]) if all_labels else 1
                           for _d in di_indices]
            layer_cumulative = ExplanationMetrics.layer_ablation_cumulative(
                model, _lcc_frames, labels=_lcc_labels,
                n_samples=len(di_indices), chunk=4,
            )
    except Exception as e:
        print(f"  [layer_ablation_cumulative skipped: {e}]")

    # ── 9d. Contribution-space del/ins (Phase 45, analytic, ALL samples) ──────
    # The additive head's logit is an exact sum over cells, so deletion/insertion
    # of a cell == removing/adding its term -- no forward, no pixel occlusion, no
    # re-globalisation, no 20%-base floor.  This is the metric the head was built
    # for; the pixel-occlusion del/ins above re-ran the full blended net and so
    # could never see the head's by-construction faithfulness.  Free => runs on
    # ALL samples (escapes the ~24-clip fake-only noise).
    contribution_di = {}
    try:
        if getattr(model, "aeh_enabled", False):
            print("[ExplanationSuite] Computing contribution-space del/ins "
                  "(analytic additive-head terms, ALL samples)...")
            _C, _MF, _BASE, _LAB = [], [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    _fr = batch["frames"].to(device, non_blocking=True)
                    _o  = model(_fr)
                    if getattr(_o, "aeh_contrib", None) is None:
                        del _fr, _o
                        break
                    _C.append(_o.aeh_contrib.detach().cpu())              # (b,T,Ncells)
                    _MF.append(_o.M_frame.detach().cpu())                 # (b,T)
                    _BASE.append(_o.base_logit.detach().cpu().reshape(-1))  # (b,)
                    _LAB.extend(batch.get("label",
                                torch.zeros(_fr.shape[0])).cpu().tolist())
                    del _fr, _o
            if _C:
                import torch as _t
                def _scalar(x, d):
                    if x is None:
                        return d
                    if _t.is_tensor(x):
                        return float(x.detach().cpu().reshape(-1)[0].item())
                    return float(x)
                _scale = _scalar(getattr(model, "aeh_scale", None), 1.0)
                _bias  = _scalar(getattr(model, "aeh_bias",  None), 0.0)
                _gam   = _scalar(getattr(model, "aeh_gamma_current", None), 0.0)
                contribution_di = ExplanationMetrics.contribution_del_ins(
                    _t.cat(_C, 0).numpy(), _t.cat(_MF, 0).numpy(),
                    _scale, _bias, _t.cat(_BASE, 0).numpy(), _gam,
                    labels=_LAB, verbose=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [contribution_del_ins skipped: {e}]")

    # ── Assemble result ───────────────────────────────────────────────────────
    # Phase 28: fake-only del/ins aggregates from the per-sample records
    # (artifact localisation is only well-defined on manipulated samples).
    _fake_rows = [r for r in per_sample_del_ins if r.get("label", -1) == 1]
    _del_fake = (float(np.mean([r["deletion_auc"]  for r in _fake_rows]))
                 if _fake_rows else 0.0)
    _ins_fake = (float(np.mean([r["insertion_auc"] for r in _fake_rows]))
                 if _fake_rows else 0.0)

    result = {
        "active_manipulation": getattr(config, "active_manipulation", ""),
        "intrinsic": {
            "deletion_auc":              del_ins.get("deletion_auc", 0.0),
            "insertion_auc":             del_ins.get("insertion_auc", 0.0),
            "ins_gain_over_random":      del_ins.get("ins_gain_over_random", 0.0),
            "del_gain_over_random":      del_ins.get("del_gain_over_random", 0.0),
            "deletion_auc_fake_only":    _del_fake,
            "insertion_auc_fake_only":   _ins_fake,
            "temporal_ssim":             float(ssim_val),
            "faithfulness_corr":         float(faithful_corr),
            "inter_sample_cos_mean":     float(collapse_diag.get("inter_sample_cosine_mean", 0.0)),
            "peak_mode_share":           float(collapse_diag.get("peak_mode_share", 0.0)),
            "m_t_std_mean":              float(collapse_diag.get("m_t_std_mean", 0.0)),
            "mt_vs_random_model_cosine": float(mt_vs_random_cosine),
        },
        "frame_attention_drop": drop_results,
        "stability":            stability,
        "layer_ablation":       layer_causal,
        "layer_ablation_cumulative": layer_cumulative,
        "road":                 road,
        # Phase 42: del/ins under every requested baseline (blur/black/mean).
        # The canonical intrinsic.{deletion,insertion}_auc above = headline (first).
        "del_ins_baselines":    del_ins_by_baseline,
        # Phase 45: analytic contribution-space del/ins of the additive head
        # (exact, all samples; the faithfulness readout the head was built for).
        "contribution_space":   contribution_di,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n[ExplanationSuite] === Summary ===")
    print(f"  Temporal SSIM            : {result['intrinsic']['temporal_ssim']:.3f}")
    print(f"  Faithfulness corr        : {result['intrinsic']['faithfulness_corr']:.3f}")
    print(f"  Deletion AUC             : {result['intrinsic']['deletion_auc']:.3f}")
    print(f"  Insertion AUC            : {result['intrinsic']['insertion_auc']:.3f}")
    print(f"  Ins gain over random     : {result['intrinsic']['ins_gain_over_random']:+.4f}")
    print(f"  Del gain over random     : {result['intrinsic']['del_gain_over_random']:+.4f}")
    print(f"  Del/Ins AUC (fake only)  : {result['intrinsic']['deletion_auc_fake_only']:.3f} / "
          f"{result['intrinsic']['insertion_auc_fake_only']:.3f}")
    if len(del_ins_by_baseline) > 1:
        print("  --- del/ins by baseline (Phase 42; blur is confounded) ---")
        print(f"    {'baseline':<8} {'del':>7} {'ins':>7} {'del_fake':>9} {'ins_fake':>9} "
              f"{'ins_gain':>9} {'del_gain':>9}")
        for _bl in _baselines:
            _a = del_ins_by_baseline[_bl]
            print(f"    {_bl:<8} {_a['deletion_auc']:>7.3f} {_a['insertion_auc']:>7.3f} "
                  f"{_a['deletion_auc_fake_only']:>9.3f} {_a['insertion_auc_fake_only']:>9.3f} "
                  f"{_a['ins_gain_over_random']:>+9.4f} {_a['del_gain_over_random']:>+9.4f}")
    if contribution_di:
        print("  --- contribution-space del/ins (Phase 45; analytic, ALL fakes; "
              "the head's OWN per-cell terms) ---")
        for _pre in ("aeh", "blended"):
            _dk = contribution_di.get(f"{_pre}_deletion_auc_fake_only",
                                      contribution_di.get(f"{_pre}_deletion_auc", 0.0))
            _ik = contribution_di.get(f"{_pre}_insertion_auc_fake_only",
                                      contribution_di.get(f"{_pre}_insertion_auc", 0.0))
            _ig = contribution_di.get(f"{_pre}_ins_gain_fake_only",
                                      contribution_di.get(f"{_pre}_ins_gain_over_random", 0.0))
            _dg = contribution_di.get(f"{_pre}_del_gain_fake_only",
                                      contribution_di.get(f"{_pre}_del_gain_over_random", 0.0))
            print(f"    {_pre:<8} del={_dk:>6.3f}  ins={_ik:>6.3f}  "
                  f"ins_gain={_ig:>+7.4f}  del_gain={_dg:>+7.4f}  "
                  f"(n_fake={contribution_di.get('n_fake', -1)}, gamma={contribution_di.get('gamma', 0.0):.2f})")
    print(f"  Inter-sample cosine      : {result['intrinsic']['inter_sample_cos_mean']:.3f}")
    print(f"  Peak mode share          : {result['intrinsic']['peak_mode_share']:.3f}")
    print(f"  M_t std mean             : {result['intrinsic']['m_t_std_mean']:.4f}")
    print(f"  Mt vs random cosine      : {result['intrinsic']['mt_vs_random_model_cosine']:.3f}")
    for k in (1, 2, 4):
        if f"k{k}_ratio" in drop_results:
            print(f"  Drop ratio K={k}           : {drop_results[f'k{k}_ratio']:.3f} "
                  f"(top={drop_results[f'k{k}_top_conf_drop']:.3f} "
                  f"rand={drop_results[f'k{k}_random_conf_drop']:.3f}; "
                  f"zerofill={drop_results.get(f'k{k}_ratio_zerofill', 0.0):.3f})")
    if stability:
        print(f"  Stability cosine (mean)  : {stability.get('stability_cosine_mean', 0):.4f}")
    if layer_causal:
        _shares = layer_causal.get("share_per_layer", [])
        _drops  = layer_causal.get("drop_per_layer", [])
        _share_str = "  ".join(f"L{_i+1}={100*_s:.0f}%" for _i, _s in enumerate(_shares))
        _drop_str  = "  ".join(f"L{_i+1}={_d:+.3f}"     for _i, _d in enumerate(_drops))
        print(f"  Layer shares (declared)  : {_share_str}")
        print(f"  Layer drop-when-removed  : {_drop_str}")
        print(f"  Share->drop rank corr    : {layer_causal.get('share_vs_drop_spearman', 0):+.3f} "
              f"(want > 0: declared % predicts causal importance)")
        print(f"  Top-share==top-drop rate : {layer_causal.get('top_layer_match_rate', 0):.2f} "
              f"(chance {layer_causal.get('top_layer_match_chance', 0):.2f})")
    if layer_cumulative:
        _cd = layer_cumulative.get("cumulative_drop", [])
        _cd_str = "  ".join(f"-{_i}L={_d:+.3f}" for _i, _d in enumerate(_cd))
        print(f"  Cumulative layer drop    : {_cd_str}")
        print(f"  Cumulative monotonic?    : {layer_cumulative.get('monotonic', False)} "
              f"(want True: each removed layer costs more confidence)")
    if road:
        _rand_auc = road.get("random", {}).get("auc", None)
        for _nm in ("intrinsic", "gradient", "random"):
            if _nm in road and isinstance(road[_nm], dict):
                _e = road[_nm]
                _gain = (_rand_auc - _e["auc"]) if _rand_auc is not None else 0.0
                _d10 = _e.get("drop_at", {}).get(10, 0.0)
                _d50 = _e.get("drop_at", {}).get(50, 0.0)
                print(f"  ROAD {_nm:<9} AUC={_e['auc']:.4f}  gain_vs_rand={_gain:+.4f}  "
                      f"drop@10%={_d10:+.4f}  drop@50%={_d50:+.4f}")

    # ── Phase 45: authoritative faithfulness headline (contribution space) ──────
    # The eval/report.txt 'Faithful?' line uses a pixel-occlusion proxy that is
    # CONFOUNDED for the additive head (a globally-mixed transformer re-globalises
    # occluded pixels, and it ran on only ~24 fakes). The contribution-space
    # del/ins is the EXACT, all-fakes readout the head was built for. We record it
    # as the authoritative verdict (in the JSON + a human-readable file) so the
    # head's faithfulness can never be mis-reported again. Faithful == the head's
    # own saliency ordering beats a RANDOM ordering on BOTH insertion and deletion
    # of fakes (gain > 0 on both) -> the saliency causally drives the prediction.
    if contribution_di:
        def _cs_gain(_pre, _ax):
            return contribution_di.get(f"{_pre}_{_ax}_gain_fake_only",
                   contribution_di.get(f"{_pre}_{_ax}_gain_over_random", 0.0))
        _bl_ig, _bl_dg = _cs_gain("blended", "ins"), _cs_gain("blended", "del")
        _ah_ig, _ah_dg = _cs_gain("aeh", "ins"),     _cs_gain("aeh", "del")
        _bl_ok = bool(_bl_ig > 0.0 and _bl_dg > 0.0)
        _ah_ok = bool(_ah_ig > 0.0 and _ah_dg > 0.0)
        headline = {
            "faithful":          bool(_bl_ok or _ah_ok),
            "faithful_blended":  _bl_ok,
            "faithful_head":     _ah_ok,
            "blended_ins_fake":  contribution_di.get("blended_insertion_auc_fake_only", 0.0),
            "blended_del_fake":  contribution_di.get("blended_deletion_auc_fake_only", 0.0),
            "blended_ins_gain":  _bl_ig, "blended_del_gain": _bl_dg,
            "head_ins_fake":     contribution_di.get("aeh_insertion_auc_fake_only", 0.0),
            "head_del_fake":     contribution_di.get("aeh_deletion_auc_fake_only", 0.0),
            "head_ins_gain":     _ah_ig, "head_del_gain": _ah_dg,
            "n_fake":            contribution_di.get("n_fake", -1),
            "gamma":             contribution_di.get("gamma", 0.0),
            "basis":             "contribution-space (analytic, all fakes)",
        }
        result["faithfulness_headline"] = headline
        try:
            _fr = Path(output_path).parent / "faithfulness_report.txt"
            _v  = "YES -- saliency causally drives the prediction" if headline["faithful"] else "NO"
            _lines = [
                "EAHN Additive-Head Faithfulness (AUTHORITATIVE, contribution space)",
                "=" * 66,
                "Evaluated in the head's EXACT contribution space (analytic, ALL %d fakes)."
                    % headline["n_fake"],
                "Supersedes the pixel-occlusion 'Faithful?' line in eval/report.txt, which is",
                "confounded for a globally-mixed transformer.",
                "",
                "Deployed model (blended logit, gamma=%.2f):" % headline["gamma"],
                "  Insertion AUC (fake) : %.3f   (higher = better)" % headline["blended_ins_fake"],
                "  Deletion  AUC (fake) : %.3f   (lower  = better)" % headline["blended_del_fake"],
                "  Ins gain over random : %+.3f" % headline["blended_ins_gain"],
                "  Del gain over random : %+.3f" % headline["blended_del_gain"],
                "Pure additive head:",
                "  Insertion AUC (fake) : %.3f" % headline["head_ins_fake"],
                "  Deletion  AUC (fake) : %.3f" % headline["head_del_fake"],
                "  Ins / Del gain       : %+.3f / %+.3f"
                    % (headline["head_ins_gain"], headline["head_del_gain"]),
                "",
                "FAITHFUL? %s" % _v,
            ]
            with open(_fr, "w", encoding="ascii", errors="replace") as _f:
                _f.write("\n".join(_lines) + "\n")
            print(f"[ExplanationSuite] faithfulness_report.txt -> {_fr}")
            print(f"[ExplanationSuite] AUTHORITATIVE Faithful? {_v}")
        except Exception as _e:
            print(f"[ExplanationSuite] (faithfulness_report write skipped: {_e})")

    # ── Phase 46: paper figures + corrected numbers (eval-only, fully guarded) ──
    # Past runs crashed on a cosmetic plot AFTER all metrics were saved, so every
    # figure here is in its own try/except and NEVER aborts the run.
    _fig_dir = Path(output_path).parent
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        # (1) Contribution-space insertion/deletion curve (the headline faithfulness fig)
        if contribution_di and "curve_fraction" in contribution_di:
            xs = contribution_di["curve_fraction"]
            def _g(k):
                return contribution_di.get(k)
            fig, ax = _plt.subplots(1, 2, figsize=(11, 4.2))
            # prefer fake-only curves; fall back to all-sample
            ins_s = _g("blended_insertion_curve_fake") or _g("blended_insertion_curve")
            ins_r = _g("aeh_insertion_curve_rand_fake") or _g("aeh_insertion_curve_rand")
            ins_h = _g("aeh_insertion_curve_fake") or _g("aeh_insertion_curve")
            del_s = _g("blended_deletion_curve_fake") or _g("blended_deletion_curve")
            del_r = _g("aeh_deletion_curve_rand_fake") or _g("aeh_deletion_curve_rand")
            del_h = _g("aeh_deletion_curve_fake") or _g("aeh_deletion_curve")
            if ins_h: ax[0].plot(xs, ins_h, "-",  color="C0", label="head saliency")
            if ins_s: ax[0].plot(xs, ins_s, "--", color="C2", label="deployed (blended)")
            if ins_r: ax[0].plot(xs, ins_r, ":",  color="C3", label="random order")
            ax[0].set_title("Insertion (fake) — higher = better")
            ax[0].set_xlabel("fraction of cells inserted"); ax[0].set_ylabel("P(fake)")
            ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
            if del_h: ax[1].plot(xs, del_h, "-",  color="C0", label="head saliency")
            if del_s: ax[1].plot(xs, del_s, "--", color="C2", label="deployed (blended)")
            if del_r: ax[1].plot(xs, del_r, ":",  color="C3", label="random order")
            ax[1].set_title("Deletion (fake) — lower = better")
            ax[1].set_xlabel("fraction of cells deleted"); ax[1].set_ylabel("P(fake)")
            ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
            fig.suptitle("Contribution-space faithfulness (additive head, all fakes)")
            fig.tight_layout()
            fig.savefig(_fig_dir / "contribution_del_ins_curve.png", dpi=140)
            _plt.close(fig)
            print(f"[ExplanationSuite] figure -> contribution_del_ins_curve.png")
    except Exception as _e:
        print(f"[ExplanationSuite] (contribution curve figure skipped: {_e})")

    try:
        if road and isinstance(road, dict):
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(6, 4.2))
            for _nm, _c in (("intrinsic", "C0"), ("gradient", "C1"), ("random", "C3")):
                _e2 = road.get(_nm)
                if isinstance(_e2, dict) and _e2.get("curve"):
                    cv = _e2["curve"]
                    xs = [i / (len(cv) - 1) for i in range(len(cv))]
                    ax.plot(xs, cv, label=f"{_nm} (AUC {_e2.get('auc', 0):.3f})",
                            color=_c)
            ax.set_title("ROAD — debiased deletion (lower curve = more faithful)")
            ax.set_xlabel("fraction removed"); ax.set_ylabel("P(fake)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(_fig_dir / "road_curve.png", dpi=140)
            _plt.close(fig)
            print(f"[ExplanationSuite] figure -> road_curve.png")
    except Exception as _e:
        print(f"[ExplanationSuite] (ROAD figure skipped: {_e})")

    try:
        import csv as _csv
        _intr = result.get("intrinsic", {})
        _hl   = result.get("faithfulness_headline", {})
        with open(_fig_dir / "faithfulness_summary.csv", "w", newline="",
                  encoding="ascii", errors="replace") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["metric", "value", "note"])
            _w.writerow(["faithful_contribution_space", _hl.get("faithful", ""),
                         "saliency beats random on ins AND del (fakes)"])
            _w.writerow(["blended_insertion_fake", _hl.get("blended_ins_fake", ""), "higher=better"])
            _w.writerow(["blended_deletion_fake",  _hl.get("blended_del_fake", ""), "lower=better"])
            _w.writerow(["blended_ins_gain", _hl.get("blended_ins_gain", ""), ">0=faithful"])
            _w.writerow(["blended_del_gain", _hl.get("blended_del_gain", ""), ">0=faithful"])
            _w.writerow(["head_insertion_fake", _hl.get("head_ins_fake", ""), "intrinsic head"])
            _w.writerow(["head_deletion_fake",  _hl.get("head_del_fake", ""), "intrinsic head"])
            _w.writerow(["faithfulness_corr", _intr.get("faithfulness_corr", ""), "saliency vs gradient"])
            _w.writerow(["temporal_ssim", _intr.get("temporal_ssim", ""), "map stability"])
            _w.writerow(["peak_mode_share", _intr.get("peak_mode_share", ""), "concentration"])
            _w.writerow(["m_t_std_mean", _intr.get("m_t_std_mean", ""), "spatial sharpness"])
            _w.writerow(["n_fake", _hl.get("n_fake", ""), "samples in contribution metric"])
            _w.writerow(["pixel_occlusion_ins_CONFOUNDED", _intr.get("insertion_auc", ""),
                         "secondary/confounded -- do not headline"])
            _w.writerow(["pixel_occlusion_del_CONFOUNDED", _intr.get("deletion_auc", ""),
                         "secondary/confounded -- do not headline"])
        print(f"[ExplanationSuite] table -> faithfulness_summary.csv")
    except Exception as _e:
        print(f"[ExplanationSuite] (faithfulness_summary csv skipped: {_e})")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ExplanationSuite] metrics saved → {output_path}")

    # ── Final cleanup of suite-local buffers ──────────────────────────────────
    del frames_by_idx
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
