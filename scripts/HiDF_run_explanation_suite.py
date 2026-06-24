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
