#!/usr/bin/env python
"""Dump the normalization stats the pi0 processor pipeline actually loads.

WHY: baseline libero_object success is 10/100 (tall bottles pass, flat objects
fail) — the pattern of a broken/identity state+action normalization. State norm
(STATE->MEAN_STD) and action un-norm (ACTION->MEAN_STD) are loaded from the
checkpoint's saved policy_preprocessor.json / policy_postprocessor.json via
make_pre_post_processors(..., model_id) -> from_pretrained branch. If those stats
are empty/placeholder (mean~0, std~1) the normalization silently no-ops.

This mirrors src/models/pi0/pi0_model.py's processor construction but loads ONLY
the config (not the 3B weights), so it runs in seconds.

Run ON BENDER, from the repo root, with the pi0 env active:
    PYTHONPATH="$(pwd)" python scripts/diagnose_norm_stats.py \
        --checkpoint ./pi0_libero_finetuned_v044

(Defaults match configs/server/pi0_libero_finetuned_v044.yaml.)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


def _as_np(x) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def _summarize(name: str, arr: np.ndarray | None) -> str:
    if arr is None:
        return f"    {name}: <missing>"
    flat = arr.reshape(-1)
    preview = np.array2string(flat[:8], precision=4, suppress_small=True)
    tail = " ..." if flat.size > 8 else ""
    finite = np.isfinite(flat).all()
    return (
        f"    {name}: shape={tuple(arr.shape)} finite={finite} "
        f"min={np.nanmin(flat):.4f} max={np.nanmax(flat):.4f}\n"
        f"        values={preview}{tail}"
    )


def _looks_trivial(stats_for_key: dict) -> str | None:
    """Return a warning string if these stats would make normalization a no-op."""
    mean = _as_np(stats_for_key.get("mean"))
    std = _as_np(stats_for_key.get("std"))
    lo = _as_np(stats_for_key.get("min"))
    hi = _as_np(stats_for_key.get("max"))

    if mean is not None and std is not None:
        if np.allclose(mean, 0.0, atol=1e-6) and np.allclose(std, 1.0, atol=1e-6):
            return "MEAN~0 & STD~1  -> normalization is effectively IDENTITY (placeholder stats?)"
        if not np.isfinite(std).all() or np.any(np.abs(std) < 1e-8):
            return "STD has zeros/NaNs -> division blows up or no-ops"
    if lo is not None and hi is not None:
        if np.allclose(lo, hi):
            return "MIN == MAX -> min/max normalization is degenerate"
    if mean is None and std is None and lo is None and hi is None:
        return "no mean/std or min/max present -> nothing to normalize with"
    return None


def _dump_pipeline(title: str, pipeline) -> int:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    n_warn = 0
    steps = list(getattr(pipeline, "steps", []))
    print(f"steps: {[s.__class__.__name__ for s in steps]}")
    for step in steps:
        cls = step.__class__.__name__
        if "Normaliz" not in cls:
            continue
        norm_map = getattr(step, "norm_map", {}) or {}
        print(f"\n[{cls}]  norm_map={{"
              + ", ".join(f"{k}: {v}" for k, v in norm_map.items()) + "}")
        stats = getattr(step, "stats", None) or {}
        if not stats:
            # fall back to the tensor copy if the numpy dict wasn't populated
            tstats = getattr(step, "_tensor_stats", None) or {}
            stats = {k: {sk: _as_np(sv) for sk, sv in v.items()} for k, v in tstats.items()}
        if not stats:
            print("  !! no stats on this step at all")
            n_warn += 1
            continue
        for key, sd in stats.items():
            print(f"  key: {key}")
            for stat_name in ("mean", "std", "min", "max"):
                if stat_name in sd:
                    print(_summarize(stat_name, _as_np(sd[stat_name])))
            warn = _looks_trivial(sd)
            if warn:
                print(f"    >>> WARNING: {warn}")
                n_warn += 1
    return n_warn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./pi0_libero_finetuned_v044",
                    help="local pi0 checkpoint dir (== model_id in the server config)")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    print(f"Loading PI0Config from: {args.checkpoint}")
    # Use the base PreTrainedConfig loader: it dispatches on the config.json "type"
    # discriminator via the policy registry and pops it, whereas the concrete
    # PI0Config.from_pretrained rejects "type" with a draccus DecodingError.
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    print(f"normalization_mapping = {cfg.normalization_mapping}")

    # Same call as src/models/pi0/pi0_model.py (model_id == checkpoint dir).
    preprocessor_overrides = {"device_processor": {"device": "cpu"}}
    tokenizer_override = os.environ.get("PI0_TOKENIZER_PATH")
    if tokenizer_override:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_override}
        print(f"Overriding tokenizer_name -> {tokenizer_override}")
    preprocess, postprocess = make_pre_post_processors(
        cfg,
        args.checkpoint,
        preprocessor_overrides=preprocessor_overrides,
    )

    n_warn = 0
    n_warn += _dump_pipeline("PRE-PROCESSOR (input: normalizes STATE)", preprocess)
    n_warn += _dump_pipeline("POST-PROCESSOR (output: un-normalizes ACTION)", postprocess)

    print(f"\n{'=' * 70}")
    if n_warn:
        print(f"RESULT: {n_warn} warning(s) -> normalization is likely BROKEN. "
              f"This explains the 10/100 grasp-precision gap.")
    else:
        print("RESULT: stats look real (non-trivial mean/std). "
              "Normalization is NOT the culprit; look at replan cadence / decoding next.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
