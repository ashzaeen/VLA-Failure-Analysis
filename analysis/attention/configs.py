"""Configuration objects for attention analysis.

RECONSTRUCTED 2026-06-24: analysis/attention/ was empty, but scripts/analyze_attention.py
imports from it. These mirror the originals in src/utils/pi0_attention_utils.py so the
two stay behaviourally identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ImageAttentionConfig:
    """One camera view and the key-token slice that holds its image patches.

    pi0 (PaliGemma backbone) lays out 256 image tokens per view at the start of
    the key sequence: base_img -> [0:256], wrist_img -> [256:512].
    """
    name: str
    token_slice: slice


@dataclass
class AnalysisConfig:
    layer_idx: int = -1               # which transformer layer (-1 = last)
    action_token_idx: int = -1        # -1 => average over all action (query) rows
    head_idx: Optional[int] = None    # None => average over all heads
    alpha: float = 0.3                # heatmap overlay opacity
    top_k: int = 5
    patch_grid_size: Optional[int] = None  # None => infer sqrt(num_tokens)
    image_configs: Tuple[ImageAttentionConfig, ...] = (
        ImageAttentionConfig("base_img", slice(0, 256)),
        ImageAttentionConfig("wrist_img", slice(256, 512)),
    )
    save_overlay_figures: bool = False
    save_mean_summary_figure: bool = True
    save_cluster_summary_figures: bool = True
    add_missing_figure_paths_to_record: bool = False
