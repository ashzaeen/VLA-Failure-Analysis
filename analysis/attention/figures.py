"""Composite attention figure -> RGB video frame.

RECONSTRUCTED 2026-06-24. scripts/analyze_attention.py collects the return value
of get_attention_overlay_figure into a list and feeds it to
src.utils.video_utils.save_heatmap_video, which appends each item via imageio.
So this MUST return an HxWx3 uint8 RGB array (a single video frame), not a Figure.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: no display on the server
import matplotlib.pyplot as plt

from src.utils.video_utils import fig_to_frame


def _bgr_overlay_to_rgb(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_attention_overlay_figure(
    mean_action_attention: Optional[np.ndarray] = None,
    base_img: Optional[np.ndarray] = None,
    wrist_img: Optional[np.ndarray] = None,
    base_max: Optional[float] = None,
    wrist_max: Optional[float] = None,
    mode: str = "mean",
    top_k: int = 5,
    sample_id: Any = None,
    object_name: Optional[str] = None,
    figsize=(12, 8),
    dpi: int = 110,
) -> np.ndarray:
    """Compose: top = mean action attention matrix, bottom = base/wrist overlays.

    Returns an RGB uint8 frame (fixed size for a given figsize/dpi so frames stack
    cleanly into an mp4).
    """
    base_rgb = _bgr_overlay_to_rgb(base_img)
    wrist_rgb = _bgr_overlay_to_rgb(wrist_img)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[0.30, 1.0], width_ratios=[1, 1], hspace=0.22, wspace=0.10
    )
    ax_top = fig.add_subplot(gs[0, :])
    ax_base = fig.add_subplot(gs[1, 0])
    ax_wrist = fig.add_subplot(gs[1, 1])

    if mean_action_attention is not None:
        maa = np.asarray(mean_action_attention, dtype=np.float64)
        im = ax_top.imshow(maa, cmap="turbo", aspect="auto")
        fig.colorbar(im, ax=ax_top, fraction=0.025, pad=0.02)
    ax_top.set_title(
        f"Mean Action Attention | idx:{sample_id} | obj:{object_name}",
        fontsize=10, fontweight="bold",
    )
    ax_top.set_xlabel("Key Tokens (K)")
    ax_top.set_ylabel("Action Tokens (Q)")

    title_suffix = "Mean Heatmap" if mode == "mean" else f"Top-{top_k} Overlay"

    if base_rgb is not None:
        ax_base.imshow(base_rgb)
    bmax = f"{base_max:.4f}" if base_max is not None else "n/a"
    ax_base.set_title(f"Base {title_suffix} | max:{bmax}", fontsize=10, fontweight="bold")
    ax_base.axis("off")

    if wrist_rgb is not None:
        ax_wrist.imshow(wrist_rgb)
    wmax = f"{wrist_max:.4f}" if wrist_max is not None else "n/a"
    ax_wrist.set_title(f"Wrist {title_suffix} | max:{wmax}", fontsize=10, fontweight="bold")
    ax_wrist.axis("off")

    frame = fig_to_frame(fig)
    plt.close(fig)
    return frame
