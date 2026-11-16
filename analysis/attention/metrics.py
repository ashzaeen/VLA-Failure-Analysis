"""Attention metric + extraction helpers.

RECONSTRUCTED 2026-06-24. Logic is lifted verbatim from
src/utils/pi0_attention_utils.py, except:
  - _build_heatmap_overlay takes an `image_name` and returns keys SUFFIXED with
    "(<image_name>)", because scripts/analyze_attention.py does row.update(overlay)
    and later reads row["max_value(base_img)"], row.pop("overlay_bgr(base_img)"), etc.
  - get_object_names is added (parses the saved task_description; falls back to a
    task_id map).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def _to_numpy_image(image: ArrayLike) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    return image


# --------------------------------------------------------------------------- #
# scalar metrics
# --------------------------------------------------------------------------- #
def _normalize_scores(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    total = scores.sum()
    if total <= eps:
        return np.full_like(scores, 1.0 / len(scores))
    return scores / total


def attention_entropy(scores: np.ndarray) -> float:
    p = _normalize_scores(scores)
    return float(-(p * np.log(p + 1e-12)).sum())


def attention_gini(scores: np.ndarray) -> float:
    x = np.asarray(scores, dtype=np.float64)
    if np.allclose(x.sum(), 0):
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumx = np.cumsum(x_sorted)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)


def topk_patches(scores: np.ndarray, grid_size: int, k: int = 5) -> List[Dict[str, Any]]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    k = min(k, len(scores))
    top_idx = np.argsort(scores)[-k:][::-1]
    total = float(scores.sum()) if scores.sum() > 0 else 1.0
    items: List[Dict[str, Any]] = []
    for rank, patch_idx in enumerate(top_idx, start=1):
        row = int(patch_idx // grid_size)
        col = int(patch_idx % grid_size)
        value = float(scores[patch_idx])
        items.append(
            {
                "rank": rank,
                "patch_idx": int(patch_idx),
                "row": row,
                "col": col,
                "value": value,
                "share": value / total,
            }
        )
    return items


# --------------------------------------------------------------------------- #
# attention extraction
# --------------------------------------------------------------------------- #
def _extract_patch_attention(
    action_attentions: Sequence[torch.Tensor],
    image_token_slice: slice,
    layer_idx: int = -1,
    action_token_idx: int = -1,
    head_idx: Optional[int] = None,
    batch_idx: int = 0,
) -> Dict[str, Any]:
    """action_attentions[layer] is assumed to be [B, H, Q, K].

    Returns patch_scores [P], all_action_patch_attn [Q, P], mean_action_attention [Q, K].
    """
    layer_attn = action_attentions[layer_idx].detach().to(torch.float32)
    batch_attn = layer_attn[batch_idx]  # [H, Q, K]

    if head_idx is None:
        head_attn = batch_attn.mean(dim=0)  # [Q, K]
    else:
        head_attn = batch_attn[head_idx]  # [Q, K]

    all_action_patch_attn_t = head_attn[:, image_token_slice]  # [Q, P]

    if action_token_idx == -1:
        patch_scores_t = all_action_patch_attn_t.mean(dim=0)  # [P]
    else:
        patch_scores_t = all_action_patch_attn_t[action_token_idx]  # [P]

    return {
        "patch_scores": patch_scores_t.detach().cpu().numpy().astype(np.float64),
        "all_action_patch_attn": all_action_patch_attn_t.detach().cpu().numpy().astype(np.float64),
        "mean_action_attention": head_attn.detach().cpu().numpy().astype(np.float64),
    }


def _build_heatmap_overlay(
    image: np.ndarray,
    patch_scores: np.ndarray,
    alpha: float = 0.3,
    patch_grid_size: Optional[int] = None,
    image_name: str = "img",
) -> Dict[str, Any]:
    """Build a JET heatmap overlay; return keys suffixed with "(<image_name>)".

    The suffixed keys match what scripts/analyze_attention.py reads back, and the
    None-branch it falls back to when an image is missing.
    """
    image = _to_numpy_image(image)
    num_patches = int(len(patch_scores))
    grid_size = patch_grid_size or int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"num_patches={num_patches} is not square; set patch_grid_size explicitly."
        )

    patch_grid = np.asarray(patch_scores, dtype=np.float64).reshape(grid_size, grid_size)
    h, w = image.shape[:2]
    heatmap = cv2.resize(patch_grid, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0).copy()

    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    cv2.circle(overlay, (int(max_x), int(max_y)), radius=12, color=(0, 255, 0), thickness=2)
    cv2.circle(overlay, (int(max_x), int(max_y)), radius=3, color=(255, 255, 255), thickness=-1)

    patch_y, patch_x = np.unravel_index(np.argmax(patch_grid), patch_grid.shape)
    return {
        f"overlay_bgr({image_name})": overlay,
        f"heatmap({image_name})": heatmap,
        f"max_point_xy({image_name})": (int(max_x), int(max_y)),
        f"max_patch_rowcol({image_name})": (int(patch_y), int(patch_x)),
        f"max_value({image_name})": float(np.max(patch_scores)),
        f"grid_size({image_name})": grid_size,
    }


# --------------------------------------------------------------------------- #
# object-name resolution
# --------------------------------------------------------------------------- #
def get_object_names(record: Dict[str, Any], task_to_object: Optional[Dict[int, str]] = None) -> str:
    """Resolve the target object for an episode record.

    Primary: parse the saved task_description ("pick up the <obj> and place ...").
    Fallback: task_id -> object map.
    """
    td = record.get("task_description")
    if td is not None:
        s = str(np.asarray(td).item()) if isinstance(td, np.ndarray) and td.shape == () else str(td)
        m = re.search(r"pick up (?:the )?(.+?) and place it", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        if s and s.lower() != "none":
            return s.strip()

    if task_to_object is not None:
        tid = record.get("task_id")
        try:
            tid = int(np.asarray(tid))
        except Exception:
            pass
        if tid in task_to_object:
            return task_to_object[tid]

    return "unknown"
