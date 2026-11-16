from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ArrayLike = np.ndarray | torch.Tensor


@dataclass
class ImageAttentionConfig:
    name: str
    token_slice: slice


@dataclass
class AnalysisConfig:
    layer_idx: int = -1
    action_token_idx: int = -1  # -1 => average over all action rows
    head_idx: Optional[int] = None  # None => average over all heads
    alpha: float = 0.3
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

from sklearn.cluster import KMeans


def _compute_mean_action_attention_map(
    action_attentions: Sequence[torch.Tensor],
    layer_idx: int = -1,
    batch_idx: int = 0,
) -> np.ndarray:
    """
    Return mean action attention map with shape [Q, K].
    action_attentions[layer]: [B, H, Q, K]
    """
    layer_attn = action_attentions[layer_idx].detach().to(torch.float32)
    mean_action_attention = layer_attn.mean(dim=1)[batch_idx]  # [Q, K]
    return mean_action_attention.cpu().numpy().astype(np.float64)


def _overlay_from_patch_scores(
    image: ArrayLike,
    patch_scores: np.ndarray,
    grid_size: int,
    alpha: float = 0.35,
    circle_max: bool = True,
) -> Dict[str, Any]:
    """
    Like _build_heatmap_overlay, but explicitly keeps the default red-style JET heatmap.
    """
    image_np = _to_numpy_image(image)
    patch_grid = np.asarray(patch_scores, dtype=np.float64).reshape(grid_size, grid_size)

    h, w = image_np.shape[:2]
    heatmap = cv2.resize(patch_grid, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # 红黄风格
    overlay_rgb = cv2.addWeighted(image_np, 1.0 - alpha, colored_bgr, alpha, 0)

    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    if circle_max:
        cv2.circle(overlay_rgb, (int(max_x), int(max_y)), radius=12, color=(0, 255, 0), thickness=2)
        cv2.circle(overlay_rgb, (int(max_x), int(max_y)), radius=3, color=(255, 255, 255), thickness=-1)

    patch_y, patch_x = np.unravel_index(np.argmax(patch_grid), patch_grid.shape)

    return {
        "overlay_rgb": overlay_rgb,
        # "overlay_rgb": cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        "heatmap": heatmap,
        "max_point_xy": (int(max_x), int(max_y)),
        "max_patch_rowcol": (int(patch_y), int(patch_x)),
        "max_value": float(np.max(patch_scores)),
        "min_value": float(np.min(patch_scores)),
        "grid_size": grid_size,
    }


def _cluster_action_tokens(
    all_action_patch_attn: np.ndarray,
    n_clusters: int = 4,
) -> Dict[str, Any]:
    """
    all_action_patch_attn: [Q, P]
    """
    x = np.asarray(all_action_patch_attn, dtype=np.float64)
    num_actions = x.shape[0]
    if num_actions == 0:
        raise ValueError("No action tokens found.")

    n_clusters = min(n_clusters, num_actions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(x)

    results = []
    for cluster_id in np.unique(clusters):
        token_indices = np.where(clusters == cluster_id)[0]
        cluster_attention = x[token_indices].mean(axis=0)  # [P]
        results.append(
            {
                "cluster_id": int(cluster_id),
                "token_indices": token_indices.tolist(),
                "num_tokens": int(len(token_indices)),
                "cluster_attention": cluster_attention,
                "max_value": float(cluster_attention.max()),
                "min_value": float(cluster_attention.min()),
            }
        )

    # 按 cluster max 从大到小排，方便看
    results = sorted(results, key=lambda z: z["max_value"], reverse=True)
    return {
        "clusters": results,
        "labels": clusters,
    }


def save_mean_summary_figure(
    *,
    action_attentions: Sequence[torch.Tensor],
    
    base_image: ArrayLike,
    wrist_image: ArrayLike,
    base_patch_scores: np.ndarray,
    wrist_patch_scores: np.ndarray,
    output_path: str | Path,
    sample_id: str | int,
    object_name: str,
    layer_idx: int = -1,
    alpha: float = 0.35,
    grid_size: int = 16,
    dpi: int = 180,
    mean_action_attention: np.ndarray=None,
) -> str:
    """
    图1:
      - 顶部: mean action attention map + colorbar
      - 第二行左: base mean heatmap
      - 第二行右: wrist mean heatmap

    要求：
      - 每个 subplot 自己有标题
      - 顶部 attention map 高度较小
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mean_action_attention is None:
        mean_action_attention = _compute_mean_action_attention_map(
            action_attentions=action_attentions,
            layer_idx=layer_idx,
        )
    mean_max = float(mean_action_attention.max())
    mean_min = float(mean_action_attention.min())

    base_overlay = _overlay_from_patch_scores(
        image=base_image,
        patch_scores=base_patch_scores,
        grid_size=grid_size,
        alpha=alpha,
        circle_max=True,
    )
    wrist_overlay = _overlay_from_patch_scores(
        image=wrist_image,
        patch_scores=wrist_patch_scores,
        grid_size=grid_size,
        alpha=alpha,
        circle_max=True,
    )

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.25, 1.0],   # 顶部压缩成约 1/4
        width_ratios=[1, 1],
        hspace=0.22,
        wspace=0.12,
    )

    ax_top = fig.add_subplot(gs[0, :])
    ax_base = fig.add_subplot(gs[1, 0])
    ax_wrist = fig.add_subplot(gs[1, 1])

    im = ax_top.imshow(mean_action_attention, cmap="turbo", aspect="auto")
    ax_top.set_title(
        f"Mean Action Attention | idx:{sample_id} | obj:{object_name} | "
        f"max:{mean_max:.4f} | min:{mean_min:.4f}",
        fontsize=10,
        fontweight="bold",
    )
    ax_top.set_xlabel("Key Tokens")
    ax_top.set_ylabel("Action Tokens")
    cbar = fig.colorbar(im, ax=ax_top, fraction=0.025, pad=0.02)
    cbar.set_label("Attention", fontsize=9)

    ax_base.imshow(base_overlay["overlay_rgb"])
    ax_base.set_title(
        f"Base Mean Heatmap | max:{base_overlay['max_value']:.4f}",
        fontsize=10,
        fontweight="bold",
    )
    ax_base.axis("off")

    ax_wrist.imshow(wrist_overlay["overlay_rgb"])
    ax_wrist.set_title(
        f"Wrist Mean Heatmap | max:{wrist_overlay['max_value']:.4f}",
        fontsize=10,
        fontweight="bold",
    )
    ax_wrist.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_topk_cluster_summary_figure(
    *,
    action_attentions: Sequence[torch.Tensor],
    base_image: ArrayLike,
    base_all_action_patch_attn: np.ndarray,
    output_path: str | Path,
    sample_id: str | int,
    object_name: str,
    layer_idx: int = -1,
    alpha: float = 0.35,
    grid_size: int = 16,
    n_clusters: int = 4,
    dpi: int = 180,
    image_label: str = "Base",
) -> str:
    """
    图2:
      - 顶部: mean action attention map
      - 下方 2x2: 4 个 cluster attention overlay

    每个 subplot 自己写：
      cluster id / token 数 / max
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_action_attention = _compute_mean_action_attention_map(
        action_attentions=action_attentions,
        layer_idx=layer_idx,
    )
    mean_max = float(mean_action_attention.max())
    mean_min = float(mean_action_attention.min())

    cluster_info = _cluster_action_tokens(
        all_action_patch_attn=base_all_action_patch_attn,
        n_clusters=n_clusters,
    )
    clusters = cluster_info["clusters"]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.25, 1.0, 1.0],  # 顶部 1/4 高度
        width_ratios=[1, 1],
        hspace=0.22,
        wspace=0.12,
    )

    ax_top = fig.add_subplot(gs[0, :])
    im = ax_top.imshow(mean_action_attention, cmap="turbo", aspect="auto")
    ax_top.set_title(
        f"Mean Action Attention | idx:{sample_id} | obj:{object_name} | "
        f"max:{mean_max:.4f} | min:{mean_min:.4f}",
        fontsize=10,
        fontweight="bold",
    )
    ax_top.set_xlabel("Key Tokens")
    ax_top.set_ylabel("Action Tokens")
    cbar = fig.colorbar(im, ax=ax_top, fraction=0.025, pad=0.02)
    cbar.set_label("Attention", fontsize=9)

    subplot_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    for ax_idx, ax in enumerate(subplot_axes):
        if ax_idx < len(clusters):
            cluster = clusters[ax_idx]
            overlay_info = _overlay_from_patch_scores(
                image=base_image,
                patch_scores=cluster["cluster_attention"],
                grid_size=grid_size,
                alpha=alpha,
                circle_max=True,
            )
            ax.imshow(overlay_info["overlay_rgb"])
            ax.set_title(
                f"{image_label} Cluster {cluster['cluster_id']} "
                f"({cluster['num_tokens']} tokens) "
                f"max:{cluster['max_value']:.4f}",
                fontsize=10,
                fontweight="bold",
            )
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_dual_topk_cluster_figures(
    *,
    action_attentions: Sequence[torch.Tensor],
    base_image: ArrayLike,
    wrist_image: ArrayLike,
    base_all_action_patch_attn: np.ndarray,
    wrist_all_action_patch_attn: np.ndarray,
    out_dir: str | Path,
    sample_id: str | int,
    object_name: str,
    layer_idx: int = -1,
    alpha: float = 0.35,
    grid_size: int = 16,
    n_clusters: int = 4,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_object = str(object_name).replace(" ", "_")

    base_path = out_dir / f"cluster/{sample_id}_{safe_object}_base_cluster_summary.png"
    wrist_path = out_dir / f"cluster/{sample_id}_{safe_object}_wrist_cluster_summary.png"

    save_topk_cluster_summary_figure(
        action_attentions=action_attentions,
        base_image=base_image,
        base_all_action_patch_attn=base_all_action_patch_attn,
        output_path=base_path,
        sample_id=sample_id,
        object_name=object_name,
        layer_idx=layer_idx,
        alpha=alpha,
        grid_size=grid_size,
        n_clusters=n_clusters,
        image_label="Base",
    )

    save_topk_cluster_summary_figure(
        action_attentions=action_attentions,
        base_image=wrist_image,
        base_all_action_patch_attn=wrist_all_action_patch_attn,
        output_path=wrist_path,
        sample_id=sample_id,
        object_name=object_name,
        layer_idx=layer_idx,
        alpha=alpha,
        grid_size=grid_size,
        n_clusters=n_clusters,
        image_label="Wrist",
    )

    return {
        "base_cluster_summary": str(base_path),
        "wrist_cluster_summary": str(wrist_path),
    }


def _extract_patch_attention(
    action_attentions: Sequence[torch.Tensor],
    image_token_slice: slice,
    layer_idx: int = -1,
    action_token_idx: int = -1,
    head_idx: Optional[int] = None,
    batch_idx: int = 0,
) -> Dict[str, Any]:
    """
    action_attentions[layer] is assumed to be [B, H, Q, K].

    Returns:
        {
            "patch_scores": np.ndarray,           # [P]
            "all_action_patch_attn": np.ndarray,  # [Q, P]
            "mean_action_attention": np.ndarray,   # [Q, K]
        }
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
) -> Dict[str, Any]:
    image = _to_numpy_image(image)
    num_patches = int(len(patch_scores))
    grid_size = patch_grid_size or int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a square number; please set patch_grid_size explicitly.")

    patch_grid = patch_scores.reshape(grid_size, grid_size)
    h, w = image.shape[:2]
    heatmap = cv2.resize(patch_grid, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)

    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    overlay = overlay.copy()
    cv2.circle(overlay, (int(max_x), int(max_y)), radius=12, color=(0, 255, 0), thickness=2)
    cv2.circle(overlay, (int(max_x), int(max_y)), radius=3, color=(255, 255, 255), thickness=-1)

    patch_y, patch_x = np.unravel_index(np.argmax(patch_grid), patch_grid.shape)
    return {
        "overlay_bgr": overlay,
        "overlay_rgb": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        "heatmap": heatmap,
        "max_point_xy": (int(max_x), int(max_y)),
        "max_patch_rowcol": (int(patch_y), int(patch_x)),
        "max_value": float(np.max(patch_scores)),
        "grid_size": grid_size,
    }


def _draw_topk_boxes_on_overlay(
    overlay_bgr: np.ndarray,
    topk_items: List[Dict[str, Any]],
    grid_size: int,
) -> np.ndarray:
    out = overlay_bgr.copy()
    h, w = out.shape[:2]
    patch_h = h / grid_size
    patch_w = w / grid_size

    for item in topk_items:
        row = int(item["row"])
        col = int(item["col"])
        rank = int(item["rank"])

        x1 = int(round(col * patch_w))
        y1 = int(round(row * patch_h))
        x2 = int(round((col + 1) * patch_w))
        y2 = int(round((row + 1) * patch_h))

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            out,
            str(rank),
            (x1 + 3, y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def save_overlay_image_with_plot(
    overlay_rgb: np.ndarray,
    output_path: str | Path,
    sample_id: int | None = None,
    object_name: str | None = None,
    max_attention: float | None = None,
    topk_patches: list | None = None,
    grid_size: int | None = None,
    figsize=(5, 5),
    dpi=150,
) -> str:
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(overlay_rgb)
    ax.axis("off")

    title_parts = []
    if sample_id is not None:
        title_parts.append(f"idx: {sample_id}")
    if object_name is not None:
        title_parts.append(f"obj: {object_name}")
    if max_attention is not None:
        title_parts.append(f"max: {max_attention:.4f}")
    ax.set_title(" | ".join(title_parts), fontsize=10, fontweight="bold")

    if topk_patches is not None and grid_size is not None:
        H, W, _ = overlay_rgb.shape
        patch_h = H / grid_size
        patch_w = W / grid_size
        for item in topk_patches:
            row = item["row"]
            col = item["col"]
            rank = item["rank"]
            y = row * patch_h
            x = col * patch_w
            rect = plt.Rectangle((x, y), patch_w, patch_h, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x, y, str(rank), color="yellow", fontsize=8, weight="bold",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def save_attention_summary_figure(
    *,
    action_attentions: Sequence[torch.Tensor],
    base_image: np.ndarray,
    wrist_image: np.ndarray,
    base_patch_scores: np.ndarray,
    wrist_patch_scores: np.ndarray,
    output_path: str | Path,
    sample_id: Optional[int] = None,
    object_name: Optional[str] = None,
    layer_idx: int = -1,
    alpha: float = 0.3,
    grid_size: int = 16,
    top_k: int = 5,
    mode: str = "mean",
    dpi: int = 180,
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layer_attn = action_attentions[layer_idx].detach().to(torch.float32)
    mean_action_attention = layer_attn.mean(dim=1).squeeze(0).cpu().numpy()  # [Q, K]
    attn_max = float(mean_action_attention.max())
    attn_min = float(mean_action_attention.min())

    base_overlay_info = _build_heatmap_overlay(
        image=base_image,
        patch_scores=base_patch_scores,
        alpha=alpha,
        patch_grid_size=grid_size,
    )
    wrist_overlay_info = _build_heatmap_overlay(
        image=wrist_image,
        patch_scores=wrist_patch_scores,
        alpha=alpha,
        patch_grid_size=grid_size,
    )

    if mode == "topk":
        base_topk = topk_patches(base_patch_scores, grid_size=grid_size, k=top_k)
        wrist_topk = topk_patches(wrist_patch_scores, grid_size=grid_size, k=top_k)
        base_vis_bgr = _draw_topk_boxes_on_overlay(base_overlay_info["overlay_bgr"], base_topk, grid_size)
        wrist_vis_bgr = _draw_topk_boxes_on_overlay(wrist_overlay_info["overlay_bgr"], wrist_topk, grid_size)
    else:
        base_topk = None
        wrist_topk = None
        base_vis_bgr = base_overlay_info["overlay_bgr"]
        wrist_vis_bgr = wrist_overlay_info["overlay_bgr"]

    base_vis_rgb = cv2.cvtColor(base_vis_bgr, cv2.COLOR_BGR2RGB)
    wrist_vis_rgb = cv2.cvtColor(wrist_vis_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.1, 1.0], width_ratios=[1, 1], hspace=0.28, wspace=0.12)
    ax_top = fig.add_subplot(gs[0, :])
    ax_base = fig.add_subplot(gs[1, 0])
    ax_wrist = fig.add_subplot(gs[1, 1])

    im = ax_top.imshow(mean_action_attention, cmap="turbo", aspect="auto")
    ax_top.set_title("Mean Action Attention Map", fontsize=12, fontweight="bold")
    ax_top.set_xlabel("Key Tokens (K)")
    ax_top.set_ylabel("Action Tokens (Q)")
    cbar_top = fig.colorbar(im, ax=ax_top, fraction=0.025, pad=0.02)
    cbar_top.set_label("Attention", fontsize=10)

    ax_base.imshow(base_vis_rgb)
    ax_base.set_title("Base Mean Heatmap" if mode == "mean" else f"Base Top-{top_k} Overlay", fontsize=11, fontweight="bold")
    ax_base.axis("off")

    ax_wrist.imshow(wrist_vis_rgb)
    ax_wrist.set_title("Wrist Mean Heatmap" if mode == "mean" else f"Wrist Top-{top_k} Overlay", fontsize=11, fontweight="bold")
    ax_wrist.axis("off")

    title_parts = []
    if sample_id is not None:
        title_parts.append(f"idx: {sample_id}")
    if object_name is not None:
        title_parts.append(f"object: {object_name}")
    title_parts.append(f"max: {attn_max:.4f}")
    title_parts.append(f"min: {attn_min:.4f}")
    fig.suptitle(" | ".join(title_parts), fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": str(output_path),
        "mean_action_attention_shape": tuple(mean_action_attention.shape),
        "mean_action_attention_max": attn_max,
        "mean_action_attention_min": attn_min,
        "base_topk": base_topk,
        "wrist_topk": wrist_topk,
    }


def save_both_attention_figures(
    action_attentions: Sequence[torch.Tensor],
    base_image: np.ndarray,
    wrist_image: np.ndarray,
    base_patch_scores: np.ndarray,
    wrist_patch_scores: np.ndarray,
    out_dir: str | Path,
    data_index: int | str,
    object_name: str,
    layer_idx: int = -1,
    alpha: float = 0.3,
    grid_size: int = 16,
    top_k: int = 5,
) -> Dict[str, Any]:
    safe_object = str(object_name).replace(" ", "_")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_path = out_dir / f"{data_index}_{safe_object}_mean_summary.png"
    topk_path = out_dir / f"{data_index}_{safe_object}_topk_summary.png"

    mean_result = save_attention_summary_figure(
        action_attentions=action_attentions,
        base_image=base_image,
        wrist_image=wrist_image,
        base_patch_scores=base_patch_scores,
        wrist_patch_scores=wrist_patch_scores,
        output_path=mean_path,
        sample_id=data_index,
        object_name=object_name,
        layer_idx=layer_idx,
        alpha=alpha,
        grid_size=grid_size,
        mode="mean",
    )

    topk_result = save_attention_summary_figure(
        action_attentions=action_attentions,
        base_image=base_image,
        wrist_image=wrist_image,
        base_patch_scores=base_patch_scores,
        wrist_patch_scores=wrist_patch_scores,
        output_path=topk_path,
        sample_id=data_index,
        object_name=object_name,
        layer_idx=layer_idx,
        alpha=alpha,
        grid_size=grid_size,
        top_k=top_k,
        mode="topk",
    )

    return {"mean": mean_result, "topk": topk_result}


def analyze_single_image_attention(
    image: ArrayLike,
    action_attentions: Sequence[torch.Tensor],
    image_token_slice: slice,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    attn_result = _extract_patch_attention(
        action_attentions=action_attentions,
        image_token_slice=image_token_slice,
        layer_idx=config.layer_idx,
        action_token_idx=config.action_token_idx,
        head_idx=config.head_idx,
    )
    aggregated_patch_scores = attn_result["patch_scores"]         # [P]
    all_action_patch_attn = attn_result["all_action_patch_attn"]  # [Q, P]

    overlay_info = _build_heatmap_overlay(
        image=image,
        patch_scores=aggregated_patch_scores,
        alpha=config.alpha,
        patch_grid_size=config.patch_grid_size,
    )

    grid_size = overlay_info["grid_size"]
    top_items = topk_patches(aggregated_patch_scores, grid_size=grid_size, k=config.top_k)
    topk_share = float(sum(item["share"] for item in top_items))

    return {
        "patch_scores": aggregated_patch_scores,
        "aggregated_patch_scores": aggregated_patch_scores,
        "all_action_patch_attn": all_action_patch_attn,
        "entropy": attention_entropy(aggregated_patch_scores),
        "gini": attention_gini(aggregated_patch_scores),
        "topk_patches": top_items,
        "topk_share": topk_share,
        **overlay_info,
    }


def analyze_pi0_attention_sample(
    *,
    policy: Any,
    preprocess: Any,
    observation: Mapping[str, Any],
    images: Mapping[str, Any],
    sample_id: str | int,
    output_dir: str | Path,
    config: Optional[AnalysisConfig] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or AnalysisConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_obs = preprocess(dict(observation))
    with torch.no_grad():
        actions, _, action_attentions = policy.predict_action_chunk(
            processed_obs,
            output_attentions=True,
        )

    extra_metadata = dict(extra_metadata or {})
    object_name = extra_metadata.get("object_name", "unknown")
    safe_object = str(object_name).replace(" ", "_")

    record: Dict[str, Any] = {
        "sample_id": sample_id,
        "pred_action_shape": tuple(actions.shape) if hasattr(actions, "shape") else None,
    }
    record.update(extra_metadata)

    image_results: Dict[str, Dict[str, Any]] = {}

    for image_cfg in config.image_configs:
        if image_cfg.name not in images:
            raise KeyError(f"Missing image '{image_cfg.name}' in images dict.")

        result = analyze_single_image_attention(
            image=images[image_cfg.name],
            action_attentions=action_attentions,
            image_token_slice=image_cfg.token_slice,
            config=config,
        )
        image_results[image_cfg.name] = result

        overlay_path = output_dir / f"{sample_id}_{safe_object}_{image_cfg.name}_highlight.png"

        # 只有开关打开才真正画图
        if config.save_overlay_figures:
            save_overlay_image_with_plot(
                overlay_rgb=result["overlay_rgb"],
                output_path=overlay_path,
                sample_id=sample_id,
                object_name=object_name,
                max_attention=float(result["aggregated_patch_scores"].max()),
                topk_patches=result["topk_patches"],
                grid_size=result["grid_size"],
            )

        record[f"entropy({image_cfg.name})"] = result["entropy"]
        record[f"Attention Gini({image_cfg.name})"] = result["gini"]
        record[f"Top-{config.top_k} patches ({image_cfg.name.replace('_img', ' image')})"] = json.dumps(
            result["topk_patches"], ensure_ascii=False
        )
        record[f"Top-{config.top_k} share({image_cfg.name})"] = result["topk_share"]
        record[f"max_patch({image_cfg.name})"] = json.dumps(result["max_patch_rowcol"])
        record[f"max_point_xy({image_cfg.name})"] = json.dumps(result["max_point_xy"])
        record[f"max_attention_value({image_cfg.name})"] = result["max_value"]

        # 是否把路径写进 record
        if config.save_overlay_figures or config.add_missing_figure_paths_to_record:
            record[f"heatmap_highlight ({image_cfg.name})"] = str(overlay_path)

    # ===== 1) mean summary 图 =====
    mean_summary_path = output_dir / "summary" / f"{sample_id}_{safe_object}_mean_summary.png"

    if config.save_mean_summary_figure:
        mean_summary_path.parent.mkdir(parents=True, exist_ok=True)
        mean_summary_path = save_mean_summary_figure(
            action_attentions=action_attentions,
            base_image=images["base_img"],
            wrist_image=images["wrist_img"],
            base_patch_scores=image_results["base_img"]["aggregated_patch_scores"],
            wrist_patch_scores=image_results["wrist_img"]["aggregated_patch_scores"],
            output_path=mean_summary_path,
            sample_id=sample_id,
            object_name=object_name,
            layer_idx=config.layer_idx,
            alpha=config.alpha,
            grid_size=image_results["base_img"]["grid_size"],
        )

    if config.save_mean_summary_figure or config.add_missing_figure_paths_to_record:
        record["mean_summary_figure"] = str(mean_summary_path)

    # ===== 2) cluster summary 图 =====
    cluster_paths = {
        "base_cluster_summary": output_dir / f"{sample_id}_{safe_object}_base_cluster_summary.png",
        "wrist_cluster_summary": output_dir / f"{sample_id}_{safe_object}_wrist_cluster_summary.png",
    }

    if config.save_cluster_summary_figures:
        cluster_paths = save_dual_topk_cluster_figures(
            action_attentions=action_attentions,
            base_image=images["base_img"],
            wrist_image=images["wrist_img"],
            base_all_action_patch_attn=image_results["base_img"]["all_action_patch_attn"],
            wrist_all_action_patch_attn=image_results["wrist_img"]["all_action_patch_attn"],
            out_dir=output_dir,
            sample_id=sample_id,
            object_name=object_name,
            layer_idx=config.layer_idx,
            alpha=config.alpha,
            grid_size=image_results["base_img"]["grid_size"],
            n_clusters=4,
        )

    if config.save_cluster_summary_figures or config.add_missing_figure_paths_to_record:
        record["base_cluster_summary_figure"] = str(cluster_paths["base_cluster_summary"])
        record["wrist_cluster_summary_figure"] = str(cluster_paths["wrist_cluster_summary"])

    return record


def batch_analyze_pi0_attention(
    items: Iterable[Mapping[str, Any]],
    *,
    policy: Any,
    preprocess: Any,
    output_dir: str | Path,
    config: Optional[AnalysisConfig] = None,
    save_csv: bool = True,
    csv_name: str = "pi0_attention_metrics.csv",
    # save_summary_figures: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    config = config or AnalysisConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(items):
        sample_id = item.get("sample_id", idx)
        row = analyze_pi0_attention_sample(
            policy=policy,
            preprocess=preprocess,
            observation=item["observation"],
            images=item["images"],
            sample_id=sample_id,
            output_dir=output_dir,
            config=config,
            extra_metadata=item.get("metadata"),
            # save_summary_figures=save_summary_figures,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if save_csv:
        csv_path = output_dir / csv_name
        df.to_csv(csv_path, index=False)
    return df


def summarize_topk_string(topk_json: str) -> str:
    items = json.loads(topk_json)
    return "; ".join(
        f"#{x['rank']}: patch={x['patch_idx']} (r{x['row']},c{x['col']}), val={x['value']:.4f}, share={x['share']:.2%}"
        for x in items
    )


__all__ = [
    "AnalysisConfig",
    "ImageAttentionConfig",
    "analyze_single_image_attention",
    "analyze_pi0_attention_sample",
    "batch_analyze_pi0_attention",
    "attention_entropy",
    "attention_gini",
    "topk_patches",
    "save_overlay_image_with_plot",
    "save_mean_summary_figure",
    "save_topk_cluster_summary_figure",
    "save_dual_topk_cluster_figures",
    "summarize_topk_string",
]
