from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import torch

from .configs import AnalysisConfig
from .figures import (
    save_dual_topk_cluster_figures,
    save_mean_summary_figure,
    save_overlay_image_with_plot,
)
from .metrics import (
    _build_heatmap_overlay,
    _extract_patch_attention,
    attention_entropy,
    attention_gini,
    topk_patches,
)


def analyze_single_image_attention(
    image,
    action_attentions,
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
    aggregated_patch_scores = attn_result["patch_scores"]
    all_action_patch_attn = attn_result["all_action_patch_attn"]

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
        actions, _, action_attentions,_,_ = policy.predict_action_chunk(
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

        if config.save_overlay_figures or config.add_missing_figure_paths_to_record:
            record[f"heatmap_highlight ({image_cfg.name})"] = str(overlay_path)

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
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if save_csv:
        csv_path = output_dir / csv_name
        df.to_csv(csv_path, index=False)
    return df
