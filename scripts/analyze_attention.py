from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from analysis.attention.configs import AnalysisConfig
from analysis.attention.constants import TASK_TO_OBJECT_LIBERO, ATTENTION_KEY, BASE_IMG_KEY_CANDIDATES, WRIST_IMG_KEY_CANDIDATES
from analysis.attention.figures import get_attention_overlay_figure
from analysis.attention.metrics import (
    _extract_patch_attention,
    _build_heatmap_overlay,
    attention_entropy,
    attention_gini,
    topk_patches,
    get_object_names
)
import argparse

from src.utils.logger import logging
import matplotlib.pyplot as plt

from src.utils.video_utils import save_heatmap_video

logger = logging.getLogger(__name__)    

def load_npz_safely(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    result = {}
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.shape == ():
            try:
                result[k] = v.item()
            except Exception:
                result[k] = v
        else:
            result[k] = v
    return result


def to_torch_attention_list(attn_obj: Any) -> List[torch.Tensor]:
    """
    Convert action_attentions loaded from npz into List[torch.Tensor].
    Expected per layer shape: [B, H, Q, K]
    """
    if isinstance(attn_obj, torch.Tensor):
        return [attn_obj]

    if isinstance(attn_obj, np.ndarray):
        # if attn_obj.dtype == object:
            layers = attn_obj.tolist()
        # else:
        #     # If only one tensor-like array stored
        #     layers = [attn_obj]
    elif isinstance(attn_obj, list):
        layers = attn_obj
    else:
        raise TypeError(f"Unsupported attention object type: {type(attn_obj)}")

    out = []
    for x in layers:
        if isinstance(x, torch.Tensor):
            out.append(x.detach().cpu().to(torch.float32))
        else:
            out.append(torch.tensor(np.asarray(x), dtype=torch.float32))
    return out


def get_first_existing(d: Dict[str, Any], candidates: List[str], default=None):
    for k in candidates:
        if k in d:
            return d[k]
    return default

def analyze_one_data_record(
    sample_id: str | int,
    episode_idx: int,
    action_attention: Any,
    image_inputs: Optional[dict[str, np.ndarray]],
    object_name: Optional[str],
    config: AnalysisConfig,
) -> Dict[str, Any]:
    image_inputs = image_inputs or {}

    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "episode_idx": int(episode_idx) if episode_idx is not None else -1,
        "object_name": object_name,
    }


    action_attention_torch = to_torch_attention_list(action_attention)

    for image_cfg in config.image_configs:
        img_name = image_cfg.name
        image = image_inputs.get(img_name, None)


        extracted = _extract_patch_attention(
            action_attentions=action_attention_torch,
            image_token_slice=image_cfg.token_slice,
            layer_idx=config.layer_idx,
            action_token_idx=config.action_token_idx,
            head_idx=config.head_idx,
        )
        patch_scores = extracted["patch_scores"]
        

        # 统计指标
        row.update(
            analyze_attention(
                patch_scores=patch_scores,
                config=config,
                img_name=img_name,
            )
        )

        # overlay 信息
        if image is not None:
            overlay_info = _build_heatmap_overlay(
                image=image,
                patch_scores=patch_scores,
                alpha=config.alpha,
                patch_grid_size=config.patch_grid_size,
                image_name=img_name,
            )
            row.update(overlay_info)
        else:
            row[f"overlay_bgr({img_name})"] = None
            row[f"heatmap({img_name})"] = None
            row[f"max_point_xy({img_name})"] = None
            row[f"max_patch_rowcol({img_name})"] = None
            row[f"max_value({img_name})"] = None
            row[f"grid_size({img_name})"] = None

    # 取 base / wrist 的 overlay 图来拼总图
    base_name = "base_img"
    wrist_name = "wrist_img"

    fig = get_attention_overlay_figure(
        mean_action_attention=extracted.get(f"mean_action_attention"),
        base_img=row.pop(f"overlay_bgr({base_name})"),
        wrist_img=row.pop(f"overlay_bgr({wrist_name})"),
        base_max=row.get(f"max_value({base_name})"),
        wrist_max=row.get(f"max_value({wrist_name})"),
        mode="mean",
        top_k=config.top_k,
        sample_id=sample_id,
        object_name=object_name,
    )

    row["heatmap_overlay_figure"] = fig
    return row





def analyze_attention(
        patch_scores: Any,
        config: AnalysisConfig,
        img_name: str,
) -> Dict[str, Any]: 
        num_patches = len(patch_scores)
        grid_size = config.patch_grid_size or int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"num_patches={num_patches} is not square; "
                f"please set config.patch_grid_size explicitly."
            )

        top_items = topk_patches(patch_scores, grid_size=grid_size, k=config.top_k)
        top_share = float(sum(x["share"] for x in top_items))
        row= {}
        row[f"entropy({img_name})"] = attention_entropy(patch_scores)
        row[f"gini({img_name})"] = attention_gini(patch_scores)
        row[f"top{config.top_k}_share({img_name})"] = top_share
        row[f"max_attention_value({img_name})"] = float(np.max(patch_scores))
        row[f"max_patch({img_name})"] = json.dumps(top_items[0] if len(top_items) > 0 else {})
        row[f"topk_patches({img_name})"] = json.dumps(top_items, ensure_ascii=False)
        return row
    

def analyze_one_npz_record(
    record: Dict[str, Any],
    file_path: str | Path,
    config: AnalysisConfig,
    episode_key: str = "episode_idx",
    base_img_keys: Optional[List[str]] = None,
    wrist_img_keys: Optional[List[str]] = None,
    attention_key: str = "action_attentions",
    output_dir: Optional[str | Path] = None,
    sample_id: Optional[str | int] = 0,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    base_img_keys = base_img_keys or BASE_IMG_KEY_CANDIDATES
    wrist_img_keys = wrist_img_keys or WRIST_IMG_KEY_CANDIDATES
    object_name = record.get("object_name", None)
    if object_name is None:
        object_name = get_object_names(record, TASK_TO_OBJECT_LIBERO)
    print(f"object_name {object_name}")

    episode_idx = record.get(episode_key, -1)
    # sample_id = record.get("task_id", Path(file_path).stem)

    if attention_key not in record:
        raise KeyError(f"{file_path} missing key: {attention_key}")

    action_attentions = record[attention_key]

    base_imgs = get_first_existing(record, base_img_keys, default=None)
    wrist_imgs = get_first_existing(record, wrist_img_keys, default=None)
        
    all_rows = []
    frams = []
    for step_idx  in range(len(action_attentions)):
        sample_id = sample_id+1
        action_attention = action_attentions[step_idx]
        image_inputs = {
            "base_img": base_imgs[step_idx],
            "wrist_img": wrist_imgs[step_idx],
        }
        row = analyze_one_data_record(
        sample_id=sample_id,
        episode_idx=episode_idx,
        action_attention=action_attention,
        image_inputs=image_inputs,
        object_name=object_name,
        config=config,
    )
        row["step_idx"] = step_idx
        row["file_path"] = str(file_path)
        row["task_id"] = record.get("task_id", "unknown_task")
        row["file_path"] = str(file_path)
        frams.append(row.pop("heatmap_overlay_figure", None))
        all_rows.append(row)
    
    fig_path = save_heatmap_video(
        heatmap_images=frams,
        rollout_dir = output_dir / "heatmaps",
        mp4_path= f"task_idx_{record.get('task_id', 0)}_episode_{episode_idx}_{sample_id}_{object_name}.mp4"
    )
    sample_df = pd.DataFrame(all_rows)
    sample_csv = output_dir / "sample_level_attention_metrics" / f"task_idx_{record.get('task_id', 0)}_episode_{episode_idx}_{sample_id}_{object_name}.csv"
    sample_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_csv, index=False)

    return all_rows,sample_id


def analyze_npz_folder_group_by_episode(
    npz_dir: str | Path,
    output_dir: str | Path,
    *,
    episode_key: str = "episode_idx",
    attention_key: str = "attention_maps",
    head_idx: int = 0,
    patch_grid_size: int = 16,
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    npz_dir = Path(npz_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    config = AnalysisConfig(
        layer_idx=-1,            # last layer
        head_idx=head_idx,       # unique head
        action_token_idx=-1,     # average over action rows
        alpha=0.3,
        top_k=top_k,
        patch_grid_size=patch_grid_size,
        save_overlay_figures=False,
        save_mean_summary_figure=False,
        save_cluster_summary_figures=False,
    )

    rows = []
    bad_files = []
    sample_idx = 0
    for file_path in npz_files:
        try:
            record = load_npz_safely(file_path)
            row, sample_idx = analyze_one_npz_record(
                record=record,
                file_path=file_path,
                sample_id=sample_idx,
                config=config,
                episode_key=episode_key,
                attention_key=attention_key,
                output_dir = output_dir ,
                base_img_keys=["front_images"],
                wrist_img_keys=["wrist_images"],
            )
            rows.extend(row)
        except Exception as e:
            bad_files.append({"file": str(file_path), "error": str(e)})

    if not rows:
        raise RuntimeError("No valid npz files were successfully analyzed.")

    sample_df = pd.DataFrame(rows)

    sample_csv = output_dir / "sample_level_attention_metrics.csv"
    sample_df.to_csv(sample_csv, index=False)

    agg_cols = [
        c for c in sample_df.columns
        if c.startswith("entropy(")
        or c.startswith("gini(")
        or c.startswith("top")
        or c.startswith("max_attention_value(")
    ]

    # episode_df = (
    #     sample_df.groupby("episode_idx")[agg_cols]
    #     .agg(["mean", "std", "max", "min", "count"])
    # )
    # episode_df.columns = ["{}_{}".format(a, b) for a, b in episode_df.columns]
    # episode_df = episode_df.reset_index()

    # episode_csv = output_dir / "episode_level_attention_summary.csv"
    # episode_df.to_csv(episode_csv, index=False)

    # if bad_files:
    #     pd.DataFrame(bad_files).to_csv(output_dir / "bad_files.csv", index=False)

    print(f"Done. sample-level csv: {sample_csv}")
    # print(f"Done. episode-level csv: {episode_csv}")
    if bad_files:
        print(f"Warning: {len(bad_files)} files failed. See bad_files.csv")

    return sample_df



if __name__ == "__main__":

        
        parser = argparse.ArgumentParser(description="Analyze attention from npz files.")
        parser.add_argument("--npz_dir", type=str, required=False, default='outputs/libero_eval_v044/episodes_npz', help="Directory containing npz files.")
        parser.add_argument("--output_dir", type=str, required=False, default='outputs/libero_eval_v044/attention_analysis', help="Directory to save analysis results.")
        parser.add_argument("--head_idx", type=int, default=0, help="Attention head index to analyze.")
        parser.add_argument("--patch_grid_size", type=int, default=16, help="Grid size for patch attention.")
        parser.add_argument("--top_k", type=int, default=5, help="Top K patches to consider for metrics.")
        args = parser.parse_args()
        
        analyze_npz_folder_group_by_episode(
                npz_dir=args.npz_dir,
                output_dir=args.output_dir,
                head_idx=args.head_idx,
                patch_grid_size=args.patch_grid_size,
                top_k=args.top_k,
                attention_key=ATTENTION_KEY,

        )