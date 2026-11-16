import os
from collections import Counter
from datasets import load_dataset
import pandas as pd
from .pi0_attention_utils import (
    batch_analyze_pi0_attention,
    AnalysisConfig,
)
from .observation_utils import build_analysis_items_from_dataset
from .video_utils import images_to_video



def analyze_one_episode_all_layers(
    *,
    dataset,
    episode_indices,
    object_name: str,
    output_root: str,
    policy,
    preprocess,
    device,
    dataset_cfg,
    base_attn_cfg,
    layers,
    pair_mode: str = "zip",
):
    """
    对一个 episode 的所有 step，在多个 layer 上做 attention 分析。
    每个 layer 保存到独立文件夹。
    """
    os.makedirs(output_root, exist_ok=True)

    # 1) 构建该 episode 的 items（所有 step）
    items = build_analysis_items_from_dataset(
        dataset,
        indices=episode_indices,
        device=device,
        config=dataset_cfg,
        object_names=[object_name] * len(episode_indices),
        pair_mode=pair_mode,
        metadata_builder=lambda sample, idx: {
            "task_index": sample.get("task_index"),
            "episode_index": sample.get("episode_index", sample.get("episode_id")),
            "step_index": sample.get("frame_index", sample.get("step_index", idx)),
            "global_index": idx,
            "object_name": object_name,
        },
    )

    all_layer_dfs = []
    # 2) 每个 layer 单独分析
    for layer_idx in layers:
        print(f"\n===== Analyzing layer {layer_idx} =====")

        layer_output_dir = os.path.join(output_root, f"layer_{layer_idx}")
        os.makedirs(layer_output_dir, exist_ok=True)

        attn_cfg = AnalysisConfig(
            layer_idx=layer_idx,
            action_token_idx=base_attn_cfg.action_token_idx,
            head_idx=base_attn_cfg.head_idx,
            alpha=base_attn_cfg.alpha,
            top_k=base_attn_cfg.top_k,
            patch_grid_size=base_attn_cfg.patch_grid_size,
            image_configs=base_attn_cfg.image_configs,
            save_overlay_figures=base_attn_cfg.save_overlay_figures,
            save_mean_summary_figure=base_attn_cfg.save_mean_summary_figure,
            save_cluster_summary_figures=base_attn_cfg.save_cluster_summary_figures,
            add_missing_figure_paths_to_record=base_attn_cfg.add_missing_figure_paths_to_record,
        )

        df = batch_analyze_pi0_attention(
            items,
            policy=policy,
            preprocess=preprocess,
            output_dir=layer_output_dir,
            config=attn_cfg,
            save_csv=True,
            csv_name=f"attention_metrics_layer_{layer_idx}.csv",
        )

        if df is not None and len(df) > 0:
            df = df.copy()
            df["layer_idx"] = layer_idx
            all_layer_dfs.append(df)

        # 3) 把该 layer 的 summary 图做成视频
        summary_folder = os.path.join(layer_output_dir, "summary")
        video_path = os.path.join(layer_output_dir, f"attention_video_layer_{layer_idx}.mp4")

        if os.path.isdir(summary_folder):
            pngs = [f for f in os.listdir(summary_folder) if f.lower().endswith(".png")]
            if len(pngs) > 0:
                images_to_video(
                    image_folder=summary_folder,
                    output_video=video_path,
                    fps=10,
                    codec="mp4v",
                )
                print(f"Saved video: {video_path}")
            else:
                print(f"No PNG files found in {summary_folder}, skip video.")
        else:
            print(f"Summary folder not found: {summary_folder}")

    # 4) 合并所有 layer 的 df
    merged_csv_path = None
    if len(all_layer_dfs) > 0:
        merged_df = pd.concat(all_layer_dfs, ignore_index=True)
        merged_csv_path = os.path.join(output_root, "attention_metrics_all_layers.csv")
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Saved merged CSV: {merged_csv_path}")
    else:
        merged_df = None

    return merged_df, merged_csv_path




def get_object_names_for_indices(dataset, indices, task_to_object):
    objs = []
    for idx in indices:
        sample = dataset[idx]
        task_id = sample.get("task_index")
        obj = task_to_object.get(task_id, "unknown")
        objs.append(obj)
    return objs


def get_main_object_for_episode(dataset, episode_indices, task_to_object):
    """
    对一个 episode，统计其中所有 sample 的 task_index，
    取出现最多的那个 task 对应的 object_name。
    """
    task_ids = []
    for idx in episode_indices:
        sample = dataset[idx]
        task_id = sample.get("task_index")
        task_ids.append(task_id)

    if len(task_ids) == 0:
        return "unknown"

    main_task_id = Counter(task_ids).most_common(1)[0][0]
    return task_to_object.get(main_task_id, "unknown")


def analyze_multiple_episodes(
    *,
    dataset,
    episode_ids,
    episode_to_indices,
    task_to_object,
    output_root,
    policy,
    preprocess,
    device,
    dataset_cfg,
    base_attn_cfg,
    layers,
):
    results = {}

    for ep_id in episode_ids:
        print("\n\n############################")
        print(f"Processing episode: {ep_id}")
        print("############################")

        episode_indices = episode_to_indices[ep_id]

        # 自动确定这个 episode 对应的 object
        object_name = get_main_object_for_episode(
            dataset,
            episode_indices,
            task_to_object,
        )
        print(f"Detected object for episode {ep_id}: {object_name}")

        ep_output_root = os.path.join(output_root, f"episode_{ep_id}_{object_name}")

        merged_df, merged_csv_path = analyze_one_episode_all_layers(
            dataset=dataset,
            episode_indices=episode_indices,
            object_name=object_name,
            output_root=ep_output_root,
            policy=policy,
            preprocess=preprocess,
            device=device,
            dataset_cfg=dataset_cfg,
            base_attn_cfg=base_attn_cfg,
            layers=layers,
            pair_mode="zip",
        )

        results[ep_id] = {
            "object_name": object_name,
            "episode_indices": episode_indices,
            "merged_df": merged_df,
            "merged_csv_path": merged_csv_path,
        }

    return results