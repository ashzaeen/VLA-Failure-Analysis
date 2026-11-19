# Evidence

Primary evidence behind the reports. Additional attention clips from the clean baseline live in [`../../report_videos/`](../../report_videos/).

## `stage1-broken-baseline/`

[`results_transformers-5.12.1_10-of-100.json`](stage1-broken-baseline/results_transformers-5.12.1_10-of-100.json) is the per-episode result file from the **original run of the inherited pipeline** (`transformers 5.12.1`): 100 episodes, **10 successes**. It is the "before" picture of the silent dependency bug. After pinning `transformers 5.3.0` the same suite scores 91/100. See [`../data/baselines.csv`](../data/baselines.csv).

The `video_path` and `episode_npz_path` fields point to files on the lab server and are not included.

## `stage3-videos/`

160 rollout clips from the Stage 3 scene-perturbation experiments, 20 episodes per condition. Each file name records the episode number, whether it succeeded and the task, e.g. `ep00_FAIL_bbq_sauce.mp4` or `ep03_success_salad_dressing.mp4` (the full task is "pick up the *X* and place it in the basket"). The original, much longer recorder file names were shortened so the repository checks out cleanly on Windows.

| Folder | Condition |
|---|---|
| `scene_move_target_smoke/distance_0/` | No-op anchor: target "moved" by 0 cm |
| `scene_move_target_smoke/distance_0.05/` · `0.1/` · `0.15/` | Target displaced by 5 / 10 / 15 cm |
| `scene_move_distractors_smoke/distance_0.05/` · `0.1/` · `0.15/` | Distractors displaced by 5 / 10 / 15 cm (control) |
| `scene_swap_smoke/mode_swap/` | Target and one distractor swapped |

Numbers for these conditions are in [`../data/stage3_scene_perturbations.csv`](../data/stage3_scene_perturbations.csv). The folder names end in `_smoke` because these were the first, 20-episode runs of the new scene-perturbation scripts.
