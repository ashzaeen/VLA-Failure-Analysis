# Evidence

Primary evidence behind the reports. Three folders:

| Folder | Contents |
|---|---|
| [`broken-baseline/`](broken-baseline/) | The original 10/100 run caused by the silent dependency bug |
| [`scene-perturbation-clips/`](scene-perturbation-clips/) | 160 rollout clips from the scene-perturbation experiments |
| [`attention-audit/`](attention-audit/) | Four attention-heat-map clips (with [`Captions.txt`](attention-audit/Captions.txt)) and per-object contact sheets from the clean baseline; see [`docs/guides/Reading-Attention-Videos.md`](../../../docs/guides/Reading-Attention-Videos.md) |

## `broken-baseline/`

[`results-transformers-5.12.1-10-of-100.json`](broken-baseline/results-transformers-5.12.1-10-of-100.json) is the per-episode result file from the **original run of the inherited pipeline** (`transformers 5.12.1`): 100 episodes, **10 successes**. It is the "before" picture of the silent dependency bug. After pinning `transformers 5.3.0` the same suite scores 91/100. See [`../data/baselines.csv`](../data/baselines.csv).

The `video_path` and `episode_npz_path` fields point to files on the lab server and are not included.

## `scene-perturbation-clips/`

160 rollout clips from the Stage 3 scene-perturbation experiments, 20 episodes per condition. Each file name records the episode number, whether it succeeded and the task, e.g. `ep00-fail-bbq-sauce.mp4` or `ep03-success-salad-dressing.mp4` (the full task is "pick up the *X* and place it in the basket"). The original, much longer recorder file names were shortened so the repository checks out cleanly on Windows.

| Folder | Condition |
|---|---|
| `move-target/0cm-no-op/` | No-op anchor: target "moved" by 0 cm |
| `move-target/5cm/` · `10cm/` · `15cm/` | Target displaced by 5 / 10 / 15 cm |
| `move-distractors/5cm/` · `10cm/` · `15cm/` | Distractors displaced by 5 / 10 / 15 cm (control) |
| `swap-target-and-distractor/` | Target and one distractor swapped |

Numbers for these conditions are in [`../data/scene-perturbations.csv`](../data/scene-perturbations.csv). These were the first, 20-episode ("smoke") runs of the new scene-perturbation scripts; the lab-server scripts that produced them are named `scene_*_smoke*.sh`.
