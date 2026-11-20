# Configuration, Outputs and Analysis

Reference for the runner and server settings, the files a run produces, how to analyze attention, and how to reproduce the headline numbers. For first-time setup see the [server guide](../setup/server-setup.md) and [client guide](../setup/client-setup.md).

## Configuration reference

The client is configured by [`configs/runner/libero_pi0.yaml`](../../configs/runner/libero_pi0.yaml). The settings you will touch most:

| Setting | Default | What it does |
|---|---|---|
| `env.task_suite` | `libero_object` | Which LIBERO suite to run |
| `env.num_trials_per_task` | `50` | Episodes per task (the study used 10 tasks × 10 trials = 100 per condition) |
| `env.max_steps` | `500` | Step cap per episode |
| `experiment.seed` | `1` | Random seed; keep it fixed across a sweep |
| `observation.beta` | `30` | Brightness offset. **Set to `0` for the clean 87/100 baseline** (30 is an inherited offset) |
| `observation.alpha` | `1` | Contrast gain (1.0 = unchanged) |
| `observation.test_zero_proprio` | `false` | Replace the robot state with a fixed pose |
| `observation.proprio_noise_std` | `0.0` | Noise on the robot state, as a unitless multiplier of a physical reference (about 2 cm at `1.0`, 4 cm at `2.0`) |
| `scene.mode` / `scene.distance` | `none` / `0.0` | Stage 3 scene perturbation: `none`, `move_target`, `move_distractors` or `swap`, with the displacement in metres (see [`libero_pi0_scene.yaml`](../../configs/runner/libero_pi0_scene.yaml)) |
| `policy_client.include_attentions` | `true` | Ask the server to return attention maps (does not change behavior: 94/100 vs 91/100) |
| `logging.save_episode_npz` / `save_rollout_video` | `true` / `true` | Record each trajectory / front-camera video (both are needed for videos) |

The server is configured by [`configs/server/`](../../configs/server/); by default it listens on `127.0.0.1:8001` in `float16` on CUDA.

## What gets recorded

Each run writes to `experiment.output_dir`:

```text
outputs/<run>/
├── episodes_npz/task_<id>_episode_<n>.npz     one compressed file per episode
├── rollouts/<time>--episode=<n>--success=<bool>--task=<task>.mp4
└── results.json                               success, task, episode, video and NPZ path per episode
```

Each NPZ holds the task description, initial state, success flag, actions (raw and processed), per-step **attention maps**, front and wrist images, and optional flow velocities, NLL values and hidden states.

## Analyzing attention

For each frame, the last layer's action-query attention is sliced into the **256 base-camera** and **256 wrist-camera** image-patch tokens and reshaped into a **16×16 heat map**. From it, four numbers are computed: **peak attention**, **entropy** (how spread out), **Gini coefficient** (how unequal), and **top-k patch share** (how much of the attention sits on the k strongest patches).

```bash
python scripts/analyze_attention.py \
    --npz_dir ./outputs/libero_eval_v044/episodes_npz \
    --output_dir ./outputs/libero_eval_v044/attention_analysis \
    --head_idx 0 --patch_grid_size 16 --top_k 5
```

This writes `sample_level_attention_metrics.csv` (all episodes), one per-episode metrics CSV, and front- and wrist-view **heat-map videos** under `heatmaps/`. To read the videos, see [`docs/guides/reading-attention-videos.md`](reading-attention-videos.md); example clips and contact sheets for four objects are in [`research/summer-failure-analysis/evidence/attention-audit/`](../../research/summer-failure-analysis/evidence/attention-audit/). Unreadable NPZ files, or ones without attention data, are skipped with a warning.

## Reproducing the headline numbers

| Number | How to get it |
|---|---|
| **10/100** (the bug) | Run with `transformers==5.12.1`. The original per-episode result is archived in [`research/summer-failure-analysis/evidence/broken-baseline/`](../../research/summer-failure-analysis/evidence/broken-baseline/) |
| **91/100** | Pin `transformers==5.3.0`, `torch==2.7.1`, keep `observation.beta: 30` |
| **94/100** | The same with `include_attentions: true` |
| **87/100 (clean anchor)** | The same with `observation.beta: 0` |
| **Contrast and proprioception sweeps** | `scripts/run_sweep.py`, see [`docs/guides/sweeps-and-scene-tooling.md`](sweeps-and-scene-tooling.md) |
| **Swap and displacement (Stage 3)** | `scripts/run_sweep.py` with [`libero_pi0_scene.yaml`](../../configs/runner/libero_pi0_scene.yaml), see [`docs/guides/object-position-experiments.md`](object-position-experiments.md) |

Run-to-run variation of the clean baseline was 84–87%, so differences of about 4 points are noise. All numbers and their sources are tabulated in [`research/summer-failure-analysis/data/`](../../research/summer-failure-analysis/data/).
