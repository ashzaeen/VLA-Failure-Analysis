# Lab-Server Run Scripts

The shell and Python helpers used to run the experiments on the lab's shared GPU server. They are published **as they were used**, so they are machine-specific. Treat them as worked examples of how the sweeps were launched, and adapt them to your own setup. The reusable code is in [`scripts/`](../../scripts/) and [`src/`](../../src/); how to use it is in [`docs/guides/sweeps-and-scene-tooling.md`](../../docs/guides/sweeps-and-scene-tooling.md).

## What is machine-specific

- They assume the repo lives at `~/projects/vla-failure-analysis` and a Conda env named `vla-libero` under `~/miniconda3`.
- They expect the policy server already running on `127.0.0.1:8001`, started by a launcher (`~/start_server.sh` or `~/start_server_gpu1.sh`) that is **not** included.
- The server and client run in separate `tmux` sessions on separate GPUs (`CUDA_VISIBLE_DEVICES`), and outputs go under `outputs/`.
- They have no secrets and no remote addresses other than localhost health checks.

## Sweep drivers

| Script | What it does |
|---|---|
| [`sweep_common.sh`](sweep_common.sh) · [`sweep_common_full.sh`](sweep_common_full.sh) | Shared preamble: health check, environment, and the smoke or **full** override string (attention, video and NPZ on for the full runs) |
| [`sweep_proprio.sh`](sweep_proprio.sh) · [`sweep_proprio_full.sh`](sweep_proprio_full.sh) | Proprioception-noise sweep, std 0, 0.5, 1, 2, 4, 8 (smoke and full) |
| [`sweep_contrast.sh`](sweep_contrast.sh) · [`sweep_contrast_full.sh`](sweep_contrast_full.sh) | Contrast sweep, gain 0.2–2.0 (smoke and full) |
| [`sweep_all.sh`](sweep_all.sh) · [`sweep_all_full.sh`](sweep_all_full.sh) | Run both back to back, proprioception first |
| [`beta_sweep.sh`](beta_sweep.sh) · [`beta_sweep_attn.sh`](beta_sweep_attn.sh) | Brightness sweep, without and with attention capture |
| [`sweep_summary.sh`](sweep_summary.sh) | Summarizes the brightness-with-attention sweep (success and attention peak/entropy per offset) into `SWEEP_SUMMARY.csv` |
| [`sweep_validate.sh`](sweep_validate.sh) | A fast sanity run: one proprioception-noise condition, 1 trial per task (10 episodes), to prove the perturbation, client and server round-trip before a full sweep |

## GPU housekeeping

| Script | What it does |
|---|---|
| [`sweep_reaper.sh`](sweep_reaper.sh) | The **detached watcher**: waits for a full sweep and, **only if it completed cleanly**, tears down the policy server and its compile workers so the shared GPU is not left idle. If the sweep crashed it leaves the server up for inspection |

## Stage 3 scene experiments

| Script | What it does |
|---|---|
| [`introspect_libero_scene.py`](introspect_libero_scene.py) | Decodes every LIBERO-Object task's object layout (the **six-slot finding**) |
| [`scene_common.sh`](scene_common.sh) | Shared preamble for the scene sweeps |
| [`scene_all_smoke.sh`](scene_all_smoke.sh) · [`scene_all_smoke_vid.sh`](scene_all_smoke_vid.sh) | The swap, move-target and move-distractors runs (2 trials per task = 20 episodes per condition), first without and then with rollout video. Re-running the same seeds doubles as a reproducibility check |
| [`wait_scene_health.sh`](wait_scene_health.sh) | Wait until the policy server reports healthy |
| [`watch_anchor.sh`](watch_anchor.sh) | Block until the no-op anchor condition reports. It must reproduce the clean baseline, otherwise the Stage 3 pipeline is broken |
| [`watch_scene_smoke.sh`](watch_scene_smoke.sh) | Block until the smoke sweep finishes, then print every per-condition result |
| [`finish_scene_smoke.sh`](finish_scene_smoke.sh) | Wait for the video run, report each condition's success rate, classify which objects the robot disturbed, and package the rollout videos into one archive |

## Attention helpers (contrast sweep)

| Script | What it does |
|---|---|
| [`map_tasks_contrast.py`](map_tasks_contrast.py) | Shows each task's first-episode success across four contrast conditions (0.2×, 0.8×, 1.0×, 2.0×) |
| [`generate_attention_contrast.py`](generate_attention_contrast.py) | Builds attention analysis and rollout bundles for selected contrast conditions |
| [`summarize_attention_contrast.py`](summarize_attention_contrast.py) | Summarizes attention metrics for the low, clean and high contrast conditions (0.2×, 1.0×, 2.0×) |
