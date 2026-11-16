# LIBERO Client Execution Guide

This document describes the responsibilities, environment setup, and startup
procedure for the objectVLA LIBERO client.

For model server environment setup and startup instructions, see:

- LeRobot server: [`docs/lerobot.md`](lerobot.md)

## Client Responsibilities

The LIBERO client uses `src.runner.libero.evaluator.LiberoEvaluator` as its main
execution entry point. It:

- Executes a configured LIBERO task suite, such as `libero_object`.
- Sends environment observations and task descriptions to the server
  `/predict` endpoint, then requests and executes actions.
- Records every executed trajectory, including actions, images, task results,
  and enabled analysis data.
- Uses the YAML configuration to control task type, execution count, action
  processing, saved content, and output locations.

The client does not load the model. Model inference is performed by the remote
server.

## Client Environment

The LIBERO client uses a dedicated Conda `libero` environment. Following the
[official LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
installation instructions, create and activate the environment:

```bash
conda create -n libero python=3.9.13
conda activate libero
```

Download the LIBERO source and install its dependencies:

```bash
export LIBERO_ROOT="/path/to/LIBERO"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_ROOT}"
cd "${LIBERO_ROOT}"

pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

After installation, run the client from the current project root and set the
Python path:

```bash
export VLA_FAILURE_ANALYSIS_ROOT="/path/to/vla-failure-analysis"
cd "${VLA_FAILURE_ANALYSIS_ROOT}"
export PYTHONPATH="${VLA_FAILURE_ANALYSIS_ROOT}"
```

Before starting the client, start the corresponding model server and confirm
that `policy_client.base_url` and `policy_client.policy_id` match the server.

## Runner Configuration

The repository provides the following LIBERO client configuration:

```text
configs/runner/libero_pi0.yaml
```

Its main runtime parameters are:

```yaml
experiment:
  output_dir: outputs/libero_object_eval_4_embeds

env:
  task_suite: libero_object
  resolution: 256
  max_steps: 500
  num_trials_per_task: 50
  num_steps_wait: 10

policy_client:
  base_url: http://127.0.0.1:8001
  policy_id: pi0
  use_chunk: true
  replan_threshold: 1
  run_steps: 20
  include_attentions: true
  include_nll: false
  include_obs_embs: false

logging:
  save_rollout_video: true
  save_episode_npz: true
```

## Startup

First, start the LeRobot LIBERO server according to
[`docs/lerobot.md`](lerobot.md). Then run the following commands in another
terminal:

```bash
cd "${VLA_FAILURE_ANALYSIS_ROOT}"
conda activate libero
export PYTHONPATH="${VLA_FAILURE_ANALYSIS_ROOT}"

python scripts/run_libero_eval.py \
  --config configs/runner/libero_pi0.yaml
```

The current `LiberoEvaluator.evaluate()` implementation runs from task ID `8`
through the end of the task suite. It executes `env.num_trials_per_task`
episodes for each task. To reduce the evaluation scope, lower
`num_trials_per_task` and `max_steps`; the current configuration structure does
not support `env.task_ids`. The episode ID used in filenames currently starts
at `400` and increases continuously across all tasks.

## Actual Recording Locations

All client records are saved directly to the `experiment.output_dir` configured
in YAML. For the current `configs/runner/libero_pi0.yaml`, this is:

```text
${VLA_FAILURE_ANALYSIS_ROOT}/outputs/libero_object_eval_4_embeds/
```

The actual directory structure is:

```text
outputs/libero_object_eval_4_embeds/
├── episodes_npz/
│   └── task_<task_id>_episode_<episode_idx>.npz
├── rollouts/
│   └── <date-time>--episode=<episode_idx>--success=<bool>--task=<task>.mp4
└── results.json
```

When `logging.save_episode_npz: true`, each compressed episode NPZ file records:

```text
task_description
initial_state
task_id
episode_idx
success
actions
raw_actions
attention_maps
front_images
wrist_images
velocitys
nll_steps
nll_totals
hidden_states
```

After all tasks finish, `scripts/run_libero_eval.py` writes `results.json`,
which records each episode's `success`, task description, task ID, episode ID,
video path, NPZ path, and `num_actions`.

Videos under `rollouts/` use only front-view images. The current implementation
collects video frames only when `logging.save_episode_npz: true`; therefore,
both `save_episode_npz` and `save_rollout_video` must be enabled to generate
videos.

## Verification

After starting the server, check the health endpoint:

```bash
curl http://127.0.0.1:8001/health
```

After the run completes, inspect the result summary and episode records:

```bash
ls "${VLA_FAILURE_ANALYSIS_ROOT}/outputs/${output_dir}"
```
