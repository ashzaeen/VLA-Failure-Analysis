# LeRobot Model Execution Guide

This document describes the environment setup, model configuration, and startup
procedure for the LeRobot model server.

For LIBERO client environment setup and startup instructions, see
[`docs/setup/Client-Setup.md`](Client-Setup.md).

All commands assume the project root directory:

```bash
export VLA_FAILURE_ANALYSIS_ROOT="/path/to/vla-failure-analysis"
cd "${VLA_FAILURE_ANALYSIS_ROOT}"
```

## Server Environment Setup

Create and activate a dedicated Conda `lerobot` environment:

```bash
conda create -n lerobot python=3.12
conda activate lerobot
```

Install the PI policy and its dependencies from the LeRobot source included in
this project:

```bash
cd "${VLA_FAILURE_ANALYSIS_ROOT}/vlas/lerobot"
pip install -e ".[pi]"
```

Install the additional dependencies required for the current objectVLA execution
and analysis workflows:

```bash
pip install scikit-learn matplotlib
pip install tensorflow==2.21.0
pip install robosuite==1.4.1 easydict bddl gym
pip install uvicorn fastapi
```

After installation, return to the project root and set the project Python path:

```bash
cd "${VLA_FAILURE_ANALYSIS_ROOT}"
export PYTHONPATH="${VLA_FAILURE_ANALYSIS_ROOT}"
```

Install these dependencies in the dedicated `lerobot` environment. After
recreating the environment, run baseline tests to confirm that dependency
resolution has not changed the current server behavior.

## LeRobot + LIBERO

### Environment and Responsibilities

- The server uses the Conda `lerobot` environment to load LeRobot PI0 and run
  inference.
- The server and client communicate through the objectVLA `/predict` control
  protocol.
- The LIBERO client uses a separate environment. See
  [`docs/setup/Client-Setup.md`](Client-Setup.md) for setup and startup instructions.

### Configuration File

- Server: `configs/server/pi0_libero_finetuned_v044.yaml`

### Model Download

Before starting the server, ensure that the LeRobot PI0 LIBERO fine-tuned model
has been downloaded:

```bash
cd "${VLA_FAILURE_ANALYSIS_ROOT}/vlas/checkpoints"

python download_model.py lerobot/pi0_libero_finetuned_v044 \
  --target_dir=.
```

`download_model.py` automatically creates a model-named directory under
`target_dir`. The command above downloads the model to:

```text
${VLA_FAILURE_ANALYSIS_ROOT}/vlas/checkpoints/pi0_libero_finetuned_v044
```

The server configuration uses the project-relative `checkpoint_path`
`vlas/checkpoints/pi0_libero_finetuned_v044`.

### Start the Server

```bash
cd "${VLA_FAILURE_ANALYSIS_ROOT}"
conda activate lerobot
export PYTHONPATH="${VLA_FAILURE_ANALYSIS_ROOT}"

python scripts/run_policy_server.py \
  --config configs/server/pi0_libero_finetuned_v044.yaml
```

By default, the server listens on `127.0.0.1:8001`.

### Start the Client

For client configuration and startup commands, see
[`docs/setup/Client-Setup.md`](Client-Setup.md).

## LeRobot + UR5

### Environment and Responsibilities

- The server uses the Conda `lerobot` environment to load the UR5 RVL PI0
  checkpoint.
- The repository currently contains only the UR5 server configuration. No
  official UR5 client configuration or startup script is provided.

### Configuration Files

- Server: `configs/server/pi0_ur5_rvl_v1.yaml`
- Client: Not currently provided.

The checkpoint path in the server configuration is:

```text
vlas/checkpoints/rvl_ur5_task1_4_pick
```

### Start the Server

```bash
cd "${VLA_FAILURE_ANALYSIS_ROOT}"
conda activate lerobot
export PYTHONPATH="${PYTHONPATH}:${VLA_FAILURE_ANALYSIS_ROOT}"

python scripts/run_policy_server.py \
  --config configs/server/pi0_ur5_rvl_v1.yaml
```

According to the configuration, the server listens on `0.0.0.0:8001` by
default.

## Verification

After starting the server, check the health endpoint:

```bash
curl http://{ip:port}/health
```
