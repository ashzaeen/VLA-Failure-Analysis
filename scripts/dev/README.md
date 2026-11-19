# Developer utilities

One-off helpers used during development. They are not part of the main pipeline, and several read local checkpoint or output folders that are not in this repository, so paths inside them are meant to be edited.

| Script | What it does |
|---|---|
| [`remap_checkpoint_keys.py`](remap_checkpoint_keys.py) | Rewrites the key prefix of the vision tower in a `model.safetensors` checkpoint and saves a corrected copy (used for the UR5 checkpoint) |
| [`inspect_normalizer.py`](inspect_normalizer.py) | Prints every tensor's name, shape and dtype in a normalizer `.safetensors` file |
| [`view_policy_outputs.py`](view_policy_outputs.py) | Loads recorded policy states and actions from `policy_outputs/` and plots them interactively with Plotly |
| [`run_single_episode.py`](run_single_episode.py) | Runs a single LIBERO episode through the `robots/` inference wrapper, for quick debugging |

For the main entry points, see [`../`](../): `run_policy_server.py`, `run_libero_eval.py`, `run_sweep.py` and `analyze_attention.py`.
