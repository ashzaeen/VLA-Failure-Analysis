# VLA Failure Analysis

> **Research results:** this repository also holds the study's findings, evidence and reports. Start with [`research/REPORT.md`](research/REPORT.md) or the [research index](research/README.md).

This project supports data collection and data analysis for LeRobot models.

- **Data collection**: Start the LeRobot model server, execute configured tasks in
  the LIBERO client, request actions through `/predict`, and record every
  trajectory.
- **Data analysis**: Read the collected episode NPZ files, analyze model
  attention, and generate analysis results.

## Project Structure

```text
configs/    # Server and LIBERO client configurations
docs/       # Environment setup and execution guides
scripts/    # Server, client, and data analysis entry points
src/        # Server, model invocation, LIBERO runner, and utility implementations
analysis/   # Attention and other analysis logic
vlas/       # LeRobot source code and model checkpoints
outputs/    # Collected trajectories, videos, and analysis results
test/       # Test code
```

## Data Collection and Model Execution

Data collection is performed jointly by the LeRobot server and LIBERO client:

1. Start the LeRobot server to load the model and provide the `/predict`
   endpoint.
2. Start the LIBERO client to execute tasks such as `libero_object` according to
   the YAML configuration.
3. The client requests and executes actions, then saves each trajectory to the
   configured output directory.

For environment setup, model download, and server startup instructions, see:

- [LeRobot Server Guide](docs/lerobot.md)

For LIBERO environment setup, client startup, and actual recording locations,
see:

- [LIBERO Client Guide](docs/libero.md)

## Data Analysis

Use the following command to analyze attention data from collected trajectories:

```bash
python scripts/analyze_attention.py \
    --npz_dir ./outputs/libero_eval_v044/episodes_npz \
    --output_dir ./outputs/libero_eval_v044/attention_analysis \
    --head_idx 0 \
    --patch_grid_size 16 \
    --top_k 5
```

Arguments:

- `--npz_dir`: Input directory containing episode NPZ files.
- `--output_dir`: Output directory for attention analysis results.
- `--head_idx`: Index of the attention head to analyze.
- `--patch_grid_size`: Side length of the image patch attention grid; `16`
  represents a `16 x 16` patch grid.
- `--top_k`: Number of highest-attention patches considered when calculating
  metrics.

After analysis, the following files are generated under `output_dir`:

```text
attention_analysis/
├── sample_level_attention_metrics.csv
├── sample_level_attention_metrics/
│   └── task_idx_<task_id>_episode_<episode_id>_<sample_id>_<object>.csv
└── heatmaps/
    └── task_idx_<task_id>_episode_<episode_id>_<sample_id>_<object>.mp4
```

- `sample_level_attention_metrics.csv`: Aggregates step-level attention metrics
  from all valid NPZ files, including entropy, Gini, Top-K attention share,
  maximum attention value, and the corresponding patch.
- `sample_level_attention_metrics/`: Stores a separate step-level attention
  metrics CSV for each episode.
- `heatmaps/`: Stores attention heatmap videos for the front and wrist views of
  each episode.

NPZ files that cannot be read or do not contain attention data are skipped with
a terminal warning. The current script does not write failed records to
`bad_files.csv`.
