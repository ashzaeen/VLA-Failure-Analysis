# Stage 2 and Stage 3 Tooling: Sweeps, Noise and Scene Perturbations

How to run the controlled-perturbation experiments behind the findings in [`research/`](../../research/README.md). This covers the single-factor **sweep driver**, the **proprioception-noise** option, the **scene-perturbation** system, and the helper scripts used on the lab server.

> These tools were developed on the lab's shared server (GPU server "bender") and added to this repository afterwards. They were run there, with the pinned environment from [`requirements-working.txt`](../../requirements-working.txt). Everything in this guide compiles, but running it needs the model server, a GPU and the LIBERO client environment ([`server-setup.md`](../setup/server-setup.md), [`client-setup.md`](../setup/client-setup.md)).

## The idea: one factor at a time

Every experiment changes **exactly one** setting and holds everything else, including the random seed, fixed. Where possible a **no-op condition** runs the whole pipeline while changing nothing, so a bug in the harness shows up before it can contaminate a real result.

| Stage | What is perturbed | Knob | Where |
|---|---|---|---|
| 2 | Image brightness (additive offset) | `observation.beta` | [`observation_builder.py`](../../src/runner/libero/observation_builder.py) |
| 2 | Image contrast (multiplicative gain) | `observation.alpha` | same |
| 2 | Robot proprioceptive state (noise) | `observation.proprio_noise_std` | same |
| 2 | Robot state frozen to a fixed pose | `observation.test_zero_proprio` | same |
| 3 | **Object positions in the scene** | `scene.mode`, `scene.distance` | [`scene_perturbation.py`](../../src/runner/libero/scene_perturbation.py) |

## The sweep driver

[`scripts/run_sweep.py`](../../scripts/run_sweep.py) loads a base runner YAML, overrides **one** parameter across a list of values, runs the LIBERO evaluator for each condition, and writes a per-condition success summary.

```bash
python scripts/run_sweep.py \
    --base configs/runner/libero_pi0.yaml \
    --param observation.alpha \
    --values 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0 \
    --fixed observation.beta=0 \
    --num-trials 10 --seed 1 \
    --out-root outputs/contrast_sweep
```

| Argument | Meaning |
|---|---|
| `--param` | Dotted config path to sweep, e.g. `observation.alpha`, `scene.mode` |
| `--values` | Comma-separated values for that parameter |
| `--fixed` | Comma-separated dotted overrides held fixed for every condition, e.g. `observation.beta=0,logging.save_rollout_video=true` |
| `--num-trials` | Episodes per task (10 tasks × 10 trials = 100 episodes per condition) |
| `--seed` | Random seed, kept identical across conditions |
| `--out-root` | Each condition writes a sub-folder here, with its own `results.json` |

The same pattern reproduces the **brightness** sweep (`--param observation.beta --values 0,10,...,100`).

### The sweeps behind the report

| Sweep | Parameter | Values | Fixed |
|---|---|---|---|
| Brightness | `observation.beta` | 0, 10, …, 100 | `alpha=1` |
| Contrast | `observation.alpha` | 0.2, 0.4, …, 2.0 | `beta=0` |
| Proprioception noise | `observation.proprio_noise_std` | 0, 0.5, 1, 2, 4, 8 | `beta=0`, `alpha=1` |

For the **full** runs (100 episodes per condition) attention, rollout video and per-episode NPZ were all switched on:

```text
policy_client.include_attentions=true,policy_client.batch_size=1,policy_client.timeout_seconds=1200,
logging.save_episode_npz=true,logging.save_rollout_video=true,logging.save_attention=true
```

The shell wrappers that ran these on the lab server are in [`tools/lab-server/`](../../tools/lab-server/README.md).

## Proprioception noise

The robot state is 8-dimensional (end-effector position in metres, orientation in radians, gripper), so a raw noise number would mean different things on different dimensions. Instead, zero-mean Gaussian noise is scaled **per dimension** by a fixed physical reference vector:

```python
PROPRIO_NOISE_REF = [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.005, 0.005]
```

`observation.proprio_noise_std` is a **unitless multiplier** on that vector. So `std = 1` injects about 2 cm of positional noise, 0.05 rad of orientation noise and 0.005 of gripper noise (1σ); `std = 2` is about 4 cm. The noise has its own seeded generator (`observation.proprio_noise_seed`), independent of the global seed, so holding it fixed while sweeping `std` changes only the magnitude. The camera image is never touched.

## Scene perturbations (Stage 3)

Stage 3 perturbs the **world** instead of the observation: object poses are rewritten inside the LIBERO initial state before the simulator adopts it, so the robot really faces a rearranged table. The full mechanics, and why LIBERO-Object needs this, are in [`object-position-experiments.md`](object-position-experiments.md). Configure it with the `scene:` block of [`configs/runner/libero_pi0_scene.yaml`](../../configs/runner/libero_pi0_scene.yaml):

| Setting | Meaning |
|---|---|
| `scene.mode` | `none`, `move_target`, `move_distractors` or `swap` |
| `scene.distance` | Radial displacement in metres for the `move_*` modes |
| `scene.seed` | Fixes the per-episode displacement direction, so only magnitude varies across a sweep |
| `scene.swap_partner` | Which distractor the target swaps with; `-1` cycles through them by episode |

The three runs reported (20 episodes per condition, rollout video on):

```bash
# swap the target with a distractor
python scripts/run_sweep.py --base configs/runner/libero_pi0_scene.yaml \
    --param scene.mode --values swap \
    --fixed scene.seed=1,logging.save_rollout_video=true \
    --num-trials 2 --seed 1 --out-root outputs/scene_swap

# move only the target; distance 0 is the no-op anchor
python scripts/run_sweep.py --base configs/runner/libero_pi0_scene.yaml \
    --param scene.distance --values 0,0.05,0.10,0.15 \
    --fixed scene.mode=move_target,scene.seed=1,logging.save_rollout_video=true \
    --num-trials 2 --seed 1 --out-root outputs/scene_move_target

# move only the distractors (the control)
python scripts/run_sweep.py --base configs/runner/libero_pi0_scene.yaml \
    --param scene.distance --values 0.05,0.10,0.15 \
    --fixed scene.mode=move_distractors,scene.seed=1,logging.save_rollout_video=true \
    --num-trials 2 --seed 1 --out-root outputs/scene_move_distractors
```

Run the **no-op anchor first**. At distance 0 the scene is rewritten but not moved, so it must reproduce the clean baseline; if it does not, the harness is broken and nothing else should be trusted.

### Two supporting scripts

- [`scripts/validate_scene_perturbation.py`](../../scripts/validate_scene_perturbation.py): a **physics sanity check** that needs no policy server. It applies each condition, lets the simulator settle, and reports drift, falls and the closest pair of objects, so a placement that floats or interpenetrates cannot silently confound a success rate. The reported runs passed 42/42 condition-episodes.
- [`scripts/analyze_scene_failures.py`](../../scripts/analyze_scene_failures.py): classifies each episode by **what the robot actually disturbed**: the `decoy` now sitting in the target's usual slot (the position-shortcut signature), the `target`, `other`, or `none`.

## Output files

Each condition writes a folder with `results.json` (one record per episode, including a `scene` block with the start and final object positions and the objects the robot disturbed) and a `success.json` summary. Rollout videos no longer require NPZ saving: frames are now collected whenever **either** `save_rollout_video` or `save_episode_npz` is on, which removed a defect that forced about 100 MB of NPZ output per episode on anyone who only wanted videos (the 160 Stage 3 clips together total 19 MB).

## Where the results are

All findings produced with these tools, as numbers, charts, clips and reports, are in [`research/`](../../research/README.md). The 160 Stage 3 clips are in [`research/evidence/stage3-videos/`](../../research/evidence/stage3-videos/).
