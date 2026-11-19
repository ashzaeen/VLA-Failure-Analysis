# Stage 3: Object-Position Perturbations

Why the object-position experiments exist, what LIBERO-Object's scenes really look like, and how [`scene_perturbation.py`](../../src/runner/libero/scene_perturbation.py) rearranges them. For how to run them, see [`sweeps-and-scene-tooling.md`](sweeps-and-scene-tooling.md).

## Why rearrange the scene?

Brightness, contrast and robot-state noise all act on the **observation** after the simulator produces it. They cannot tell you whether the model is looking at the instructed object, because the objects never move. Stage 3 changes the **scene itself**: the robot faces a genuinely rearranged table, so any reliance on "the target is always over there" shows up as a failure.

## What LIBERO-Object actually contains

Before writing any perturbation code, every object's pose was decoded across all 10 tasks and their 50 provided initial states (the script is [`tools/lab-server/introspect_libero_scene.py`](../../tools/lab-server/introspect_libero_scene.py)). Three facts reframe the benchmark:

1. All 10 tasks lay their objects on the **same six (x, y) slots**, with the basket at the same place. Only the assignment of which object sits on which slot changes between tasks.
2. Positional "randomization" across the 50 initial states is **at most 2 cm, and usually exactly 0**.
3. The **target object always sits on one of only two** of the six slots.

A policy that ignores language and vision and just reaches toward one of two memorized points therefore has a structural route to a high score. This does not prove π0 does that, but it means the benchmark alone **cannot separate grounding from memorization**.

## How a scene is rewritten

LIBERO initial states are flattened MuJoCo states:

```text
[ time (1) | qpos (nq) | qvel (nv) ]
```

A free joint whose qpos address is `a` owns state indices `1 + a … 1 + a + 7`, which hold `(x, y, z, qw, qx, qy, qz)`. The perturber rewrites **only `x` and `y`**. The height `z` is left alone because it encodes the object's resting height (geometry-specific), and the orientation is left alone so rotation is never silently confounded with position. The rewritten state is handed to `env.set_init_state`.

## The conditions

| `scene.mode` | What changes |
|---|---|
| `none` | Nothing. The scene is exactly as LIBERO ships it and earlier runs stay bit-identical |
| `swap` | The target and one distractor exchange places. **No new position appears anywhere in the scene**, which is what makes it the cleanest test |
| `move_target` | Only the target is displaced, by `scene.distance` metres, in a seeded direction |
| `move_distractors` | Only the distractors are displaced, with the target fixed. This is the **control** |

For the `move_*` modes the displacement direction is chosen per episode from a ring of candidates (36 by default), subject to placement constraints: a minimum gap between any two objects (`min_separation`, 0.08 m), a larger exclusion zone around the basket (`basket_clearance`, 0.16 m), and table bounds (`x_bounds`, `y_bounds`). Fixing `scene.seed` fixes the direction, so sweeping `distance` varies only magnitude.

## Checking the rewrite is sound

Moving an object by rewriting `qpos` only means something if it then rests stably. [`scripts/validate_scene_perturbation.py`](../../scripts/validate_scene_perturbation.py) applies each condition, lets the simulator settle under dummy actions, and reports drift, falls and the closest pair. The reported experiments validated **42/42 condition-episodes** before any GPU time was spent.

## Recording what the robot did

A success rate says the policy failed; it does not say *how*. The evaluator now records each object's start and final position and lists the **objects the robot actually disturbed** (moved by 3 cm or more), with each object's role (target or distractor). [`scripts/analyze_scene_failures.py`](../../scripts/analyze_scene_failures.py) turns that into one of four labels per episode:

| Label | Meaning |
|---|---|
| `decoy` | The distractor now sitting in the target's usual slot was moved: the **position-shortcut signature** |
| `target` | The instructed object was moved: the policy found it but did not finish, which is evidence *against* a pure position shortcut |
| `other` | Some third object was moved |
| `none` | Nothing was disturbed: the arm never made meaningful contact |

## What it showed

With 20 episodes per condition, swapping the target with a distractor dropped success from **80% to 0/20**, with the robot grasping the decoy in 60% of episodes and touching the true target in 15%. Moving only the target by 5 cm dropped success to 10%, while moving distractors up to 15 cm left it at 85–100%. The numbers, clips and full discussion are in [`research/`](../../research/README.md).
