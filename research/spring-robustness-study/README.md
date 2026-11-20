# Spring 2026: Robustness Evaluation on LIBERO-plus

The spring project evaluated the zero-shot robustness of π0 against the LIBERO-plus benchmark, which adds controlled perturbations on a difficulty scale. It was presented as a poster at the **CRA Undergraduate Conference**, and is the starting point for the summer failure analysis.

## Result in One Table

| Perturbation category | Result | Pattern |
|---|---|---|
| Object layout | ~100%, flat at every intensity | Robust |
| Background textures | ~100%, flat at every intensity | Robust |
| Language, light conditions, sensor noise | ~90–100% | Graceful |
| **Camera viewpoint** | average ~67%, as low as 50% at the hardest level | Volatile |
| **Robot initial state** | average ~67%, as low as 50% at the hardest level | Volatile |

The model survived having **objects** moved but not the **robot or camera**, which suggested reliance on memorized, ego-centric spatial frames. The [summer study](../summer-failure-analysis/README.md) set out to test that suspicion directly.

## Files

| File | What it is |
|---|---|
| [`Proposal.pdf`](Proposal.pdf) | Project proposal: *Robustness and Sensitivity Analysis of Vision-Language-Action Models via Multi-Modal Perturbation* |
| [`Abstract.pdf`](Abstract.pdf) | Abstract |
| [`Research-Poster.pdf`](Research-Poster.pdf) | The research poster, presented at the CRA Undergraduate Conference |
| [`code/run_robustness_eval.py`](code/run_robustness_eval.py) | Evaluation script: takes a task ID, perturbation type and intensity, and runs the LIBERO rollout |
| [`code/perturbations.py`](code/perturbations.py) and [`code/perturbations.json`](code/perturbations.json) | Linguistic perturbations (verbosity, syntax, synonym) looked up per task |

The scripts are kept as they were used in the spring; they depend on the LIBERO evaluation environment described in the main [README](../../README.md).
