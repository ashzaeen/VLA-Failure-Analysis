# Summer Research: Failure Analysis and Object Grounding in VLA Models

## Project Overview

This project investigates whether Vision-Language-Action (VLA) models genuinely understand task instructions and identify target objects, or whether they succeed through learned shortcuts - relying on familiar object positions, robot trajectories, or other spurious correlations rather than true visual-language grounding.

**Core Question:** When instructed to "pick up the orange juice and place it in the basket," does the model:
1. Understand that the target is the orange juice
2. Locate the orange juice in the visual observations
3. Generate actions that reach, grasp, and move *that* object

...or does it learn a shortcut like "move to this familiar position and grasp" that happens to work in the training distribution but breaks under perturbation?

**Long-term direction:** Improve VLA models through better visual-language grounding, ensuring actions are driven by the instructed object rather than accidental training-data correlations. This summer project is the failure-analysis stage of that larger effort - and is designed as a self-contained undergraduate research project with a complete final report as its deliverable.

---

## Existing Infrastructure

The shared codebase provides a preliminary experimental pipeline for running π0 in LIBERO:

- **Policy server** - loads and runs π0
- **LIBERO evaluation client** - executes manipulation tasks
- **Data recording** - task success, actions, camera observations, rollout videos, NPZ files
- **Attention analysis** - extraction, visualization, entropy, Gini coefficient, top-attended patches
- **Preliminary probes** - action uncertainty, flow velocity, negative log-likelihood, hidden-state representations

> **Note:** Treat this as an exploratory starting point, not a complete evaluation framework. Some experiments may be incomplete, poorly controlled, or notebook-only. Part of the work is assessing what's usable and what needs fixing.

---

## Stage 1 - Understand and Reproduce the Existing Work

**Goal:** Become familiar with the pipeline and reproduce current experiments.

### Tasks
- [ ] Understand how the π0 server and LIBERO evaluation client communicate
- [ ] Run several unperturbed baseline episodes successfully
- [ ] Examine saved rollout videos and NPZ files
- [ ] Reproduce selected brightness, object-position, and robot-state experiments
- [ ] Run existing attention-analysis tools and interpret their outputs
- [ ] Record configuration, task, initial state, and random seed for each experiment

### What to Document
- Problems: missing instructions, inconsistent settings, incomplete scripts, unstable behavior
- Results that cannot be reproduced
- Gaps or ambiguities in the existing implementation

> Resolving and documenting these issues is itself a meaningful research contribution.

### Expected Output
- Clear description of the current pipeline
- Set of reproduced results
- List of limitations in the present implementation

---

## Stage 2 - Establish a Controlled Evaluation Procedure

**Goal:** Make experiments systematic - change one factor at a time, hold everything else fixed.

### Evaluation Checklist (per experiment)
- [ ] Unperturbed baseline included
- [ ] Perturbation types and magnitudes clearly defined
- [ ] Multiple trials per condition
- [ ] Consistent initial states (or documented randomization)
- [ ] Task success rate computed
- [ ] Relative performance degradation from baseline computed
- [ ] Rollout videos saved
- [ ] Qualitative failure descriptions written

### On Attention Maps
Attention maps are useful diagnostic evidence, but **not** definitive proof of object understanding. Interpret them alongside robot behavior and task outcomes - not in isolation.

---

## Stage 3 - Design New Failure-Analysis Experiments

**Goal:** Identify unevaluated conditions and design experiments to distinguish genuine object grounding from shortcuts.

### Candidate Experiment Directions
- Move the target object; leave distractors unchanged
- Move distractors; keep the target fixed
- Place another object at the target's usual location
- Swap the locations of the target and a distractor
- Increase the number or visual similarity of distractor objects
- Change the camera viewpoint or scene background
- Rephrase the instruction while preserving its meaning
- Give different target-object instructions for the same scene
- Combine visual, spatial, state, and language changes

### Shortcut Types to Probe
| Shortcut Type | Description |
|---|---|
| Position shortcut | Model moves to a memorized location rather than localizing the target |
| State shortcut | Model follows a trajectory tied to the robot's initial configuration |
| Language shortcut | Model maps instruction tokens to actions without visual grounding |
| Trajectory shortcut | Model replays a learned motion pattern regardless of object identity |

> The final experiment set will be selected based on feasibility, reproduced results, and weaknesses identified in Stage 2.

---

## Stage 4 - Analyze and Report the Findings

**Goal:** Explain not just *whether* the model failed, but *how* its behavior changed.

### Per-Condition Analysis
- Task success rate
- Relative performance drop from baseline
- Stage at which execution failed (approach / grasp / transport / placement)
- Whether the robot approached or grasped the wrong object
- Whether attention shifted away from the instructed object
- Whether failures were consistent across repeated trials

### Final Report Structure
1. Motivation
2. Experimental design
3. Implementation details
4. Quantitative results
5. Representative failure cases
6. Limitations
7. Conclusions - under what conditions does π0 rely on shortcuts, and what are the most important directions for improving object grounding?

---

## Possible Extensions (if time permits)

If initial work is completed early or reveals a particularly valuable pattern, the project may expand to:

- Additional perturbation types
- Other VLA models
- Real-robot experiments
- Improved grounding metrics
- Preliminary methods for improving object grounding

The exact direction will evolve in collaboration with Dr. William Beksi as results come in.

---

## Key Terms

| Term | Definition |
|---|---|
| VLA | Vision-Language-Action model - takes visual observations and language instructions as input and outputs robot actions |
| π0 (pi-zero) | The VLA model being evaluated in this project |
| LIBERO | Benchmark suite for lifelong robot learning; provides manipulation tasks and environments |
| Object grounding | The model's ability to identify and localize a specific object based on a language instruction |
| Shortcut learning | When a model achieves task success using spurious correlations rather than the intended reasoning |
| Attention map | Visualization of which image regions or tokens the model weights most heavily when generating outputs |
