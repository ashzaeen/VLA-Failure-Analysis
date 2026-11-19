## Summer Research Plan: Failure Analysis and Object Grounding in VLA Models 

## **Project Motivation** 

Last semester, you studied the robustness of Vision-Language-Action (VLA) models by reproducing experiments from LIBERO-Plus. That project examined whether VLA performance remains stable when the environment changes, such as changes in lighting, camera conditions, object positions, or scene appearance. 

This summer project builds directly on that experience but focuses on a more specific question: **Does a VLA model genuinely understand the instruction and identify the requested object, or does it sometimes succeed by relying on irrelevant shortcuts?** 

For example, when instructed to “pick up the orange juice and place it in the basket,” the model should: 

1. Understand that the target is the orange juice. 

2. Locate the orange juice in the visual observations. 

3. Generate actions that reach, grasp, and move that object. 

However, the model may instead learn that a particular instruction is usually associated with a particular object location or robot trajectory. It could therefore learn something closer to “move to this familiar position and grasp” without reliably recognizing the orange juice. Such a shortcut may work in the original LIBERO environment but fail when the object is moved, the robot begins from a different state, or another object occupies the familiar location. 

Our long-term research direction is to improve VLA models through better visual-language grounding, ensuring that their actions are based on the instructed object rather than accidental correlations in the training data. Your summer project represents an early failure-analysis stage of this larger effort. It is also designed as a self-contained undergraduate research project that can produce a complete final report. 

## **Existing Code and Experiments** 

The current code I shared with you provides a preliminary experimental pipeline for running _π_ 0 in the LIBERO environment and collecting information from each rollout. It includes: 

- A policy server for loading and running _π_ 0 

- A LIBERO evaluation client for executing manipulation tasks 

- Recording of task success, actions, camera observations, and rollout videos 

- Storage of episode information in NPZ files 

- Extraction and visualization of model attention 

- Attention statistics such as entropy, Gini coefficient, and top-attended image patches 

- Preliminary exploration of action uncertainty, flow velocity, negative log-likelihood, and hidden-state representations 

The existing experiments examine how _π_ 0 responds to changes such as image brightness, target-object position, and robot state. Attention heatmaps are used to investigate whether the model attends to the instructed object, another object, or irrelevant background regions. 

This code should be treated as an exploratory starting point rather than a complete evaluation framework. Some experiments may be incomplete, difficult to reproduce, insufficiently controlled, or implemented only in notebooks. Part of your work is to understand what is currently available and determine what can be corrected or improved. 

1 

## **Stage 1: Understand and Reproduce the Existing Work** 

Your first objective is to become familiar with the existing pipeline and reproduce the current experiments. 

You should: 

- Understand how the _π_ 0 server and LIBERO evaluation client communicate. 

- Run several unperturbed baseline episodes successfully. 

- 

- Reproduce selected brightness, object-position, and robot-state experiments. 

- Run the existing attention-analysis tools and interpret their outputs. 

- Record the configuration, task, initial state, and random seed used in each experiment. 

Do not assume that every existing result or implementation is correct. Carefully document problems such as missing instructions, inconsistent settings, incomplete scripts, unstable behavior, or results that cannot be reproduced. Resolving and documenting these issues is itself a meaningful research contribution. 

The expected output from this stage is a clear description of the current pipeline, a set of reproduced results, and a list of limitations in the present implementation. 

## **Stage 2: Establish a Controlled Evaluation Procedure** 

After reproducing the existing work, the next objective is to make the experiments more systematic. 

A valid comparison should change one factor at a time while keeping the other conditions fixed. For example, when testing object-position sensitivity, the instruction, lighting, robot initial state, camera configuration, and other objects should remain unchanged. 

The evaluation should include: 

- An unperturbed baseline 

- Clearly defined perturbation types and magnitudes 

- Multiple trials for each condition 

- Consistent initial states or documented randomization 

- Task success rate and relative performance degradation 

- Rollout videos and qualitative failure descriptions 

Attention maps can provide useful diagnostic evidence, but they should not be treated as definitive proof that the model understands an object. They should be interpreted together with the robot’s behavior and task outcomes. 

## **Stage 3: Design New Failure-Analysis Experiments** 

Once the current experiments are reproducible, you will identify important conditions that have not yet been evaluated and design new experiments. 

Possible directions include: 

- Moving the target object while leaving distractors unchanged 

- Moving distractors while keeping the target fixed 

- Placing another object at the target’s usual location 

- Swapping the locations of the target and a distractor 

- Increasing the number or similarity of distractor objects 

- Changing the camera viewpoint or scene background 

- Rephrasing the instruction while preserving its meaning 

- Giving different target-object instructions for the same scene 

- Combining visual, spatial, state, or language changes 

These experiments should help distinguish genuine object grounding from position, state, language, or trajectory shortcuts. The final experiment set will be selected based on feasibility, the reproduced results, and the weaknesses identified in the existing pipeline. 

## **Stage 4: Analyze and Report the Findings** 

The analysis should explain not only whether the model failed, but also how its behavior changed. 

For each condition, examine: 

- Task success rate 

- Relative performance drop from the baseline 

- The stage at which execution failed 

- Whether the robot approached or grasped the wrong object 

- Whether attention shifted away from the instructed object 

- Whether failures were consistent across repeated trials 

Your final report should describe the motivation, experimental design, implementation, quantitative results, representative failure cases, limitations, and conclusions. A useful final conclusion would identify the conditions under which _π_ 0 appears to rely on shortcuts and recommend the most important directions for improving object grounding. 

## **Continuing the Project** 

This guide defines the initial direction, not a fixed limit on the summer project. We will continue refining and updating the plan as you become familiar with the code, reproduce the existing experiments, and discover new results. 

If the initial work is completed early, or if it reveals a particularly valuable pattern, we may expand the project to additional perturbations, other VLA models, real-robot experiments, improved grounding metrics, or preliminary methods for improving object grounding. The exact direction will also evolve as we develop a clearer plan for the broader long-term research project. 

