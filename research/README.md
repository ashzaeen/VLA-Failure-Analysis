# Research Index

Everything produced by the **π0 failure-analysis and object-grounding study** (spring – summer 2026), organized in one place.

> **Start here:** [`REPORT.md`](REPORT.md) is the readable, illustrated write-up.

**Headline:** π0 scores 84–94% on LIBERO-Object, but three independent experiments suggest it gets there by going to the right *place* with the right *motion*, not by finding the right *object*.

## What's in this folder

```text
research/
├── REPORT.md                    The readable report (start here)
├── reports/                     Full technical report + plain-language summary (Word and PDF)
├── figures/                     Charts used in the reports (PNG and SVG)
├── data/                        The numbers behind every claim (CSV)
├── evidence/
│   ├── stage1-broken-baseline/  The original 10/100 run caused by the silent dependency bug
│   ├── stage3-videos/           160 rollout clips from the scene-perturbation experiments
│   └── attention-audit/         Attention clips and contact sheets from the clean baseline
├── presentation/                Exit presentation and speaker notes
├── progress-reports/            Weekly progress reports, May 14 – Aug 21, 2026
├── spring-2026/                 The spring robustness study: proposal, abstract, poster, code
└── literature.md                Papers this work builds on
```

The code that produced these results is documented in [`../docs/guides/sweeps-and-scene-tooling.md`](../docs/guides/sweeps-and-scene-tooling.md) and [`../docs/guides/object-position-experiments.md`](../docs/guides/object-position-experiments.md), with the lab-server run scripts in [`../tools/lab-server/`](../tools/lab-server/README.md). The repository [README](../README.md) explains how to run the pipeline.

## Evidence map: claim → numbers → proof

| # | Claim | Numbers | Where to look |
|---|---|---|---|
| 1 | A silent library-version bug made the inherited pipeline score 10/100; fixing it gives 91/100 and a clean 87/100 anchor | [`baselines.csv`](data/baselines.csv) | [`evidence/stage1-broken-baseline/`](evidence/stage1-broken-baseline/), [`docs/project-log/stage-1-reproduction-log.md`](../docs/project-log/stage-1-reproduction-log.md), reports [week 7](progress-reports/week-07_2026-06-26.pdf) and [8](progress-reports/week-08_2026-07-03.pdf) |
| 2 | Brightness and contrast barely affect success (86–96% and 72–97%) | [`brightness_sweep_summary.csv`](data/brightness_sweep_summary.csv), [`contrast_sweep.csv`](data/contrast_sweep.csv) | [week 9](progress-reports/week-09_2026-07-10.pdf), [week 11](progress-reports/week-11_2026-07-24.pdf), [`docs/project-log/baseline-and-brightness-progress.md`](../docs/project-log/baseline-and-brightness-progress.md) |
| 3 | A few centimetres of noise on the robot's own state collapses success (86% → 4%) | [`proprioception_noise_sweep.csv`](data/proprioception_noise_sweep.csv) | [week 11](progress-reports/week-11_2026-07-24.pdf), [figure](figures/proprioception-noise-sweep.png) |
| 4 | Where the model "looks" does not track what it does | [`attention_audit.csv`](data/attention_audit.csv) | [`research/evidence/attention-audit/`](evidence/attention-audit/), [`docs/project-log/baseline-and-attention-findings.md`](../docs/project-log/baseline-and-attention-findings.md), [week 8](progress-reports/week-08_2026-07-03.pdf) |
| 5 | Swapping or moving the target collapses success; moving distractors does not | [`stage3_scene_perturbations.csv`](data/stage3_scene_perturbations.csv) | [`evidence/stage3-videos/`](evidence/stage3-videos/), [Aug 21 update](progress-reports/update_2026-08-21.pdf), [presentation](presentation/) |
| 6 | LIBERO-Object places all objects on the same six spots, so it cannot separate grounding from memorization | (described in the reports) | [technical report §9](reports/VLA-Failure-Analysis-Technical-Report.pdf), [Aug 21 update](progress-reports/update_2026-08-21.pdf) |

## Reading guide

| If you are… | Read |
|---|---|
| In a hurry | The "short version" at the top of [`REPORT.md`](REPORT.md) |
| A non-specialist | [`reports/VLA-Research-Plain-Language-Summary.pdf`](reports/VLA-Research-Plain-Language-Summary.pdf) |
| An advisor or researcher | [`reports/VLA-Failure-Analysis-Technical-Report.pdf`](reports/VLA-Failure-Analysis-Technical-Report.pdf) |
| Checking the numbers | [`data/`](data/) |
| Following the story over time | [`progress-reports/`](progress-reports/) |

## Status

Stages 1–3 are complete and were presented in the exit presentation. Stage 4 (per-condition failure-stage analysis) and recording attention during the swap condition are open. Stage 3 used 20 episodes per condition and newly written scripts; see the limitations in the technical report.

## Credits

Work by **Ashzaeen Fatmi Khan** in the Robotic Vision Laboratory at UT Arlington, advised by **Dr. William Beksi**, with **Zongyao Lyu** as graduate mentor. The model is π0 from Physical Intelligence; the benchmark is LIBERO. See [`literature.md`](literature.md).
