# Summer 2026: Failure Analysis and Object Grounding in π0

Everything produced by the summer **π0 failure-analysis and object-grounding study**, organized in one place. It follows on from the [spring robustness study](../spring-robustness-study/README.md).

> **Start here:** [`reports/Illustrated-Report.md`](reports/Illustrated-Report.md) is the readable, illustrated write-up.

**Headline:** π0 scores 84–94% on LIBERO-Object, but three independent experiments suggest it gets there by going to the right *place* with the right *motion*, not by finding the right *object*.

## What's in this folder

```text
summer-failure-analysis/
├── reports/                   Illustrated report, technical report, plain-language summary
├── figures/                   Charts used in the reports (PNG and SVG)
├── data/                      The numbers behind every claim (CSV)
├── evidence/
│   ├── broken-baseline/           The original 10/100 run caused by the silent dependency bug
│   ├── scene-perturbation-clips/  160 rollout clips from the scene-perturbation experiments
│   └── attention-audit/           Attention clips and contact sheets from the clean baseline
├── presentation/              Exit presentation and speaker notes
└── progress-reports/          Weekly progress reports, May 14 – Aug 21, 2026
```

The papers this work builds on are in [`../Literature.md`](../Literature.md). The code that produced these results is documented in [`docs/guides/Sweeps-and-Scene-Tooling.md`](../../docs/guides/Sweeps-and-Scene-Tooling.md) and [`docs/guides/Object-Position-Experiments.md`](../../docs/guides/Object-Position-Experiments.md), with the lab-server run scripts in [`tools/lab-server/`](../../tools/lab-server/README.md). The repository [README](../../README.md) explains how to run the pipeline.

## Evidence map: claim → numbers → proof

| # | Claim | Numbers | Where to look |
|---|---|---|---|
| 1 | A silent library-version bug made the inherited pipeline score 10/100; fixing it gives 91/100 and a clean 87/100 anchor | [`baselines.csv`](data/baselines.csv) | [`evidence/broken-baseline/`](evidence/broken-baseline/), [`docs/project-log/2026-06-25-Stage-1-Reproduction-Log.md`](../../docs/project-log/2026-06-25-Stage-1-Reproduction-Log.md), reports [week 7](progress-reports/2026-06-26-Week-07.pdf) and [8](progress-reports/2026-07-03-Week-08.pdf) |
| 2 | Brightness and contrast barely affect success (86–96% and 72–97%) | [`brightness-sweep-summary.csv`](data/brightness-sweep-summary.csv), [`contrast-sweep.csv`](data/contrast-sweep.csv) | [week 9](progress-reports/2026-07-10-Week-09.pdf), [week 11](progress-reports/2026-07-24-Week-11.pdf), [`docs/project-log/2026-07-14-Baseline-and-Brightness-Progress.md`](../../docs/project-log/2026-07-14-Baseline-and-Brightness-Progress.md) |
| 3 | A few centimetres of noise on the robot's own state collapses success (86% → 4%) | [`proprioception-noise-sweep.csv`](data/proprioception-noise-sweep.csv) | [week 11](progress-reports/2026-07-24-Week-11.pdf), [figure](figures/proprioception-noise-sweep.png) |
| 4 | Where the model "looks" does not track what it does | [`attention-audit.csv`](data/attention-audit.csv) | [`evidence/attention-audit/`](evidence/attention-audit/), [`docs/project-log/2026-07-03-Baseline-and-Attention-Findings.md`](../../docs/project-log/2026-07-03-Baseline-and-Attention-Findings.md), [week 8](progress-reports/2026-07-03-Week-08.pdf) |
| 5 | Swapping or moving the target collapses success; moving distractors does not | [`scene-perturbations.csv`](data/scene-perturbations.csv) | [`evidence/scene-perturbation-clips/`](evidence/scene-perturbation-clips/), [Aug 21 update](progress-reports/2026-08-21-Progress-Update.pdf), [presentation](presentation/) |
| 6 | LIBERO-Object places all objects on the same six spots, so it cannot separate grounding from memorization | (described in the reports) | [technical report §9](reports/Technical-Report.pdf), [Aug 21 update](progress-reports/2026-08-21-Progress-Update.pdf) |

## Reading guide

| If you are… | Read |
|---|---|
| In a hurry | The "short version" at the top of [`Illustrated-Report.md`](reports/Illustrated-Report.md) |
| A non-specialist | [`reports/Plain-Language-Summary.pdf`](reports/Plain-Language-Summary.pdf) |
| An advisor or researcher | [`reports/Technical-Report.pdf`](reports/Technical-Report.pdf) |
| Checking the numbers | [`data/`](data/) |
| Following the story over time | [`progress-reports/`](progress-reports/) |

## Status

Stages 1–3 are complete and were presented in the exit presentation. Stage 4 (per-condition failure-stage analysis) and recording attention during the swap condition are open. Stage 3 used 20 episodes per condition and newly written scripts; see the limitations in the technical report.

## Credits

Work by **Ashzaeen Fatmi Khan** in the Robotic Vision Laboratory at UT Arlington, advised by **Dr. William Beksi**, with **Zongyao Lyu** as graduate mentor. The model is π0 from Physical Intelligence; the benchmark is LIBERO. See [`Literature.md`](../Literature.md).
