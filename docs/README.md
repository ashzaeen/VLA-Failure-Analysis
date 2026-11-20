# Documentation

Everything in `docs/`, grouped by what you are trying to do. For the research results themselves, see [`../research/`](../research/README.md).

## Setup: Get the Pipeline Running

| Document | Covers |
|---|---|
| [`setup/Server-Setup.md`](setup/Server-Setup.md) | The model server: environment, checkpoint download, configuration and startup (π0 on LIBERO, and UR5) |
| [`setup/Client-Setup.md`](setup/Client-Setup.md) | The LIBERO client: environment, runner configuration, startup and where results are recorded |

The one thing to get right is the `transformers==5.3.0` pin; see [`../requirements-working.txt`](../requirements-working.txt).

## Guides: How to Do Things

| Document | Covers |
|---|---|
| [`guides/Configuration-and-Outputs.md`](guides/Configuration-and-Outputs.md) | Every runner setting, the files a run writes, the attention metrics, and how to reproduce the headline numbers |
| [`guides/Sweeps-and-Scene-Tooling.md`](guides/Sweeps-and-Scene-Tooling.md) | The single-factor sweep driver, proprioception noise and the scene-perturbation system |
| [`guides/Object-Position-Experiments.md`](guides/Object-Position-Experiments.md) | Why and how the Stage 3 object-position experiments work, including the six-slot finding |
| [`guides/Reading-Attention-Videos.md`](guides/Reading-Attention-Videos.md) | How to read the attention heat-map videos |

## Project Log: The Working Record

Written as the work happened, in order, and kept as it was.

| Document | Covers |
|---|---|
| [`project-log/2026-06-19-Research-Plan.md`](project-log/2026-06-19-Research-Plan.md) | The original summer research plan |
| [`project-log/2026-06-25-Stage-1-Reproduction-Log.md`](project-log/2026-06-25-Stage-1-Reproduction-Log.md) | A plain-language log of every problem hit while reproducing the baseline, and the fixes |
| [`project-log/2026-07-02-Baseline-Reproduction-Daily-Log.md`](project-log/2026-07-02-Baseline-Reproduction-Daily-Log.md) | Day-by-day log of the baseline reproduction |
| [`project-log/2026-07-03-Baseline-and-Attention-Findings.md`](project-log/2026-07-03-Baseline-and-Attention-Findings.md) | Baseline and per-object attention findings |
| [`project-log/2026-07-14-Baseline-and-Brightness-Progress.md`](project-log/2026-07-14-Baseline-and-Brightness-Progress.md) | Baseline and brightness-sweep progress report |
| [`project-log/2026-07-14-Stages-1-2-Progress-Report.md`](project-log/2026-07-14-Stages-1-2-Progress-Report.md) | Stages 1–2 progress report, July 14, 2026 |

## Elsewhere in the Repository

- [`assets/readme/`](assets/readme/): the hero and button images used by the main README.
- [`../tools/lab-server/README.md`](../tools/lab-server/README.md): the run scripts used on the lab server.
- [`../research/summer-failure-analysis/progress-reports/`](../research/summer-failure-analysis/progress-reports/README.md): the weekly progress reports sent to the advisor.
- [`../research/Literature.md`](../research/Literature.md): the papers this work builds on.
