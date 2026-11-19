# Documentation

Everything in `docs/`, grouped by what you are trying to do. For the research results themselves, see [`../research/`](../research/README.md).

## Setup: get the pipeline running

| Document | Covers |
|---|---|
| [`setup/server-setup.md`](setup/server-setup.md) | The model server: environment, checkpoint download, configuration and startup (π0 on LIBERO, and UR5) |
| [`setup/client-setup.md`](setup/client-setup.md) | The LIBERO client: environment, runner configuration, startup and where results are recorded |

The one thing to get right is the `transformers==5.3.0` pin; see [`../requirements-working.txt`](../requirements-working.txt).

## Guides: how to do things

| Document | Covers |
|---|---|
| [`guides/configuration-and-outputs.md`](guides/configuration-and-outputs.md) | Every runner setting, the files a run writes, the attention metrics, and how to reproduce the headline numbers |
| [`guides/sweeps-and-scene-tooling.md`](guides/sweeps-and-scene-tooling.md) | The single-factor sweep driver, proprioception noise and the scene-perturbation system |
| [`guides/object-position-experiments.md`](guides/object-position-experiments.md) | Why and how the Stage 3 object-position experiments work, including the six-slot finding |
| [`guides/reading-attention-videos.md`](guides/reading-attention-videos.md) | How to read the attention heat-map videos |

## Project log: the working record

Written as the work happened, in order, and kept as it was.

| Document | Covers |
|---|---|
| [`project-log/research-plan.md`](project-log/research-plan.md) | The original summer research plan |
| [`project-log/stage-1-reproduction-log.md`](project-log/stage-1-reproduction-log.md) | A plain-language log of every problem hit while reproducing the baseline, and the fixes |
| [`project-log/baseline-reproduction-daily-log.md`](project-log/baseline-reproduction-daily-log.md) | Day-by-day log of the baseline reproduction |
| [`project-log/baseline-and-attention-findings.md`](project-log/baseline-and-attention-findings.md) | Baseline and per-object attention findings |
| [`project-log/baseline-and-brightness-progress.md`](project-log/baseline-and-brightness-progress.md) | Baseline and brightness-sweep progress report |
| [`project-log/progress-report-2026-07-14.md`](project-log/progress-report-2026-07-14.md) | Stages 1–2 progress report, July 14, 2026 |

## Elsewhere in the repository

- [`../tools/lab-server/README.md`](../tools/lab-server/README.md): the run scripts used on the lab server.
- [`../research/progress-reports/`](../research/progress-reports/README.md): the weekly progress reports sent to the advisor.
- [`../research/literature.md`](../research/literature.md): the papers this work builds on.
