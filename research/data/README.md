# Data

The numbers behind the reports as small CSV tables. Every row names the weekly report it comes from (in [`../progress-reports/`](../progress-reports/)).

| File | Contents |
|---|---|
| [`baselines.csv`](baselines.csv) | The four baseline configurations: 10/100 → 91/100 → 94/100 → clean 87/100 |
| [`brightness_sweep_summary.csv`](brightness_sweep_summary.csv) | Brightness sweep summary (offsets 0–100, 100 episodes each) |
| [`contrast_sweep.csv`](contrast_sweep.csv) | Contrast sweep, gains 0.2×–2.0×, 100 episodes each |
| [`proprioception_noise_sweep.csv`](proprioception_noise_sweep.csv) | Robot-state noise sweep, 0–16 cm, 100 episodes each |
| [`attention_audit.csv`](attention_audit.csv) | Per-object attention audit on the clean baseline, 10 trials each |
| [`stage3_scene_perturbations.csv`](stage3_scene_perturbations.csv) | Swap and displacement conditions as reported, 20 episodes each |
| [`stage3_by_distance.csv`](stage3_by_distance.csv) | Per-distance success counts, **counted directly from the clips** in [`../evidence/stage3-videos/`](../evidence/stage3-videos/) (they match the report: 16/20 anchor, 2/20 at 5 cm, 0/20 swap, 17–20/20 for distractors) |

## Provenance

- These tables were **transcribed from the weekly progress reports**, not regenerated from the raw per-run files (`SWEEP_SUMMARY.csv`, `sample_level_attention_metrics.csv`), which stay on the lab server. The 1,600-episode sweep data is about 102 GB.
- Clean baselines from different runs were 84%, 86% and 87%, so differences of about 4 points are within run-to-run noise.
- Success is out of 100 episodes (10 tasks × 10 trials) unless a table says otherwise.
