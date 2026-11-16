# Progress Update — New (Clean) Baseline and Brightness Perturbations

**Author:** Ashzaeen Fatmi Khan · **Advisor:** the lab PI
**Date:** 2026-07-14 · **Model:** π0 (VLA) · **Benchmark:** LIBERO-Object

---

## 1. Summary

- **Previous baseline** was actually run with a small brightness offset baked in (**β = 30**) and
  scored **91/100**.
- I established a proper **new, clean baseline** with **no brightness perturbation (β = 0)**, which
  scored **87/100** — **essentially the same** as the previous β = 30 baseline.
- I then ran a full **brightness sweep (β = 0 → 100)**: success stays **robust (86–96%)** across the
  whole range, so brightness is not a meaningful failure mode for this model.

The main outcome is a **true unperturbed anchor (β = 0)** that every future perturbation can be
measured against, plus the finding that the model is essentially **brightness-invariant**.

---

## 2. New (clean) baseline: β = 0 vs. the previous β = 30

The earlier "baseline" runs inherited a config value of **β = 30** — i.e. a constant **+30
brightness offset was added to every camera frame** before the model saw it. That is technically a
mild visual perturbation, not a truly clean condition, so the number it produced (91/100) was not a
proper unperturbed baseline.

I re-ran the identical evaluation with **β = 0** (no brightness change at all) to get the real,
clean baseline:

| Baseline | Brightness | Success |
|---|---|---|
| Previous | β = 30 (mild offset baked in) | **91/100** |
| **New (clean)** | **β = 0 (no perturbation)** | **87/100** |

The two are **within normal run-to-run variation** (~4/100), so removing the brightness offset did
**not** change performance. This is useful for two reasons:

1. We now have a **legitimate unperturbed anchor** (β = 0 → 87/100) for all future comparisons,
   instead of a baseline that already had a perturbation folded in.
2. The near-equality is itself an early signal that the model is **insensitive to brightness** — a
   hypothesis the sweep below then confirms directly.

*(Context: this assumes the corrected pipeline. The pipeline previously scored only 10/100 due to a
wrong `transformers` library version that loaded the vision encoder on random weights; fixing that
to `transformers 5.3.0` is what recovered the ~90% baseline. Details in
`docs/stage1_issues_and_reproduction.md`.)*

---

## 3. Brightness-perturbation sweep

### Setup
Starting from the clean β = 0 baseline, I swept the brightness offset **β = 0, 10, 20, …, 100** —
11 conditions, **100 episodes each** (10 tasks × 10 trials), with the **same random seed** across
all conditions so that *only* brightness changes. Attention was also recorded for every episode.

### Results

| β | Success | | β | Success |
|---|---|---|---|---|
| **0 (clean)** | **87/100** | | 60 | 86/100 |
| 10 | 91/100 | | 70 | 90/100 |
| 20 | 95/100 | | 80 | 91/100 |
| **30 (old baseline)** | **92/100** | | 90 | 87/100 |
| 40 | 96/100 | | 100 | 92/100 |
| 50 | 96/100 | | | |

**Finding: success is robust to brightness.** Across the entire β = 0 → 100 range the success rate
stays in the **86–96%** band with no downward trend. Adding brightness — even a large +100 offset —
does not degrade performance, most likely because the model's vision encoder normalizes images
internally and largely cancels a constant offset. (The β = 30 point, 92/100, also lines up with the
old baseline, confirming consistency.)

*Secondary observation:* although success is flat, the model's **attention becomes more concentrated
as brightness rises** (the most-attended patch strengthens and the attention spreads less), peaking
around β ≈ 70–80. So its internal attention pattern shifts even though its behavior does not — a
hint that success here does not depend heavily on where the attention lands. (Full attention
metrics per β are in `outputs/beta_sweep_attn/SWEEP_SUMMARY.csv`.)

---

## 4. Takeaways and next steps

- There is now a **clean β = 0 baseline (87/100)** as the reference point, replacing the previous
  β = 30 baseline; the two are equivalent, so past results remain comparable.
- **Brightness is not a meaningful failure mode** — a useful negative result and a good control
  showing the evaluation behaves sensibly.
- **Next**, I will move to perturbations that test *grounding* rather than appearance — **moving the
  target object**, **adding distractors**, and **rephrasing the instruction** — measuring both the
  drop in success and whether the model's attention follows the object.

---

*Data:* per-condition success rates and attention metrics are in
`outputs/beta_sweep_attn/SWEEP_SUMMARY.csv`. Environment: `requirements-working.txt`
(`transformers==5.3.0`).
