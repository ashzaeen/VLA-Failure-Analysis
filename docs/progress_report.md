# Progress Report — Failure Analysis and Object Grounding in π0 (VLA) on LIBERO

**Author:** Ashzaeen Fatmi Khan
**Advisor:** the lab PI
**Report date:** 2026-07-14
**Model:** π0 (pi-zero) VLA · **Benchmark:** LIBERO-Object · **Environment:** `transformers 5.3.0`, `torch 2.7.1+cu128`, Python 3.12

---

## Abstract

This report covers the first two stages of a study into whether the π0 vision-language-action
(VLA) model genuinely grounds the instructed object or succeeds via shortcuts. **Stage 1**
established a trustworthy baseline: the shared pipeline was scoring a pathological **10/100** on
LIBERO-Object; we traced this to a single wrong dependency version (`transformers 5.12.1`) that
silently loaded the vision encoder on random weights, and fixed it (`transformers 5.3.0`),
recovering **91/100**. We also recorded and analyzed the model's attention for every episode
(**94/100** with attention on). **Two attention findings** emerged and were verified by a
full per-trial audit: (1) attention concentration is strongly **object-dependent** (distinctive
objects grounded sharply, generic objects diffusely), and (2) **success does not require attending
to the target** for several objects — a shortcut signature. **Stage 2** ran a controlled
brightness sweep (β = 0…100): success is **robust to brightness** (86–96%, no trend), yet
attention **sharpens markedly** as brightness rises — a decoupling of attention pattern from
behavior. All results, artifacts, and reproduction steps are catalogued at the end.

---

## 1. Motivation and research question

VLA models map camera images + a language instruction to robot actions. The core question of
this project:

> When told to "pick up the orange juice and place it in the basket," does π0 actually **locate
> the orange juice** and act on it — or does it exploit a **shortcut** (a memorized position, a
> learned trajectory, or instruction-token → action mapping) that happens to work in the training
> distribution but would fail under perturbation?

We probe this with **task success** (does it work?) and **attention maps** (where does it look?),
interpreted together. Four shortcut types are of interest: **position**, **state**, **language**,
and **trajectory**.

---

## 2. Stage 1 — Reproducing a trustworthy baseline

Before studying *how* the model fails, we had to confirm it was set up correctly. It was not.

### 2.1 Symptom
Unperturbed success was **10/100**. Successes were confined to tall bottles; every flat object
(cream cheese, butter, milk, chocolate pudding, …) failed **0/10**. The arm reached the right
region but never grasped precisely — the fingerprint of a model that can *sort of* see but not
localize.

### 2.2 What we ruled out
Methodically cleared as causes: **normalization statistics** (verified real and correct via a
dedicated `diagnose_norm_stats.py`), **image resizing**, the **config files** (identical to the
reference), and the **checkpoint**.

### 2.3 Root cause — a `transformers` version mismatch
π0's SigLIP vision encoder is named differently across `transformers` versions. The server ran
`transformers 5.12.1`, in which the checkpoint's vision-tower weights no longer matched their
expected names, so **most of the vision encoder silently loaded on random initialization** while
appearing to load fine. The model was reaching largely from memory, not sight — exactly why tall
bottles (roughly fixed positions) sometimes worked and precise grasps never did.

**Fix:** pin **`transformers==5.3.0`** (the version where the checkpoint loads with **0 missing
weights**). The repo's own `requirements-ubuntu.txt` pin (`4.57.1`) is stale and *crashes*; the
`pyproject` floor `>=5.3.0` was the correct signal. See `requirements-working.txt`.

### 2.4 Other pipeline defects found and fixed
Documented fully in `docs/stage1_issues_and_reproduction.md` (8 issues). Highlights: missing
config files, a transformers-5 import crash (groot dataclass), a gated-tokenizer blocker, an
evaluator that only ran 2 of 10 tasks and mislabeled episodes, and a latent 5-vs-6 unpack bug that
crashed the server whenever attention was requested. Resolving and documenting these is itself the
Stage 1 contribution the project plan calls for.

### 2.5 Reproduced baseline
On the corrected environment, LIBERO-Object (10 tasks × 10 trials, `beta=30`):

| Task | Success | (broken 5.12.1) | Task | Success | (broken) |
|---|---|---|---|---|---|
| alphabet soup | 10/10 | 0/10 | tomato sauce | 8/10 | 0/10 |
| cream cheese | 9/10 | 0/10 | butter | 10/10 | 0/10 |
| salad dressing | 10/10 | 6/10 | milk | 9/10 | 0/10 |
| bbq sauce | 9/10 | 0/10 | chocolate pudding | 8/10 | 0/10 |
| ketchup | 8/10 | 0/10 | orange juice | 10/10 | 4/10 |
| | | | **Overall** | **91/100** | **10/100** |

---

## 3. Method — attention extraction and analysis

The plain baseline runs with attention off. To study attention, episodes are re-run with
`include_attentions: true`; the server returns the model's last-layer action-token attention over
all key tokens, saved per timestep. π0's PaliGemma backbone places **256 base-camera patch tokens
at key positions [0:256]** and **256 wrist-camera tokens at [256:512]**, then language/state/action
tokens. For each frame we take the last layer, head 0, averaged over action queries, slice each
camera's 256 patch tokens, and reshape into a **16×16 heatmap**.

**Per-frame metrics** (`scripts/analyze_attention.py`): **peak** (strength of the hottest patch —
focus vs. spread), **entropy** (low = concentrated), Gini / top-k share, and peak-patch location.
Each episode also gets a heatmap overlay **video**. Reading guide:
`docs/how_to_read_attention_videos.md`.

**Caveat used throughout:** attention is a **diagnostic, not proof**. It is one layer, one head,
mean-pooled; the model also has a wrist camera and learned motion. Attention flags hypotheses to
be confirmed by behavior.

---

## 4. Results — attention baseline

### 4.1 Success with attention: 94/100
The full suite with attention on scored **94/100** — statistically the same as the 91/100
attention-off run, confirming attention capture does not change behavior and the residual failures
are genuine grasp misses (they run the full episode without a successful grasp).

### 4.2 Attention concentration is object-dependent
Averaged over all steps of all 100 episodes (base camera):

| Object | Peak | Entropy (↓=focused) | Success |
|---|---|---|---|
| **orange juice** | **0.283** | **2.00** | 10/10 |
| salad dressing | 0.110 | 2.88 | 10/10 |
| alphabet soup | 0.075 | 2.91 | 10/10 |
| milk | 0.066 | 3.62 | 8/10 |
| ketchup | 0.064 | 3.41 | 9/10 |
| tomato sauce | 0.062 | 3.90 | 9/10 |
| butter | 0.053 | 3.12 | 10/10 |
| bbq sauce | 0.050 | 3.99 | 8/10 |
| cream cheese | 0.049 | 3.38 | 10/10 |
| chocolate pudding | 0.047 | 3.97 | 10/10 |

The two tall, visually distinctive bottles receive sharp, concentrated attention; every flat/boxy
object receives diffuse attention (~5× lower peak).

### 4.3 Attention concentration correlates with success
Averaged over all steps, split by episode outcome:

| Episodes | Mean peak | Mean entropy |
|---|---|---|
| **Success** | **0.084** | **3.32** (more focused) |
| **Failure** | **0.058** | **3.76** (more diffuse) |

Failing episodes have ~30% less-peaked, more-diffuse attention — a quantitative link between
concentration and success. *(Correlation, not causation: hard-to-ground objects are both diffuse
and failure-prone.)*

### 4.4 Per-trial audit (all 10 trials of four objects)
Every trial (10 each) of four objects was **visually audited** (contact sheets in
`report_videos/audit_contact_sheets/`). Exact frequencies:

- **Orange juice — grounding, 10/10.** Base-camera peak sits on the target bottle in **all 10**
  trials. Textbook grounding.
- **Alphabet soup — shortcut signature, 10/10.** In **all 10** trials the peak is on the far
  **background**, never the soup can, yet **all 10 succeed**. Success without looking at the
  target — the clearest repeatable shortcut signature observed.
- **Chocolate pudding — diffuse base; wrist-framed (interpretive).** Base diffuse in all 10; the
  box is framed near the center of the wrist camera at grasp in **~8/10**. But wrist attention
  magnitude is ≈0, so the wrist *view* (approach trajectory) *plausibly* carrying the grasp is an
  **interpretation**, not attention-based evidence.
- **Milk — competing attention on failures (subtle).** **2/10 fail**; failures show more diffuse,
  split attention with a competing hotspot away from the target and the arm never reaching the
  carton, whereas successes concentrate more. Real but modest; clearest in the failed episode.

Representative clips: `report_videos/1_orange_juice…`, `2_alphabet_soup…`, `3_chocolate_pudding…`,
`4_milk…FAILURE.mp4`.

---

## 5. Results — Stage 2, controlled brightness sweep

A brightness offset β was added to every frame, swept **β = 0, 10, …, 100** at 100 episodes each,
**same seed** across conditions (only brightness varies), attention on. β = 0 is the true
unperturbed anchor (the earlier baseline used β = 30).

| β | Success | Base peak | Entropy | | β | Success | Base peak | Entropy |
|---|---|---|---|---|---|---|---|---|
| 0 | 87/100 | 0.073 | 3.74 | | 60 | 86/100 | 0.144 | 2.40 |
| 10 | 91/100 | 0.088 | 3.55 | | 70 | 90/100 | **0.254** | **1.41** |
| 20 | 95/100 | 0.109 | 3.28 | | 80 | 91/100 | 0.236 | 1.92 |
| 30 | 92/100 | 0.080 | 3.39 | | 90 | 87/100 | 0.152 | 2.79 |
| 40 | 96/100 | 0.100 | 3.08 | | 100 | 92/100 | 0.120 | 3.09 |
| 50 | 96/100 | 0.112 | 2.84 | | | | | |

**Two findings:**
1. **Success is robust to brightness.** It stays in the **86–96%** band across the entire range
   with no downward trend — additive brightness does not break the model (consistent with SigLIP's
   internal normalization largely canceling a constant offset).
2. **Attention sharpens with brightness.** Peak rises (0.07→0.25) and entropy falls (3.74→1.41),
   sharpest at **β ≈ 70–80**, then relaxes at β = 100. So the attention *pattern* changes
   substantially while *behavior* does not — a decoupling that reinforces §4.3–4.4: concentrated
   on-target attention is not what drives success here.

---

## 6. Discussion

- π0 **does** ground distinctive targets (orange juice: sharp, on-target attention, 10/10) and, in
  aggregate, concentrated attention tracks success.
- But it does **not always need to look at the target** to succeed — alphabet soup succeeds 10/10
  with attention consistently on the background. This is a **shortcut signature**: for generic
  objects the model likely leans on memorized position / trajectory plus the wrist view, rather
  than base-camera grounding.
- The brightness sweep strengthens this: the model tolerates a large visual perturbation with no
  behavioral cost even as its attention pattern shifts dramatically — behavior is not tightly
  coupled to where the base-camera attention lands.
- The clearest failure mode is **distractor competition** (milk), where attention splits across
  similar objects.

These are exactly the patterns needed to motivate the decisive Stage 3 experiment.

---

## 7. Limitations

- **Attention is diagnostic, not proof** (one layer/head, mean-pooled; wrist camera + learned
  motion also contribute).
- **Correlation, not causation** for the concentration↔success link.
- **Base-view metric.** Quantitative tables use the base camera; wrist attention is recorded but
  largely ≈0 in magnitude and not yet summarized.
- **Brightness is one perturbation type** and an "easy" one (additive, partly normalized away);
  it does not test grounding directly.
- **10 trials/task**; larger N would tighten estimates.

---

## 8. Conclusions and next steps

**Conclusions.** The corrected pipeline reproduces a ~90% LIBERO-Object baseline. Attention
analysis shows object-dependent grounding and clear evidence that **success does not require
attending to the instructed object** for several objects — a shortcut signature — while the model
is robust to brightness perturbation.

**Next steps (Stage 3 — the decisive grounding test).** Attention videos give hypotheses; only
*moving the world* gives proof:
1. **Move the target object** and re-measure — if attention tracks it and the grasp still
   succeeds, grounding is real; if attention stays at the old location or the grasp fails, it was a
   **position shortcut**. (The evaluator already exposes the `set_init_state` hook for this.)
2. **Add / vary distractors** — directly test the milk-style distractor-competition failure.
3. **Rephrase / swap the instruction** (`task_description` override) — test **language** shortcuts.
4. A clean **β = 0** baseline is now available (87/100) as the anchor for all perturbations.

---

## Appendix — reproducibility and artifacts

**Environment:** `requirements-working.txt` (`transformers==5.3.0`, `torch==2.7.1+cu128`, py3.12).

**Commands:**
```
# baseline
python scripts/run_libero_eval.py --config <runner>.yaml
# attention: same with include_attentions: true, then
python scripts/analyze_attention.py --npz_dir outputs/<run>/episodes_npz \
    --output_dir outputs/<run>/attention_analysis --head_idx 0 --patch_grid_size 16 --top_k 5
```

**Artifacts:**
- `outputs/baseline_attn_full/` — 100-episode attention baseline (NPZs, results.json, 100 heatmap
  videos, per-step metrics CSV).
- `outputs/beta_sweep_attn/` — brightness sweep: `SWEEP_SUMMARY.csv` (the two curves), per-β
  `results.json` and per-step metrics CSVs. (Raw NPZs and 1,100 heatmap videos remain on the lab
  server; pulled on request.)
- `report_videos/` — 4 representative clips + captions + all-10-trials audit contact sheets.

**Companion docs:** `stage1_issues_and_reproduction.md` (defect log), `attention_baseline_findings.md`
(detailed findings), `how_to_read_attention_videos.md` (reading guide), `progress.md` (working log).
