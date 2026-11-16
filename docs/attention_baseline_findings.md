# Findings: Baseline Reproduction and Attention Analysis of π0 on LIBERO-Object

**Project:** Failure Analysis and Object Grounding in VLA Models (π0 / LIBERO)
**Author:** Ashzaeen Fatmi Khan
**Last updated:** 2026-07-03
**Status:** Stage 1 complete (reproduce + document); first attention findings; ready for Stage 2/3.

---

## Executive summary

- **Reproduced a trustworthy baseline.** The existing pipeline was scoring **10/100** on
  LIBERO-Object. The cause was a single wrong dependency version (`transformers 5.12.1`); on
  the correct version (`transformers 5.3.0`) the model scores **91/100**. All flat-object
  tasks that were 0/10 recovered.
- **Recorded and analyzed attention for every episode.** A full attention-enabled run scored
  **94/100** (statistically the same as 91/100 — attention capture does not change behavior),
  and produced a heatmap video and per-step attention metrics for all 100 episodes.
- **Two quantitative findings:**
  1. **Attention concentration is strongly object-dependent** — tall, distinctive bottles
     draw sharp attention on the target; flat/boxy objects draw diffuse attention (~5× lower
     peak).
  2. **Failures have measurably more diffuse attention than successes** (~30% lower peak,
     higher entropy) — a real, if correlational, link between *concentration of attention* and
     *task success*.

---

## 1. Getting to a trustworthy baseline (Stage 1)

Before studying *how* the model fails, we had to be sure it was set up correctly. It was not.

**Symptom.** Unperturbed success was **10/100**. Successes were confined to tall bottles;
every flat object (cream cheese, butter, milk, chocolate pudding, …) failed 0/10 — the arm
reached the right region but never grasped precisely.

**What we ruled out.** Normalization statistics (verified real and correct), image resizing,
the config files (identical to the reference), and the checkpoint — none were the cause.

**Root cause: a `transformers` library version mismatch.** The π0 checkpoint's vision encoder
(SigLIP) is named differently across `transformers` versions. The server was running
`transformers 5.12.1`, in which the checkpoint's vision-tower weights no longer matched their
expected names, so **most of the vision encoder silently loaded on random initialization** —
while appearing to load fine. The model was reaching largely from memory, not sight, which is
exactly why tall bottles (roughly fixed, easy positions) sometimes worked and precise grasps
of flat objects never did.

**Fix.** Pin **`transformers==5.3.0`** (+ `torch==2.7.1`, Python 3.12; see
`requirements-working.txt`), on which the checkpoint loads with **0 missing weights**. On the
correct version the model scores:

| Task | Success | (broken 5.12.1) | Task | Success | (broken) |
|---|---|---|---|---|---|
| alphabet soup | 10/10 | 0/10 | tomato sauce | 8/10 | 0/10 |
| cream cheese | 9/10 | 0/10 | butter | 10/10 | 0/10 |
| salad dressing | 10/10 | 6/10 | milk | 9/10 | 0/10 |
| bbq sauce | 9/10 | 0/10 | chocolate pudding | 8/10 | 0/10 |
| ketchup | 8/10 | 0/10 | orange juice | 10/10 | 4/10 |
| | | | **Overall** | **91/100** | **10/100** |

A full list of the pipeline defects found and fixed (8 in total, including two latent code
bugs) is in `docs/stage1_issues_and_reproduction.md`. Resolving and documenting these is, per
the project plan, itself a Stage 1 research contribution.

---

## 2. Method: how attention was recorded and measured

**Recording.** The plain baseline runs with attention off. To study attention we re-ran the
full suite with `include_attentions: true`, so the server returns the model's action-token
attention over all key tokens, saved per timestep into each episode's NPZ.

**What the model attends over.** π0's PaliGemma backbone places **256 base-camera image-patch
tokens at key positions [0:256]** and **256 wrist-camera tokens at [256:512]**, followed by
language, state, and action tokens. For each frame we take the last attention layer, head 0,
averaged over the action-query rows, slice out each camera's 256 patch tokens, and reshape
into a **16×16 heatmap** over that view.

**Metrics per frame** (`scripts/analyze_attention.py` → `sample_level_attention_metrics.csv`):
- **peak** (`max`) — strength of the single most-attended patch (focus vs. spread).
- **entropy** — high = attention spread evenly; low = concentrated.
- **Gini / top-5 share** — alternative concentration measures.
- **peak patch location** — *where* the hottest patch is.

Each episode also gets a **heatmap overlay video** (attention over base + wrist views, per
timestep, with the peak patch circled). Guide to reading them: `docs/how_to_read_attention_videos.md`.

**Important condition.** These runs keep the reference config's `beta = 30` brightness offset
(a mild perturbation baked into every frame), matching the 91/100 baseline. A clean `beta = 0`
run is a pending item (Section 6); we expect it to be at least as high.

---

## 3. Results

### 3.1 Success rate with attention: 94/100

Running the full suite with attention on scored **94/100** — statistically indistinguishable
from the 91/100 no-attention run, confirming that attention capture does not alter behavior
and that the ~9% residual failures are genuine grasp misses (they run the full episode length
without a successful grasp), not crashes.

### 3.2 Attention concentration is object-dependent

Averaged over all steps of all 100 episodes (base camera):

| Object | Peak attention | Entropy (↓ = focused) | Success |
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

The two **tall, visually distinctive bottles** (orange juice, salad dressing) receive sharp,
concentrated attention. Every **flat/boxy** object receives diffuse attention — roughly **5×
lower peak** and higher entropy. The model appears to ground distinctive objects far more
strongly than visually generic ones.

### 3.3 Attention concentration correlates with success

Averaged over all steps, split by episode outcome:

| Episodes | Mean peak | Mean entropy |
|---|---|---|
| **Success** | **0.084** | **3.32** (more focused) |
| **Failure** | **0.058** | **3.76** (more diffuse) |

When the model fails, its base-view attention is **~30% less peaked and more spread out**.
This is direct quantitative evidence linking concentrated (well-grounded) attention to task
success.

### 3.4 Per-episode audit (all 10 trials of four objects)

Beyond the aggregate metrics, every trial (10 each) of four objects was **visually audited**;
the per-object contact sheets and the representative videos are under `report_videos/`
(`audit_contact_sheets/`). Counts are exact frequencies from that audit.

- **Orange juice — grounding, 10/10.** The base-camera attention peak sits directly on the
  target bottle in **all 10 trials** (peak ~0.31). Consistent, textbook grounding.
- **Alphabet soup — shortcut signature, 10/10.** In **all 10 trials** the peak sits on the far
  background band, never the soup can, yet **all 10 grasps succeed**. Success does not require
  looking at the target — the clearest repeatable shortcut signature observed.
- **Chocolate pudding — diffuse base; wrist-framed (interpretive).** Base attention is diffuse
  in all 10 trials; by the grasp the pudding box is framed near the center of the wrist camera
  in **~8/10**. However, wrist attention magnitude is ≈0, so we read the wrist *view* (via the
  approach trajectory) as *plausibly* carrying the grasp — an **interpretation**, not
  attention-based evidence that the model looks at the target there.
- **Milk — failures show competing attention (subtle).** **2/10 trials fail**; in the failures
  base attention is more diffuse and split, with a competing hotspot away from the target and
  the arm never reaching the carton, whereas successes concentrate more on one object. The
  effect is **real but modest**, clearest in the failed episode shown.

Together: π0 grounds distinctive objects reliably (10/10 on-target), does not always need to
attend to the target to succeed on generic ones (soup: 10/10 off-target yet 10/10 success), and
shows more diffuse/competing attention on the trials that fail.

---

## 4. Interpretation

- The model **does** ground distinctive targets: sharp, on-target attention for tall bottles,
  and concentrated attention correlates with success overall.
- But it does **not always** need to look at the target to succeed — several flat-object
  grasps succeed with diffuse base-view attention. Combined with the wrist camera and learned
  trajectories, this is consistent with **partial reliance on shortcuts** (memorized
  positions/motions) for the less distinctive objects.
- The clearest failure mode observed is **distractor competition**: when the scene contains
  several similar boxes, attention splits and grasps fail.

These are exactly the patterns the project set out to detect. They motivate the Stage 3
experiments that can separate genuine grounding from shortcuts.

---

## 5. Limitations

- **Correlation, not causation.** Diffuse attention and failure co-occur, but hard-to-ground
  objects are *both* diffuse *and* failure-prone; we cannot yet claim diffuse attention
  *causes* failure.
- **Attention is a diagnostic, not proof.** It is one layer, one head, mean-pooled over action
  queries; the model also has a wrist camera and learned motion. Attention flags hypotheses;
  it does not confirm grounding on its own.
- **`beta = 30` is on.** All runs include a mild brightness offset; a clean `beta = 0` baseline
  is pending.
- **Single-view metric.** The quantitative tables use the base camera; wrist-view attention is
  recorded but not yet summarized.
- **10 trials/task.** Enough to see the pattern; larger N would tighten the success/attention
  correlation.

---

## 6. Next steps

1. **Clean `beta = 0` baseline** for the canonical unperturbed number.
2. **Stage 2 — controlled perturbations:** vary one factor at a time (brightness, object
   position, robot state) with baseline + multiple trials + degradation metrics.
3. **Stage 3 — the decisive grounding test:** **move the target object and re-measure.** If
   attention tracks the object and the grasp still succeeds → genuine grounding; if attention
   stays at the old location or the grasp fails once the object moves → a **position shortcut**.
   Add-distractor experiments should directly test the distractor-competition failure mode
   observed for milk.

---

## Reproducibility

- Environment: `requirements-working.txt` (`transformers==5.3.0`).
- Baseline: `python scripts/run_libero_eval.py --config <runner>.yaml`.
- Attention: same, with `include_attentions: true`, then
  `python scripts/analyze_attention.py --npz_dir outputs/<run>/episodes_npz --output_dir outputs/<run>/attention_analysis`.
- Artifacts from this report live under `outputs/baseline_attn_full/` (100 episodes: NPZs,
  `results.json`, 100 heatmap videos, and `sample_level_attention_metrics.csv`).
