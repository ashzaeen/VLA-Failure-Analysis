# How to Read the Attention Heatmap Videos — A Plain-Language Guide

**Project:** Failure Analysis and Object Grounding in VLA Models (π0 / LIBERO)
**Author:** Ashzaeen Fatmi Khan
**Last updated:** 2026-07-03

---

## Why This Guide Exists

The central question of this project is: **does π0 actually look at the object it was told
to pick up, or does it succeed by memorizing positions and movements?** The attention
heatmap videos are our main tool for investigating that. But a heatmap is only useful if you
know how to read it. This guide explains, in plain terms, what these videos show, how to
interpret them, and — most importantly — how to decide what they *mean*.

---

## 1. What You Are Looking at

Each video is **one episode** — one attempt at one task (e.g. "pick up the orange juice and
place it in the basket"), played back frame by frame. Watching it is like **watching where
the robot's eyes go** while it works. Every frame has three parts:

- **Bottom-left — the overhead ("base") camera.** The whole tabletop scene.
- **Bottom-right — the wrist camera.** Mounted on the gripper; a close-up of whatever the
  hand is near.
- **The colored splotches over those images — the attention heatmap.** This is the point.
- **Top strip — the raw attention numbers** (action tokens × key tokens). Mostly ignore this
  for intuition; it is the same information in raw form.

## 2. What "Attention" Means

The model sees each camera image as a grid of little squares ("patches"). For every action
it decides, it effectively asks *"which parts of the image mattered for this decision?"*
Attention is the model's own answer.

- **Warm colors (red / orange / yellow) = "I looked here a lot."**
- **Cool / dark (blue / none) = "I ignored this."**
- **The green circle = the single hottest spot** — where it looked *most*.

So the heatmap literally shows **where the model's attention went** at that moment.

## 3. The One Number That Matters Most: `max`

Each panel shows a `max:` value (e.g. orange juice base `max: 0.28`, alphabet soup
`max: 0.058`). It is **how strong the strongest spot is**, which tells you *how focused* the
model is:

- **High `max` (bright, tight blob) → locked onto one spot.** Focused, confident looking.
- **Low `max` (dim, smeared) → attention spread thin.** Not really focusing on anything.

(The CSV `sample_level_attention_metrics.csv` also gives **entropy** and **Gini** per frame —
both are just numeric versions of "focused vs. spread": low entropy / high Gini = focused;
high entropy / low Gini = spread out. `top5_share` = what fraction of all attention the 5
hottest patches got.)

## 4. The Core Question You Ask While Watching

Just one:

> **"Is the model looking at the object I told it to pick up?"**

While the video plays, track the green circle / red blob and ask:

1. **Where is it?** On the target object? On a *different* object (a distractor)? On the
   robot arm? On empty floor?
2. **Is it focused or smeared?** (the `max` value and how tight the blob is)
3. **Does it follow the target** as the arm moves, or does it wander?

## 5. How to Decide What It Means — The 2×2

Attention on its own doesn't tell you much. The trick is to **pair "where it looked" with
"did the episode succeed"**. That gives four cases:

|                              | **Task succeeded** ✅                                                                 | **Task failed** ❌                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Looked at the target**     | 🟢 **Genuine grounding** — saw the object and grabbed it. The good case.               | 🟡 **Motor / grasp problem** — found the object but fumbled execution. Not a grounding issue. |
| **Did NOT look at the target** | 🔴 **Shortcut smell** — succeeded *without* looking at the object. May be relying on a memorized position or trajectory. | ⚪ **Grounding failure** — never located the object, and failed. Expected.   |

The **🔴 red box is the most important for this project**: success *without* looking at the
target is the fingerprint of a shortcut.

## 6. Two Rules That Keep the Analysis Honest

- **Watch the whole video, not one frame.** Attention shifts over time — early on it often
  scans broadly (approach phase) and may tighten near the grasp. A single mid-episode frame
  can mislead. Judge the *trajectory* of where it looks.
- **Attention is a clue, not a verdict.** The model has *two* cameras plus learned motion. It
  can succeed with diffuse base-view attention because the **wrist camera** or a memorized
  movement carried it. So "attention was not on the target" *flags* a possible shortcut — it
  does not *prove* one.

## 7. Where This Leads — The Real Test

A single video can only *suggest* grounding vs. shortcut. The decisive experiment (Stage 2/3)
is **comparison — move the target object and re-watch:**

- If attention **moves with the object** and the grasp still succeeds → **real grounding.** ✅
- If attention **stays at the old location**, or the model fails once the object moves →
  **it was a position shortcut all along.** 🔴

Attention videos give you the *hypothesis*; moving things around and re-measuring gives you
the *proof*.

## 8. Worked Examples (From the 10-Episode Attention Run, 2026-07-03)

| Object | Base `max` | Where attention landed | Outcome | Reading |
| --- | --- | --- | --- | --- |
| Orange juice (tall bottle) | 0.28 (sharp) | on the target bottle | ✅ | 🟢 clean grounding |
| Alphabet soup (flat) | 0.058 (diffuse) | background / table edge | ✅ | 🔴 shortcut smell |
| Chocolate pudding (flat) | 0.044 (diffuse) | near gripper; target centered in *wrist* view | ✅ | 🔴/wrist-carried |
| Milk (busy scene) | 0.078 (split across distractors) | never locked on target | ❌ | ⚪ grounding failure under clutter |

Takeaways from this small sample: attention is **sharp and on-target for distinctive tall
bottles**, but **diffuse for flat/boxy objects** — yet those still often succeed, which is the
shortcut smell. The one failure lined up with **attention split across similar distractors**.

## 9. How to Regenerate These Videos

Attention must be recorded during the rollout (it is *not* in the plain baseline, which ran
with `include_attentions: false`). Two steps:

```bash
# 1) Re-run episodes with attention ON (runner config: include_attentions: true, batch_size: 1)
python scripts/run_libero_eval.py --config <runner-with-include_attentions-true>.yaml

# 2) Turn the saved attention into heatmap videos + a per-step metrics CSV
python scripts/analyze_attention.py \
    --npz_dir   outputs/<run>/episodes_npz \
    --output_dir outputs/<run>/attention_analysis \
    --head_idx 0 --patch_grid_size 16 --top_k 5
```

Outputs land in `outputs/<run>/attention_analysis/`: one `heatmaps/*.mp4` per episode and
`sample_level_attention_metrics.csv` (entropy / Gini / top-k share / peak patch per frame).

Token layout note (for the curious): π0's PaliGemma backbone puts **256 base-image patch
tokens at key positions [0:256]** and **256 wrist-image tokens at [256:512]**, followed by
language, state, and action tokens. The analyzer slices out those image ranges and reshapes
each into a 16×16 grid to draw the heatmap.
