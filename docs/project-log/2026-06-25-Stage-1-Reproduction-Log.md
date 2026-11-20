# Stage 1 — Getting the Pipeline Running: A Plain-Language Log

**Project:** Failure Analysis and Object Grounding in VLA Models (π0 / LIBERO)
**Stage:** 1 — *Understand and reproduce the existing work*
**Author:** Ashzaeen Fatmi Khan
**Last updated:** 2026-06-25

---

## How to read this document

This is a diary of everything that went wrong while getting the existing code to run, and
how each problem was fixed. It's written so that **someone brand new to this project — or
even new to this kind of robotics/AI setup — can follow along.** Jargon is explained the
first time it appears, and every problem is told as a little story: *what we saw, what was
actually wrong, why it mattered, and what we did about it.*

Why bother writing all this down? Because the project plan explicitly says so:

> *Carefully document problems such as missing instructions, inconsistent settings,
> incomplete scripts, unstable behavior, or results that cannot be reproduced. Resolving
> and documenting these issues is itself a meaningful research contribution.*

In plain terms: before we can study *how a robot AI fails*, we have to be absolutely sure
the AI is set up correctly in the first place. Otherwise we'd be studying *our own setup
bugs* instead of the model. This document is the proof that the foundation is solid.

---

## The 30-second background

A few terms show up constantly. Here's all you need to know:

- **VLA model (Vision-Language-Action).** An AI that looks at camera images, reads a text
  instruction (e.g. *"pick up the orange juice and put it in the basket"*), and outputs
  robot movements. **π0 ("pi-zero")** is the specific VLA model we study.
- **LIBERO.** A library of simulated tabletop robot tasks. It runs a physics simulator
  (**MuJoCo**) and produces the camera images the model sees. Think of it as the
  "video game world" the robot lives in.
- **The two halves of the pipeline.** The code is split into two programs that talk to
  each other over a local network connection (HTTP):
  1. the **policy server** — loads π0 and answers the question *"given these images and
     this instruction, what should the robot do next?"*
  2. the **evaluation client** — runs the LIBERO simulation, sends each camera frame to
     the server, and applies the movements the server sends back.

  They're split in two because each half needs a different, conflicting set of software.
  Keeping them separate (in two **conda environments** — isolated Python installations)
  lets each one have exactly the libraries it needs without stepping on the other.
- **Where it all runs.** Everything heavy runs on a lab server called **"bender"** (it has
  two big NVIDIA A40 GPUs). A laptop just sends commands to it over SSH.

That's the whole mental model. Now, the problems.

---

## The problems we hit (and fixed)

Each problem below follows the same simple shape:
**What we saw → What was really wrong → Why it mattered → What we did → Status.**

---

### Problem 1 — The settings files were simply missing

**What we saw.** The two programs each expect a small settings file (a "config") telling
them things like *which model to load, which port to talk on, which camera is which.*
Those files weren't in the project. The folders where they belong were empty.

**What was really wrong.** Whoever set this up originally never saved those files into the
project. The knowledge of "what settings make this run" was lost.

**Why it mattered.** Without them, neither program could even start. And there's a deeper
issue: research requires a fixed, written-down "normal" setup (the **baseline**) so that
later, when we deliberately change one thing, we can prove the change is what caused any
difference. No baseline settings = no controlled experiments.

**What we did.** We read the code to figure out exactly which settings it looks for, then
wrote two clearly-commented config files:
- `configs/server/pi0_libero_finetuned_v044.yaml` (server settings)
- `configs/runner/libero_pi0.yaml` (client settings; this also defines the "untouched"
  baseline, with all the perturbation knobs set to "do nothing")

**Status.** ✅ Fixed (commit `92adcdb`). Note: these settings were reverse-engineered from
the code, so they're a careful best guess — we'll confirm them once we can measure how well
the model actually performs.

---

### Problem 2 — The model wouldn't even load (a crash from unrelated code)

**What we saw.** Just trying to load π0 in Python crashed immediately with a confusing
error: `non-default argument 'backbone_cfg' follows default argument`. The error pointed
at a *completely different* model called "GR00T" that we don't even use.

**What was really wrong.** The library is organized so that loading *any one* model
accidentally triggers loading code for *all* models. One of those other models (GR00T) had
a small bug that recent versions of an underlying library (`transformers`) turned into a
hard crash. Because everything loads together, that unrelated bug took π0 down with it.

> *A tiny bit of detail for the curious:* the GR00T config class told Python to
> auto-generate a setup function, but it had already written its own. The two collided.
> The fix tells Python "don't auto-generate one — use the one we wrote."

**Why it mattered.** This was a total roadblock: the server couldn't start, so *nothing*
could run.

**What we did.** A one-line fix to that GR00T class so it stops auto-generating the
conflicting code.

**Status.** ✅ Fixed (commit `1256487`).

---

### Problem 3 — ⭐ The model loaded, but half of it was secretly *untrained*

> **This is the most important discovery of Stage 1.** It's dangerous precisely because
> nothing crashed — the model *looked* fine while being quietly broken.

**What we saw.** When loading the trained model, a *warning* scrolled by —
`Could not load state dict… Missing keys… Unexpected keys…` — and then loading continued
as if nothing happened. Easy to miss.

**A quick concept you need here.** A trained neural network is basically a giant bag of
numbers ("weights") that were learned during training. Loading a model means copying those
saved numbers into the right slots. The slots have names. If the saved names don't match
the slot names, the numbers don't get copied — and those slots keep whatever **random
starting values** they had before training. Random values = that part of the model knows
nothing.

**What was really wrong.** The model file was saved by an *older* version of a library that
named the slots for the **vision system** (the part that actually *sees* the camera images)
one way. The *newer* library we use names them slightly differently. The loading code was
*supposed* to translate the old names to the new names, but for the vision part it only
printed a warning and never actually did the translation. So **every one of the 437 vision
slots failed to load**, and the model's "eyes" were left completely random.

**Why it mattered — a lot.** A π0 whose vision system is random can't actually see the
scene. It would still output movements, and still draw attention maps — but they'd be
meaningless. Worse, it fails *silently*: there's no crash, the robot still moves, the
numbers still flow. Every single experiment about "where the model looks" or "does it find
the right object" would have been **garbage, with no warning sign.** Catching this is the
single most valuable thing we did in Stage 1.

**What we did, and how we proved it.**
1. We pulled out the exact list of name mismatches: **437 names the model wanted but
   couldn't find, and 437 names in the file that went unused.**
2. We tested a simple rename rule (drop the extra `vision_model.` from the old names) and
   confirmed it makes the two lists match **perfectly — 437 down to 0.**
3. We put that rename into the loading code.
4. We reloaded the model and confirmed it now says **"All keys loaded successfully!" — 0
   missing, 0 unused, all 4.03 billion numbers in place.**

**Status.** ✅ Fixed and verified (commit `a485ebc`).

**Takeaway baked into our procedure:** from now on, *any* "missing keys" message when
loading the model is treated as a hard stop, not a warning to scroll past.

---

### Problem 4 — Building the simulator environment was a minefield

The simulator side (LIBERO + MuJoCo + supporting libraries) is famously fussy to install,
and there were no instructions. We hit several separate snags. All of them are now solved
inside a single repeatable setup script (`build_libero_env.sh`), but here's what each was:

**4a — A "clean" environment wasn't actually clean.**
*What we saw:* brand-new environments mysteriously already contained old versions of
libraries we hadn't installed.
*Why:* Python has a per-user "shared shelf" of packages (`~/.local/...`) that *every*
matching Python automatically reads from. It was leaking old libraries into our fresh
setup. *Fix:* flip a switch (`PYTHONNOUSERSITE=1`) that tells Python to ignore that shared
shelf.

**4b — A small piece needed a compiler tool (CMake) that wasn't installed.**
One dependency builds a little C program during installation and needs a build tool called
CMake. It wasn't present, so installation failed. *Fix:* install CMake.

**4c — …and then CMake "existed" but still didn't work.**
*What we saw:* even after installing CMake, the same step failed, claiming CMake was
broken — yet typing `cmake --version` ourselves worked fine.
*Why (this one's sneaky):* the easy way to install CMake (via `pip`) doesn't install the
real program — it installs a little stand-in that *looks up* the real program. During
installation, software is built in a sealed-off sandbox where that lookup fails. So CMake
worked for us but not inside the sandbox. *Fix:* install the **real** CMake program (via
conda) instead of the stand-in, and add one setting so the modern CMake tolerates the very
old build instructions this dependency ships with.

**What we did overall.** Wrapped all of this into one script that anyone can run to rebuild
the environment from scratch, plus a checker script (`verify_client.py`) that confirms the
important pieces import correctly.

**Status.** ✅ Fixed at the "does it install and import" level — the checker prints
`VERIFY_OK`, and the `libero_object` task set we need is available. ⚠️ **Still to confirm:**
that the simulator can actually *render camera images* (this needs a free GPU); that gets
its first real test when we run an actual episode.

---

### Problem 5 — A nearby environment looked reusable but was broken

**What we saw.** An existing environment on the server (`libero_plus`, from a different
project) seemed to already have LIBERO installed.
**What was really wrong.** It was a "live link" to a folder that had since been deleted, so
LIBERO couldn't actually be imported there.
**Why it mattered.** It was a tempting shortcut that would have wasted time and caused
baffling errors.
**What we did.** Documented it and deliberately built our own fresh environment instead.
**Status.** ✅ Documented / avoided.

---

### Problem 6 — Small repository "housekeeping" issues

A few low-level gotchas that quietly cause "works on my machine, not on yours" problems:

- **Leftover test files blocked syncing the project.** The bundled copy of a library
  included ~45 broken test fixtures that prevented uploading the project to GitHub. We
  removed and ignored them (they're irrelevant test data).
- **Empty folders vanish when copying via git.** Git doesn't track empty folders, so
  `configs/` and `docs/` silently disappeared after copying the project to the server —
  a recurring source of confusion. (Now that they have real files in them, this fixes
  itself.)
- **The required Python version was understated.** The project's notes say Python 3.11 is
  enough, but the code actually uses a feature that only works on **Python 3.12+**. The
  server environment must be 3.12 or newer.

**Status.** ✅ All documented and handled.

---

## How to reproduce the working setup (step by step)

If someone needs to rebuild everything from zero, this is the proven recipe:

1. **Create the model-server environment** (Python 3.12): install the bundled `lerobot`
   library with its π0 extras, plus `fastapi`, `uvicorn`, `matplotlib`, `pyyaml`.
2. **Make sure the two code fixes are present** (they're already committed): `1256487`
   (Problem 2) and `a485ebc` (Problem 3). ⚠️ Without `a485ebc`, the model loads *silently
   wrong* — see Problem 3.
3. **Download the model** into the project folder so its location matches the config:
   `python vlas/checkpoints/download_model.py lerobot/pi0_libero_finetuned_v044 --target_dir .`
4. **Check the load before trusting anything.** You must see *"All keys loaded
   successfully!", 0 missing / 0 unused, 4.03 B parameters.* If you see *any* "missing
   keys" warning, stop — the model is broken (Problem 3).
5. **Create the simulator environment** by running `build_libero_env.sh`, then run
   `verify_client.py` and confirm it prints `VERIFY_OK`.
6. **Write down, for every run:** which config files, the random seed, and which starting
   layout was used — so the result can be reproduced exactly.

---

## What still needs doing

- [ ] **Run one full baseline episode end-to-end.** Currently waiting on a free GPU (both
      were in use by another project).
- [ ] **Confirm the simulator can render images** (needs a GPU; only the install was
      checked so far — Problem 4).
- [ ] **Measure the baseline success rate** on `libero_object`, to confirm the
      reverse-engineered settings (Problem 1) actually reproduce expected performance.
- [ ] **Try the attention-analysis tools** on real recorded episodes.
- [ ] **Revisit a harmless-looking warning** from the simulator-environment install (a
      package-version complaint); everything imports fine, but worth a second look if
      anything misbehaves later.

---

## Why this work counts as research

Two of these problems would each have ruined the project's results, in opposite ways:

- **Problem 2** was loud — it stopped everything, so we couldn't have gotten fake results.
- **Problem 3** was the dangerous kind — it let a *half-untrained* model run while looking
  perfectly healthy. If we hadn't caught it, we could have spent the whole summer carefully
  measuring the behavior of a model that literally couldn't see.

The rest (missing settings, the fragile install, the housekeeping issues) are the
difference between a pipeline only one person can run once, and one that anybody can rebuild
and trust. Pinning all of this down *is* the Stage 1 contribution: it's the verified
foundation that every later "what makes the robot fail?" experiment will stand on.

---

## Update (2026-07-02) — The Real Root Cause, and the Corrected Baseline

**Headline:** the model was only succeeding **10 times out of 100**. After a full
investigation we found the cause was **one wrong library version** (`transformers`), fixed
it, and the success rate jumped to **91/100 (91%)** — the expected lab result. This section
tells that story and **corrects Problem 3 above**, which turned out to be a band-aid on the
wrong wound.

## Problem 7 — ⭐⭐ Only 10% success, and it was the `transformers` version all along

**What we saw.** With everything "working," the robot succeeded on only **10 of 100**
episodes. The successes were all *tall bottles*; every *flat object* (cream cheese, butter,
milk, chocolate pudding, …) failed **0/10**. The arm reached the right area but never quite
grasped — the fingerprint of a model that can *sort of* see but not precisely.

**What we ruled out (important — these were NOT the problem).** We methodically checked and
cleared: the normalization statistics (real and correct), the image resizing (fine), the
config files (identical to the lab's), and the checkpoint (correct). A dedicated tool,
`scripts/diagnose_norm_stats.py`, proved the stats were healthy in seconds.

**What was actually wrong.** The AI library `transformers` had changed how it names the
pieces of the vision encoder between versions. The lab's model was built for an **older
`transformers` (5.3.x)**; our server was running a **newer one (5.12.1)**. On the newer
version the vision encoder's weights no longer matched their expected names, so **most of
the "eyes" of the model loaded as random noise** — while everything *looked* like it loaded
fine. The robot was reaching from memory, not sight. That's why tall bottles (easy, roughly
fixed positions) sometimes worked and precise grasps of flat objects never did.

**Why Problem 3 above was misleading.** Problem 3 described adding a "SigLIP `vision_tower`
key remap" as the *fix* for the untrained-vision issue. In fact that remap was a **band-aid
for the wrong version** — it papered over the newer library's renaming but did **not**
restore the vision encoder correctly, which is exactly why success stayed at 10%. On the
**correct** version (`transformers==5.3.0`) the weights load **natively, with zero missing
keys, and no remap at all**. So the real fix was to *remove* that patch and *pin the right
version*, not to add code.

**What we did.**
- Rebuilt the server environment on **`transformers==5.3.0`** (+ `torch==2.7.1+cu128`,
  Python 3.12). See `requirements-working.txt`.
- **Reverted** the three "compatibility patches" we'd added for the newer library (the
  SigLIP remap, plus two crash-workarounds in `pi_gemma.py` and `groot_n1.py`) — none are
  needed on 5.3.0.
- Re-ran the full baseline: **91/100**. Every flat-object task that was 0/10 recovered
  (alphabet soup 10/10, cream cheese 9/10, butter 10/10, chocolate pudding 8/10, …).

**The trap to avoid.** The repo's own `vlas/lerobot/requirements-ubuntu.txt` pins
`transformers==4.57.1`, but that is **stale and wrong** — the model code needs 5.x and
*crashes* on 4.57.1 (a missing `pooler_output` attribute). And `vlas/lerobot/pyproject.toml`
allows `>=5.3.0,<6.0.0`, so a normal install grabs the *latest* 5.x (the broken one). The
one value that works is **exactly `5.3.0`** — pinned now in `requirements-working.txt`.

## Problem 8 — Turning on attention maps crashes the server

**What we saw.** Setting `include_attentions: true` makes every prediction return **HTTP 500**.

**What was wrong.** In `src/models/pi0/pi0_model.py`, the attention branch unpacks **5**
return values, but the model's `predict_action_chunk` returns **6**
(`…, nll_res, obs_embs`) → `ValueError: too many values to unpack`. The attentions-off
branch reads the results by index and is unaffected, which is why normal runs are fine.

**What we did.** One-line fix (unpack 6 values). Attention maps aren't needed for measuring
success, but they *are* needed for the object-grounding analysis later, so this unblocks that.

## Corrected baseline result (the number to trust)

**LIBERO-Object, all 10 tasks × 10 trials = 100 episodes, `transformers==5.3.0`,
`beta=30`** (2026-07-02):

| Task | Success | (was, on the broken 5.12.1) |
|---|---|---|
| alphabet soup | 10/10 | 0/10 |
| cream cheese | 9/10 | 0/10 |
| salad dressing | 10/10 | 6/10 |
| bbq sauce | 9/10 | 0/10 |
| ketchup | 8/10 | 0/10 |
| tomato sauce | 8/10 | 0/10 |
| butter | 10/10 | 0/10 |
| milk | 9/10 | 0/10 |
| chocolate pudding | 8/10 | 0/10 |
| orange juice | 10/10 | 4/10 |
| **Overall** | **91/100 (91%)** | **10/100** |

_(Note: this baseline was run with the lab config's `beta=30` brightness offset still on;
a clean `beta=0` run should be at least as high. Even under the perturbation the fixed
model reaches 91%.)_

## Updated "what needs doing"

- [x] **Run a full baseline** — done: **91/100** on `transformers==5.3.0`.
- [x] **Confirm the simulator renders + measure baseline success** — done.
- [ ] Optionally re-run the clean `beta=0` baseline for the canonical number.
- [ ] Proceed to Stage 2 (controlled perturbation experiments) on this verified foundation.
