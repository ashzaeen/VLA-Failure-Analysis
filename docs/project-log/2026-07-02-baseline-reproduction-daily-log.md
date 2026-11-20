# Progress Log — π0 / LIBERO Baseline Reproduction

_Last updated: 2026-07-02_

## TL;DR (current status) — SOLVED ✅

The 10/100 gap was the **`transformers` library version**. Reproduced the lab baseline:

**Full baseline: 91/100 (91%)** — pristine lab code + `transformers==5.3.0` + `torch==2.7.1+cu128`
(env `vla-pi0-lab` on bender), all 10 LIBERO-Object tasks × 10 trials, **with `beta=30` on** (2026-07-02):

| Task | 5.3.0 | broken (5.12.1) | | Task | 5.3.0 | broken |
|---|---|---|---|---|---|---|
| alphabet soup | 10/10 | 0 | | tomato sauce | 8/10 | 0 |
| cream cheese | 9/10 | 0 | | butter | 10/10 | 0 |
| salad dressing | 10/10 | 6 | | milk | 9/10 | 0 |
| bbq sauce | 9/10 | 0 | | chocolate pudding | 8/10 | 0 |
| ketchup | 8/10 | 0 | | orange juice | 10/10 | 4 |

**10/100 → 91/100.** Every flat-object task that was 0/10 recovered. Successful episodes finish in
~130–170 steps (clean grasp-and-place); the 9 failures ran the full 510 (genuine misses, no crashes).

- On 5.3.0 the checkpoint loads with **0 missing keys** (vision_tower.vision_model.* matches natively, no
  remap). On 5.12.1 those keys flattened → our remap patch mis-loaded the vision tower → degraded vision →
  grasp failures = the 10/100.
- Earlier ruled out: normalization stats, image resize, configs, checkpoint. The `requirements-ubuntu.txt`
  pin `4.57.1` was a red herring (crashes on `.pooler_output`); the `pyproject` floor `>=5.3.0` was correct.

**Corrected repo (merged to `main`):** reverted the 3 transformers-5.12 fork patches (SigLIP remap /
pi_gemma / groot), pinned `transformers==5.3.0` (`requirements-working.txt`), kept the genuine fixes +
diagnostics, and documented all issues in `docs/project-log/2026-06-25-stage-1-reproduction-log.md`.

**Attention baseline (2026-07-03):** full attention-enabled run scored **94/100**; attention recorded +
analyzed for all 100 episodes (heatmap videos + per-step metrics). Findings: attention concentration is
object-dependent (tall bottles sharp, flat objects diffuse) and successes have ~30% more peaked attention
than failures. Full write-up: `docs/project-log/2026-07-03-baseline-and-attention-findings.md`; reading guide:
`docs/guides/reading-attention-videos.md`. Artifacts: `outputs/baseline_attn_full/`.

---

## Goal

Reproduce the lab's working π0 + LIBERO baseline (expected ~90%), so we have a trustworthy starting point
for the failure-analysis / object-grounding experiments. Our runs sit at **10/100** — the gap is what we're
chasing.

Baseline detail (from earlier work): successes were concentrated on **tall bottles** (salad dressing 6/10,
orange juice 4/10); all **flat objects** (cream cheese, butter, milk, chocolate pudding, …) were **0/10**.
Pattern looked like a grasp-precision / action-fidelity gap.

---

## What we RULED OUT this session

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Image resize / preprocessing | ❌ not the cause | Model resizes internally (`resize_with_pad`→224); confirmed earlier |
| Normalization stats broken/placeholder | ❌ not the cause | `scripts/diagnose_norm_stats.py` on bender shows **real, non-trivial** state/action stats; local checkpoint == published repo, byte-identical (`observation.state` mean `[-0.0465, 0.0344, 0.7646, 2.9722, -0.2205, …]`) |
| Configs wrong | ❌ not the cause | lab vs our repo `configs/` are **byte-identical** |
| Checkpoint wrong | ❌ not the cause | Same checkpoint (`pepijn223/pi0_libero_finetuned_extra2`); loads fine |

---

## The real lead: `transformers` version

We were given a **known-good reference repo** at
`C:\Users\ashza\Downloads\vla-failure-analysis-lab\vla-failure-analysis-main` (pristine, "worked before").
Diffing it against our working repo:

- Only **5 code files** differ, and **all 5 are our transformers-5 compatibility patches**:
  1. `vlas/lerobot/.../pi0/modeling_pi0.py` — SigLIP `vision_tower.vision_model.` → `vision_tower.` key remap
  2. `vlas/lerobot/.../pi_gemma.py` — dropped the `cache_position` kwarg (tf5 masking refactor)
  3. `vlas/lerobot/.../groot/groot_n1.py` — `@dataclass(init=False)` (tf5 dataclass ordering crash)
  4. `src/models/pi0/pi0_model.py` — gated-tokenizer redirect + 6-value unpack fix
  5. `src/runner/libero/evaluator.py` — our bookkeeping fixes (task range, denominator, per-task tally)
- **`configs/` are byte-identical.**

So our repo is essentially **the lab repo + transformers-5 scaffolding**. That reframed the whole problem:
the difference between "lab worked" and "our 10/100" is either those patches or the exact library version.

### The version contradiction in the repo

The repo pins **disagree with each other**:

- `vlas/lerobot/requirements-ubuntu.txt` (a stale lock): `transformers==4.57.1`, `torch==2.7.1`,
  `tokenizers==0.22.1`, `torchvision==0.22.1`, `numpy==2.2.6`, `accelerate==1.11.0`,
  `safetensors==0.6.2`, `huggingface-hub==0.35.3`
- `vlas/lerobot/pyproject.toml`: `transformers>=5.3.0,<6.0.0`, `torch>=2.2.1,<2.11.0`, `requires-python>=3.11`

`4.57.1` does **not** satisfy `>=5.3.0`. They can't both be right.

### Hard evidence (what we actually observed)

| Environment | Result |
|---|---|
| pristine code + `transformers==4.57.1` (the lock) | ❌ **crashes** at `embed_image`: `AttributeError: 'Tensor' object has no attribute 'pooler_output'` — this vision API only exists in transformers **5.x** |
| pristine code + `transformers==5.12.1` | ❌ would crash at the **groot import** and **pi_gemma** masking — the exact things our patches fix (tf 5.12.1 is too *new*) |
| our patched code + `transformers==5.12.1` | ✅ runs, but **10/100** |

**Conclusion:** the pristine code needs transformers **5.x** (for `.pooler_output`) but breaks on **5.12.1**
(groot/pi_gemma). So the lab ran on a **specific older 5.x** — consistent with the `pyproject` floor
`>=5.3.0`. Both versions we've tried are outside that window (4.57.1 too old, 5.12.1 too new).

The checkpoint's `config.json` / `train_config.json` do **not** record the transformers version
(`train_config.json` `env` is `null`), so the repo can't tell us the exact minor. The usable signal is the
`pyproject` floor → **transformers 5.3.x**.

---

## The `include_attentions` bug (why the smoke runs with attentions OFF)

With `include_attentions: true`, the server 500s on the first `/predict`. Cause:

- Pristine `pi0_model.py` (attentions branch) unpacks **5** values:
  `actions, _, action_attentions, v_ts, nll_res = self.vla.predict_action_chunk(...)`
- The fork's `predict_action_chunk` (`modeling_pi0.py:1781`) **returns 6**:
  `return actions, language_attentions, action_attentions, v_ts, nll_res, obs_embs`
- 5 ← 6 → `ValueError: too many values to unpack (expected 5)` → HTTP 500.

The `include_attentions: false` branch reads the tuple by index and tolerates any length, so it runs fine.

- This is a **latent bug in the lab code itself** (its working runs used attentions off).
- Attentions are a **diagnostic add-on**, not needed to measure success rate.
- **Fix is one line** (unpack 6 instead of 5) — deferred to keep this run verbatim.
- **DECISION (user):** keep verbatim for now, run with attentions **off**; add the 1-line fix later once the
  environment is nailed down.

---

## Decisions made

- **Keep the lab config verbatim.** Do not swap lab config values for a "cleaner" baseline. In particular
  **`beta` stays at 30** (brightness perturbation), per user preference — twice confirmed.
- **`include_attentions: false`** for now (forced by the unpack bug; re-enable later with the 1-line fix).
- Smoke deviations that are plumbing, not science: `num_trials_per_task 50→3`, `timeout_seconds 300→600`.
- **Reproduce the environment, not just the code** — the code alone doesn't run without matching the
  library versions.

### What has / hasn't been changed vs the lab build

- **Code: 0 changes** in the deployed pristine repo (`~/projects/vla-failure-analysis-lab`). Verified
  pristine markers: no SigLIP remap, 5-value unpack, `range(8, n_tasks)`.
- **Config: server verbatim; runner has 3 flagged knobs** (trials, timeout, attentions); `beta=30` kept.
- **Environment: NOT verbatim** — this is the one real deviation and the open question.

---

## Current bender setup (state we can resume from)

- **Server GPU box:** RVL "bender", 2× A40. SSH via `ssh.rvlserver.com` (Windows OpenSSH + cloudflared;
  use PowerShell, not Git Bash).
- **Pristine lab repo:** `~/projects/vla-failure-analysis-lab` (unmodified code; checkpoint symlinked to
  `~/projects/vla-failure-analysis/pi0_libero_finetuned_v044`).
- **Our patched repo:** `~/projects/vla-failure-analysis` (the transformers-5 patched version).
- **Envs (conda, `~/miniconda3`):**
  - `vla-pi0` — transformers **5.12.1**, torch 2.10 (our patched-code env)
  - `vla-pi0-lab` — transformers **4.57.1**, torch 2.7.1+cu128 (**wrong version, crashes** — needs rebuild to 5.3.x)
  - `vla-libero` — LIBERO sim client env (unchanged, works)
- **HF token:** stored at `~/.cache/huggingface/token`, **verified working** with PaliGemma gated access
  (`whoami: Ashzaeen`). So the gated tokenizer downloads normally now — no local redirect needed.
- **Disk fix:** root `/` is ~97% full (3 GB free); `$HOME` is on a 4.5 TB NFS mount. Builds must set
  `TMPDIR=$HOME/tmp` so wheel extraction doesn't overflow root.
- **Diagnostic:** `scripts/diagnose_norm_stats.py` (loads config+processors only, seconds, no GPU) — proved
  norm stats are fine.

---

## Next steps

1. **Rebuild `vla-pi0-lab` on `transformers==5.3.0`** (keep torch 2.7.1+cu128, Python 3.12, `TMPDIR` on NFS).
   - Check that PI0Policy loads with **no missing/unexpected vision_tower keys** (tests whether the SigLIP
     remap is actually needed on this version) and that the groot import + `pi_gemma` masking don't crash.
2. If it loads clean, run the **verbatim smoke** (beta=30, attentions off, 3 trials → tasks 8–9 only, per
   the pristine evaluator's `range(8, n_tasks)`), read the success rate, compare to 10/100.
3. If 5.3.0 is slightly off, bisect the small 5.3–5.11 range.
4. (Optional but recommended) Ask the lab source for their exact `transformers`/`torch`/Python versions or a
   `pip freeze` — the definitive answer if the bisect is painful.
5. Once the environment reproduces the lab result, re-apply the **1-line attention unpack fix** to run with
   `include_attentions: true` for the attention-analysis work.

---

## Key facts to remember

- The 10/100 gap is **not** norm stats, resize, config, or checkpoint. It is (almost certainly) the
  `transformers` version and/or its interaction with our compat patches.
- The lab code requires transformers **5.x** but **not** 5.12.1 → target **~5.3.x**.
- The repo's `requirements-ubuntu.txt` pin (`4.57.1`) is **stale/wrong** — don't trust it.
- Keep the lab config **verbatim** (esp. `beta=30`); only deviate for a real blocker and say so.
