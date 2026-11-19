from __future__ import annotations

"""Run a single-factor perturbation sweep.

Loads a base runner YAML, overrides one observation parameter across a list of
values (holding everything else fixed), runs the LIBERO evaluator for each
condition, and writes a per-condition success-rate summary. This is the driver
behind the contrast (observation.alpha) and proprioception
(observation.proprio_noise_std) sweeps; the same pattern reproduces the
brightness (observation.beta) sweep.

Example:
    python scripts/run_sweep.py \
        --base configs/runner/libero_pi0.yaml \
        --param observation.alpha \
        --values 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0 \
        --fixed observation.beta=0 \
        --num-trials 2 --seed 1 \
        --out-root outputs/contrast_sweep_smoke
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import yaml

from src.runner.configs import RunnerConfig
from src.runner.libero.evaluator import LiberoEvaluator


def _coerce(value: str):
    """Parse a CLI string into int/float/bool where possible, else keep as str."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        # Autovivify so a sweep can introduce an optional block (e.g. `scene`)
        # that the base YAML omits.
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/runner/libero_pi0.yaml")
    parser.add_argument("--param", required=True,
                        help="Dotted config path to sweep, e.g. observation.alpha")
    parser.add_argument("--values", required=True,
                        help="Comma-separated values for --param, e.g. 0.2,0.4,1.0")
    parser.add_argument("--fixed", default="",
                        help="Comma-separated dotted overrides held fixed for every "
                             "condition, e.g. observation.beta=0,observation.alpha=1.0")
    parser.add_argument("--num-trials", type=int, default=None,
                        help="Override env.num_trials_per_task (episodes per task).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override experiment.seed (kept identical across conditions).")
    parser.add_argument("--out-root", required=True,
                        help="Root output dir; each condition writes a subdir under it.")
    args = parser.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    fixed_overrides = []
    if args.fixed.strip():
        for pair in args.fixed.split(","):
            key, val = pair.split("=", 1)
            fixed_overrides.append((key.strip(), _coerce(val.strip())))

    values = [_coerce(v.strip()) for v in args.values.split(",")]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    param_tag = args.param.split(".")[-1]
    summary_rows = []

    for value in values:
        cfg = copy.deepcopy(base_cfg)
        for key, val in fixed_overrides:
            _set_dotted(cfg, key, val)
        _set_dotted(cfg, args.param, value)
        if args.num_trials is not None:
            cfg["env"]["num_trials_per_task"] = args.num_trials
        if args.seed is not None:
            cfg["experiment"]["seed"] = args.seed

        cond_name = f"{param_tag}_{value}"
        cond_dir = out_root / cond_name
        cfg["experiment"]["output_dir"] = str(cond_dir)
        cfg["experiment"]["name"] = cond_name

        cfg_path = cfg_dir / f"{cond_name}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"\n=== Sweep condition: {args.param} = {value} "
              f"({cfg['env']['num_trials_per_task']} trials/task) ===", flush=True)

        runner_cfg = RunnerConfig.from_yaml(cfg_path)
        evaluator = LiberoEvaluator(runner_cfg)
        results = evaluator.evaluate()

        n = len(results)
        n_success = sum(int(r["success"]) for r in results)
        rate = (n_success / n) if n else 0.0
        row = {"param": args.param, "value": value,
               "success": n_success, "total": n, "success_rate": round(rate, 4)}
        summary_rows.append(row)
        cond_dir.mkdir(parents=True, exist_ok=True)
        (cond_dir / "success.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

        # Per-episode records too: scene perturbations need the per-episode
        # geometry (which slot the target moved to, which distractor it swapped
        # with) to interpret failures, and success.json alone throws that away.
        per_episode = [
            {
                **{k: v for k, v in r.items() if k != "actions"},
                "num_actions": int(len(r["actions"])),
            }
            for r in results
        ]
        (cond_dir / "results.json").write_text(
            json.dumps(per_episode, indent=2), encoding="utf-8")
        print(f"--- {cond_name}: {n_success}/{n} = {rate:.4f}", flush=True)

    summary_csv = out_root / "SWEEP_SUMMARY.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "value", "success", "total", "success_rate"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSweep complete. Summary -> {summary_csv.as_posix()}")
    for row in summary_rows:
        print(f"  {row['value']}: {row['success']}/{row['total']} = {row['success_rate']}")


if __name__ == "__main__":
    main()
