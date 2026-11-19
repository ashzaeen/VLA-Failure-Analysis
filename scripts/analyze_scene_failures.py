from __future__ import annotations

"""Classify Stage 3 scene-perturbation episodes by what the robot actually did.

A success rate says the policy failed; it does not say *how*. For a swap episode
the decisive question is which object the robot disturbed:

  decoy      the distractor now occupying the target's usual slot was moved.
             This is the position-shortcut signature: the arm went to the
             memorised place and grasped whatever happened to be there.
  target     the instructed object was moved (the policy found it, but did not
             finish) - evidence *against* a pure position shortcut.
  other      some third object was moved.
  none       nothing was disturbed; the arm never made meaningful contact.

Reads the per-episode results.json written by run_sweep.py / run_libero_eval.py.

Example:
    python scripts/analyze_scene_failures.py \
        outputs/scene_swap_full/mode_swap/results.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def classify(entry: Dict[str, Any]) -> str:
    scene = entry.get('scene') or {}
    disturbed: List[Dict[str, Any]] = scene.get('disturbed') or []
    # The basket is the goal receptacle; it moving is not a manipulation event.
    disturbed = [d for d in disturbed if d.get('role') != 'basket']
    if not disturbed:
        return 'none'

    primary = disturbed[0]           # already sorted by displacement, descending
    partner = scene.get('partner')   # swap mode: the object now at the home slot

    if partner is not None and primary['object'] == partner:
        return 'decoy'
    if primary.get('role') == 'target':
        return 'target'
    return 'other'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+', help='One or more results.json files.')
    ap.add_argument('--failures-only', action='store_true',
                    help='Classify only the failed episodes.')
    args = ap.parse_args()

    for path in args.results:
        p = Path(path)
        if not p.exists():
            print(f'!! missing: {p}')
            continue
        entries = json.loads(p.read_text(encoding='utf-8'))
        if args.failures_only:
            entries = [e for e in entries if not e.get('success')]

        counts = Counter(classify(e) for e in entries)
        by_task: Dict[int, Counter] = defaultdict(Counter)
        for e in entries:
            by_task[e.get('task_id', -1)][classify(e)] += 1

        n = len(entries)
        n_success = sum(int(bool(e.get('success'))) for e in entries)
        print(f'\n=== {p} ===')
        print(f'episodes={n}  success={n_success} ({n_success / n:.0%})'
              if n else f'episodes=0')
        if not n:
            continue

        print('\n  what the robot disturbed:')
        for label in ('decoy', 'target', 'other', 'none'):
            c = counts.get(label, 0)
            if c:
                print(f'    {label:<8} {c:>4}  ({c / n:.0%})')

        print('\n  per task:')
        for task_id in sorted(by_task):
            row = by_task[task_id]
            desc = next((e.get('task_description', '') for e in entries
                         if e.get('task_id') == task_id), '')
            parts = ' '.join(f'{k}={v}' for k, v in row.most_common())
            print(f'    [{task_id}] {desc[:46]:<46} {parts}')


if __name__ == '__main__':
    main()
