from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runner.configs import RunnerConfig
from src.runner.libero.evaluator import LiberoEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/runner/libero_pi0.yaml')
    args = parser.parse_args()

    config = RunnerConfig.from_yaml(args.config)
    evaluator = LiberoEvaluator(config)
    results = evaluator.evaluate()
    out_dir = Path(config.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    for item in results:
        serializable.append({
            **{k: v for k, v in item.items() if k != 'actions'},
            'num_actions': int(item['actions'].shape[0]) if hasattr(item['actions'], 'shape') else len(item['actions']),
        })
    (out_dir / 'results.json').write_text(json.dumps(serializable, indent=2), encoding='utf-8')
    print(f'Saved results to {(out_dir / "results.json").as_posix()}')


if __name__ == '__main__':
    main()
