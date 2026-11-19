from __future__ import annotations

"""Physics sanity-check for Stage 3 scene perturbations. No policy server needed.

Moving an object by rewriting qpos is only meaningful if the object then rests
stably on the table: a placement that floats, inter-penetrates another object, or
slides off would silently confound every success rate we later measure. This
script applies each perturbation condition, lets the simulator settle under
dummy actions, and reports how far objects drifted, whether any fell, and how
close the closest pair ended up. It optionally renders the settled scene so the
placements can be eyeballed.

Example:
    python scripts/validate_scene_perturbation.py \
        --config configs/runner/libero_pi0_scene.yaml \
        --modes move_target,move_distractors,swap \
        --distances 0.05,0.10,0.15 \
        --episodes 3 --render-dir outputs/scene_validation/renders
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.runner.configs import RunnerConfig, ScenePerturbationConfig
from src.runner.libero.env_factory import LiberoEnvFactory
from src.runner.libero.scene_perturbation import ScenePerturber
from src.runner.libero.task_suite import LiberoTaskSuiteLoader

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
# An object that ends up this far below its commanded height has fallen off the
# table or through a surface rather than settled on one.
FALL_TOLERANCE = 0.05
# Settling should not translate an object appreciably; more than this means it
# was spawned inside something and got shoved out.
DRIFT_TOLERANCE = 0.03


def settle(env, steps: int):
    for _ in range(steps):
        env.step(DUMMY_ACTION)


def object_xyz(state_getter, slot):
    qpos = state_getter()
    return np.array(qpos[slot.qpos_addr: slot.qpos_addr + 3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/runner/libero_pi0.yaml')
    ap.add_argument('--modes', default='move_target,move_distractors,swap')
    ap.add_argument('--distances', default='0.05,0.10,0.15',
                    help='Distances (m) applied to the move_* modes.')
    ap.add_argument('--episodes', type=int, default=3,
                    help='Episodes (init states) checked per task per condition.')
    ap.add_argument('--tasks', default='',
                    help='Comma-separated task ids; empty = all tasks in the suite.')
    ap.add_argument('--settle-steps', type=int, default=20)
    ap.add_argument('--render-dir', default='',
                    help='If set, save one agentview PNG per task per condition.')
    ap.add_argument('--out', default='outputs/scene_validation/report.json')
    args = ap.parse_args()

    cfg = RunnerConfig.from_yaml(args.config)
    suite = LiberoTaskSuiteLoader().get_suite(cfg.env.task_suite)
    factory = LiberoEnvFactory()

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    distances = [float(d) for d in args.distances.split(',') if d.strip()]
    task_ids = ([int(t) for t in args.tasks.split(',') if t.strip()]
                if args.tasks.strip() else list(range(suite.n_tasks)))

    conditions = []
    for mode in modes:
        if mode == 'swap':
            conditions.append((mode, 0.0))
        else:
            conditions.extend((mode, d) for d in distances)

    render_dir = Path(args.render_dir) if args.render_dir else None
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)

    findings = []
    problems = 0

    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = np.asarray(suite.get_task_init_states(task_id))
        env, task_description = factory.create(
            task, resolution=cfg.env.resolution, seed=cfg.experiment.seed)
        try:
            for mode, distance in conditions:
                scene_cfg = ScenePerturbationConfig(
                    **{**cfg.scene.__dict__, 'mode': mode, 'distance': distance})
                perturber = ScenePerturber(scene_cfg)
                perturber.resolve_layout(env, task_id, task_description)
                layout = perturber._layouts[task_id]

                for ep in range(min(args.episodes, len(init_states))):
                    env.reset()
                    state, info = perturber.apply(init_states[ep], task_id, ep)
                    env.set_init_state(state)

                    commanded = {s.name: np.array(state[s.xy_slice]) for s in layout.movable}
                    commanded_z = {s.name: float(state[s.z_index]) for s in layout.movable}

                    settle(env, args.settle_steps)
                    qpos = env.env.sim.data.qpos

                    settled, drift, fell = {}, {}, []
                    for s in layout.movable:
                        xyz = np.array(qpos[s.qpos_addr: s.qpos_addr + 3])
                        settled[s.name] = xyz
                        drift[s.name] = float(np.linalg.norm(xyz[:2] - commanded[s.name]))
                        if xyz[2] < commanded_z[s.name] - FALL_TOLERANCE:
                            fell.append(s.name)

                    names = list(settled)
                    pair_min, pair_who = np.inf, None
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            d = float(np.linalg.norm(settled[names[i]][:2] - settled[names[j]][:2]))
                            if d < pair_min:
                                pair_min, pair_who = d, (names[i], names[j])

                    max_drift = max(drift.values()) if drift else 0.0
                    bad = bool(fell) or max_drift > DRIFT_TOLERANCE or info.get('fallbacks', 0)
                    problems += int(bad)

                    findings.append({
                        'task_id': task_id,
                        'task': task_description,
                        'mode': mode,
                        'distance': distance,
                        'episode': ep,
                        'max_drift_m': round(max_drift, 4),
                        'worst_drift_object': max(drift, key=drift.get) if drift else None,
                        'fell': fell,
                        'min_pair_distance_m': round(float(pair_min), 4),
                        'closest_pair': list(pair_who) if pair_who else None,
                        'fallbacks': info.get('fallbacks', 0),
                        'ok': not bad,
                    })

                    if bad:
                        print(f'  !! task {task_id} {mode} d={distance} ep={ep}: '
                              f'drift={max_drift:.3f} fell={fell} '
                              f'fallbacks={info.get("fallbacks", 0)}', flush=True)

                    if render_dir and ep == 0:
                        obs = env.env._get_observations()
                        img = obs.get('agentview_image')
                        if img is not None:
                            _save_png(np.flipud(img),
                                      render_dir / f'task{task_id:02d}_{mode}_d{distance:g}.png')

                print(f'[task {task_id}] {mode} d={distance:g}: checked '
                      f'{min(args.episodes, len(init_states))} episodes', flush=True)
        finally:
            env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=2), encoding='utf-8')

    n = len(findings)
    print(f'\n{n - problems}/{n} condition-episodes clean; {problems} flagged.')
    print(f'Report -> {out.as_posix()}')
    if render_dir:
        print(f'Renders -> {render_dir.as_posix()}')


def _save_png(array: np.ndarray, path: Path) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        from PIL import Image
        Image.fromarray(array.astype(np.uint8)).save(path)
        return
    imageio.imwrite(path, array.astype(np.uint8))


if __name__ == '__main__':
    main()
