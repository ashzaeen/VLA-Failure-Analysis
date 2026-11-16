from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.runner.client.policy_client import RemotePolicyClient
from src.runner.configs import RunnerConfig
from src.runner.libero.action_postprocess import LiberoActionPostprocessor
from src.runner.libero.env_factory import LiberoEnvFactory
from src.runner.libero.observation_builder import LiberoObservationBuilder
from src.runner.libero.task_suite import LiberoTaskSuiteLoader
from src.utils.logger import get_logger
from src.utils.seed import set_seed_everywhere
from src.utils.video_utils import save_rollout_video

logger = get_logger(__name__)


class LiberoEvaluator:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self.env_factory = LiberoEnvFactory()
        self.task_suite_loader = LiberoTaskSuiteLoader()
        self.action_postprocessor = LiberoActionPostprocessor(**cfg.robot.__dict__)
        self.observation_builder = LiberoObservationBuilder(cfg.observation, self.action_postprocessor)
        self.policy_client = RemotePolicyClient(cfg.policy_client)

    def get_dummy_action(self) -> List[float]:
        return [0, 0, 0, 0, 0, 0, -1]
    
    def _save_episode_npz(self, episode_data: Dict[str, Any]) -> str:
        out_dir = Path(self.cfg.experiment.output_dir) / "episodes_npz"
        out_dir.mkdir(parents=True, exist_ok=True)

        task_id = episode_data["task_id"]
        episode_idx = episode_data["episode_idx"]

        file_path = out_dir / f"task_{task_id:03d}_episode_{episode_idx:03d}.npz"

        save_dict = {}

        for key, value in episode_data.items():
            if key in ["task_id", "episode_idx"]:
                save_dict[key] = np.array(value, dtype=np.int32)
            elif key == "success":
                save_dict[key] = np.array(value, dtype=np.bool_)
            elif key == "task_description":
                save_dict[key] = np.array(value, dtype=object)
            elif key in ["attention_maps", "velocitys", "nll_steps", "nll_totals"]:
                save_dict[key] = np.asarray(value, dtype=object)
            else:
                save_dict[key] = np.asarray(value)

        np.savez_compressed(file_path, **save_dict)

        return str(file_path)


    def run_episode(self, env, task_description: str, initial_state, task_id: int, episode_idx: int) -> Dict[str, Any]:
        env.reset()
        self.policy_client.reset()
        obs = env.set_init_state(initial_state)
        success = False

        actions: List[np.ndarray] = []
        raw_actions: List[np.ndarray] = []
        attention_maps: List[np.ndarray] = []

        front_images: List[np.ndarray] = []
        wrist_images: List[np.ndarray] = []

        velocitys = []
        nll_steps = []
        nll_totals = []
        hidden_states = []
        for t in range(self.cfg.env.max_steps + self.cfg.env.num_steps_wait):
            if t < self.cfg.env.num_steps_wait:
                obs, _, _, _ = env.step(self.get_dummy_action())
                continue

            canonical_obs = self.observation_builder.build(
                obs,
                task_description,
                metadata={'task_id': task_id, 'episode_idx': episode_idx, 'step_idx': t},
            )

            response = self.policy_client.predict(canonical_obs)

            raw_action = response.raw_action
            attention = response.action_attentions
            v_ts = response.v_ts
            # print(f"Step {t}: action={response.action}, success={success}")
            action = np.array(response.action)
            # response.action is already a single (action_dim,) vector; only squeeze a
            # leading batch dim if one is present. (Was `if action.shape != 1` — a
            # tuple-vs-int compare that is always True and collapsed (7,) to a scalar.)
            if action.ndim > 1:
                action = action[0]
            action = self.action_postprocessor.postprocess_action(action)

            raw_actions.append(np.asarray(raw_action))
            actions.append(np.asarray(action))
            velocitys.append(np.asarray(v_ts))
            attention_maps.append(np.asarray(attention))
            # logger.info(f"Step {t}: action={action}, success={success}")

            nll_steps.append(response.nll_steps)
            nll_totals.append(response.nll_res)
            hidden_states.append(response.obs_embs)

            # 保存两个视角图像
            front_key =list(canonical_obs.images_np.keys())[0]
            wrist_key = list(canonical_obs.images_np.keys())[1]
            # logger.info(f"Step {t}: front image shape={canonical_obs.images_np}")
            if self.cfg.logging.save_episode_npz:
                front_images.append(canonical_obs.images_np[front_key])
                wrist_images.append(canonical_obs.images_np[wrist_key])
            logger.info(f"Step {t}: action={action}, success={success}")

            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break

        video_path = None
        logger.info(f"Episode {episode_idx} finished with success={success}. Total steps: {t+1}")
        if front_images and self.cfg.logging.save_rollout_video:
            out_dir = Path(self.cfg.experiment.output_dir) / 'rollouts'
            out_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = save_rollout_video(
                front_images,
                episode_idx,
                success,
                task_description=task_description,
                rollout_dir=out_dir,
            )

        episode_npz_path = None
        if self.cfg.logging.save_episode_npz:
            episode_data = {
                "task_description": task_description,
                "initial_state": initial_state,
                "task_id": task_id,
                "episode_idx": episode_idx,
                "success": success,
                "actions": actions,
                "raw_actions": raw_actions,
                "attention_maps": attention_maps,
                "front_images": front_images,
                "wrist_images": wrist_images,
                "velocitys": velocitys,
                "nll_steps": nll_steps,
                "nll_totals": nll_totals,
                "hidden_states":hidden_states
            }

            episode_npz_path = self._save_episode_npz(episode_data)

        return {
            'success': success,
            'task_description': task_description,
            'episode_idx': episode_idx,
            'task_id': task_id,
            'video_path': video_path,
            'episode_npz_path': episode_npz_path,
            'actions': np.asarray(actions),
        }

    def evaluate(self) -> List[Dict[str, Any]]:
        set_seed_everywhere(self.cfg.experiment.seed)
        suite = self.task_suite_loader.get_suite(self.cfg.env.task_suite)
        results: List[Dict[str, Any]] = []
        total = 0            # was 400 — inflated the success-rate denominator
        success_total = 0

        for task_id in range(suite.n_tasks):   # was range(8, ...) — only ran tasks 8-9
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            env, task_description = self.env_factory.create(
                task,
                resolution=self.cfg.env.resolution,
                seed=self.cfg.experiment.seed,
            )

            logger.info('Evaluating task %s/%s: %s', task_id + 1, suite.n_tasks, task_description)

            task_success = 0
            task_trials = 0
            try:
                for episode_idx in range(self.cfg.env.num_trials_per_task):
                    # pass the real per-task episode_idx (was `total`, which mis-numbered
                    # the saved NPZ/video filenames as episode_400, 401, ...)
                    result = self.run_episode(env, task_description, initial_states[episode_idx], task_id, episode_idx)
                    results.append(result)
                    total += 1
                    task_trials += 1
                    success_total += int(result['success'])
                    task_success += int(result['success'])
            finally:
                env.close()

            logger.info(
                'Task %s/%s success: %s/%s — %s',
                task_id + 1, suite.n_tasks, task_success, task_trials, task_description,
            )

        logger.info(
            'Overall Success Rate: %s/%s = %.4f',
            success_total,
            total,
            (success_total / total) if total else 0.0
        )
        return results