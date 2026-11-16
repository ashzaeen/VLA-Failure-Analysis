from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from common.schemas import PredictRequest
from src.runner.configs import RunnerConfig
from src.runner.libero.observation_builder import LiberoObservationBuilder
from src.runner.client.policy_client import RemotePolicyClient
from src.runner.libero.robot_arm import LiberoRobotArm


@dataclass
class EpisodeResult:
    success: bool
    task_description: str
    episode_idx: int
    executed_steps: int
    actions: List[List[float]]
    raw_actions: List[List[float]]


class LiberoRunner:
    def __init__(self, cfg: RunnerConfig, quat_to_axisangle_fn: callable):
        self.cfg = cfg
        self.client = RemotePolicyClient(cfg.server_url, timeout_s=cfg.request_timeout_s)
        self.robot_arm = LiberoRobotArm(
            action_scale=cfg.action_scale,
            normalize_gripper=cfg.normalize_gripper,
            invert_gripper=cfg.invert_gripper,
            binarize_gripper=cfg.binarize_gripper,
            quat_to_axisangle_fn=quat_to_axisangle_fn,
        )
        self.obs_builder = LiberoObservationBuilder(cfg.image_size, self.robot_arm)
        self._cached_chunk: list[list[float]] = []
        self._chunk_index = 0

    def reset_policy_state(self) -> None:
        self._cached_chunk = []
        self._chunk_index = 0
        self.client.reset_policy(self.cfg.policy_id)

    def get_dummy_action(self) -> list[float]:
        if self.cfg.warmup_dummy_action is not None:
            return self.cfg.warmup_dummy_action
        return [0, 0, 0, 0, 0, 0, -1]

    def _query_chunk(self, obs: Dict[str, np.ndarray], task_description: str, episode_idx: int, step_idx: int) -> Dict[str, Any]:
        canonical_obs = self.obs_builder.build(obs, task_description, episode_idx, step_idx)
        req = PredictRequest(
            policy_id=self.cfg.policy_id,
            observation=canonical_obs,
            return_attentions=False,
            reset_policy=False,
        )
        response = self.client.predict(req)
        self._cached_chunk = response["actions"]
        self._chunk_index = 0
        return response

    def _next_action(self, obs: Dict[str, np.ndarray], task_description: str, episode_idx: int, step_idx: int) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if self._chunk_index >= len(self._cached_chunk):
            response = self._query_chunk(obs, task_description, episode_idx, step_idx)
        else:
            response = {"did_replan": False, "info": {}}

        raw_action = np.asarray(self._cached_chunk[self._chunk_index], dtype=np.float32)
        self._chunk_index += 1
        action = self.robot_arm.postprocess_action(raw_action)
        return action, raw_action, response

    def run_episode(
        self,
        env: Any,
        task_description: str,
        initial_state: Dict[str, np.ndarray],
        episode_idx: int,
        max_steps: Optional[int] = None,
    ) -> EpisodeResult:
        max_steps = max_steps or self.cfg.max_steps
        if self.cfg.reset_policy_per_episode:
            self.reset_policy_state()

        env.reset()
        obs = env.set_init_state(initial_state)

        actions: List[List[float]] = []
        raw_actions: List[List[float]] = []
        success = False

        for t in range(max_steps + self.cfg.num_steps_wait):
            if t < self.cfg.num_steps_wait:
                obs, _, _, _ = env.step(self.get_dummy_action())
                continue

            action, raw_action, _ = self._next_action(obs, task_description, episode_idx, t)
            raw_actions.append(raw_action.tolist())
            actions.append(action.tolist())
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break

        return EpisodeResult(
            success=success,
            task_description=task_description,
            episode_idx=episode_idx,
            executed_steps=len(actions),
            actions=actions,
            raw_actions=raw_actions,
        )


def evaluate_libero_suite(
    cfg: RunnerConfig,
    benchmark_factory: callable,
    get_env_fn: callable,
    quat_to_axisangle_fn: callable,
) -> list[dict[str, Any]]:
    runner = LiberoRunner(cfg, quat_to_axisangle_fn=quat_to_axisangle_fn)
    task_suite = benchmark_factory()[cfg.task_suite_name]()
    results: list[dict[str, Any]] = []

    total_success = 0
    total_episodes = 0

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_env_fn(task, resolution=cfg.image_size, seed=cfg.seed)

        for episode_idx in range(cfg.num_trials_per_task):
            result = runner.run_episode(
                env=env,
                task_description=task_description,
                initial_state=initial_states[episode_idx],
                episode_idx=episode_idx,
                max_steps=cfg.max_steps,
            )
            result_dict = result.__dict__
            results.append(result_dict)
            total_episodes += 1
            total_success += int(result.success)
            torch.cuda.empty_cache()
            gc.collect()

        env.close()

    summary_path = Path(cfg.out_dir)
    summary_path.mkdir(parents=True, exist_ok=True)
    np.save(summary_path / "results.npy", results, allow_pickle=True)
    print(f"Overall success: {total_success}/{total_episodes} = {total_success / max(total_episodes, 1):.3f}")
    return results
