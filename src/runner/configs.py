
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str


@dataclass
class EnvConfig:
    task_suite: str
    resolution: int
    max_steps: int
    num_trials_per_task: int
    num_steps_wait: int
    render: bool


@dataclass
class ObservationConfig:
    image_size: int
    image_keys: List[str]
    rotate_180: bool
    use_proprio: bool
    state_dim: int
    alpha: float
    beta: float
    test_zero_proprio: bool


@dataclass
class RobotConfig:
    action_scale: float
    normalize_gripper: bool
    invert_gripper: bool
    binarize_gripper: bool


@dataclass
class PolicyClientConfig:
    base_url: str
    timeout_seconds: int
    policy_id: str
    use_chunk: bool
    replan_threshold: int 
    include_attentions: bool = False
    run_steps: int = 1
    batch_size: int = 1
    include_nll: bool = False 
    nll_hutchinson_samples: int = 1
    include_obs_embs: bool = False


@dataclass
class LoggingConfig:
    save_rollout_video: bool
    save_attention: bool
    save_logs: bool
    save_episode_npz: bool = False


@dataclass
class RunnerConfig:
    experiment: ExperimentConfig
    env: EnvConfig
    observation: ObservationConfig
    robot: RobotConfig
    policy_client: PolicyClientConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'RunnerConfig':
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        return cls(
            experiment=ExperimentConfig(**raw['experiment']),
            env=EnvConfig(**raw['env']),
            observation=ObservationConfig(**raw['observation']),
            robot=RobotConfig(**raw['robot']),
            policy_client=PolicyClientConfig(**raw['policy_client']),
            logging=LoggingConfig(**raw['logging']),
        )
