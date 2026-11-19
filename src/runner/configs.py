
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

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
    # Magnitude of additive Gaussian noise injected into the proprio state before it is
    # sent to the policy server. 0.0 = no perturbation (clean). The value is a unitless
    # multiplier applied to a per-dimension reference scale (see observation_builder.py),
    # so a single number sweeps a comparable magnitude across all 8 proprio dims.
    proprio_noise_std: float = 0.0
    # Seed for the proprio-noise RNG so the injected noise is reproducible across runs
    # (holding this fixed while sweeping proprio_noise_std changes only the magnitude).
    proprio_noise_seed: int = 0


@dataclass
class ScenePerturbationConfig:
    """Stage 3: rearrange objects in the scene before the episode starts.

    Unlike the observation-side perturbations above, this rewrites object poses
    inside the LIBERO initial state, so the robot faces a genuinely different
    table. See src/runner/libero/scene_perturbation.py.
    """

    # none | move_target | move_distractors | swap
    mode: str = 'none'
    # Radial displacement in metres for the move_* modes. LIBERO-Object slots are
    # ~0.1-0.17 m apart, so 0.10 m is already a large move relative to the grid.
    distance: float = 0.0
    # Seeds the per-episode displacement direction. Holding this fixed while
    # sweeping `distance` keeps the direction constant so only magnitude varies.
    seed: int = 0
    # Which distractor the target swaps with in `swap` mode. Negative = cycle
    # through the distractors by episode index (covers all slots over a run).
    swap_partner: int = -1
    # Minimum centre-to-centre gap (m) enforced between two grocery objects.
    min_separation: float = 0.08
    # Larger exclusion radius (m) around the basket, which is much bigger than
    # the grocery items and is the goal receptacle.
    basket_clearance: float = 0.16
    # Table region objects may be placed in (metres, object-frame x/y as stored
    # in the init state). Derived from the observed slot extents plus margin.
    x_bounds: Tuple[float, float] = (-0.25, 0.20)
    y_bounds: Tuple[float, float] = (-0.30, 0.12)
    # Number of candidate directions on the displacement ring.
    direction_candidates: int = 36


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
    # Optional so every pre-Stage-3 config keeps loading unchanged; an absent
    # block means mode='none', i.e. the scene is left exactly as LIBERO ships it.
    scene: ScenePerturbationConfig = field(default_factory=ScenePerturbationConfig)

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
            scene=_scene_from_raw(raw.get('scene')),
        )


def _scene_from_raw(raw: dict | None) -> ScenePerturbationConfig:
    if not raw:
        return ScenePerturbationConfig()
    raw = dict(raw)
    # YAML gives lists for the bound pairs; the dataclass wants tuples.
    for key in ('x_bounds', 'y_bounds'):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(float(v) for v in raw[key])
    return ScenePerturbationConfig(**raw)
