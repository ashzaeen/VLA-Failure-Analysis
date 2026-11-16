from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.zeros(3, dtype=np.float64)
    quat = quat / norm
    xyz = quat[:3]
    w = quat[3]
    angle = 2.0 * np.arctan2(np.linalg.norm(xyz), w)
    if np.linalg.norm(xyz) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = xyz / np.linalg.norm(xyz)
    return axis * angle


@dataclass
class LiberoActionPostprocessor:
    action_scale: float = 1.0
    normalize_gripper: bool = True
    invert_gripper: bool = True
    binarize_gripper: bool = True

    def preprocess_proprio(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']
        gripper_pos = obs['robot0_gripper_qpos']
        axis_angle = quat2axisangle(eef_quat)
        return np.concatenate([eef_pos, axis_angle, gripper_pos]).astype(np.float32)

    def postprocess_action(self, raw_action) -> np.ndarray:
        if torch is not None and isinstance(raw_action, torch.Tensor):
            action = raw_action.detach().cpu().numpy().copy()
        else:
            action = np.asarray(raw_action, dtype=np.float32).copy()
        action[:6] = action[:6] * self.action_scale
        if self.normalize_gripper:
            action = self.normalize_gripper_action(action)
        if self.invert_gripper:
            action = self.invert_gripper_action(action)
        return action

    def normalize_gripper_action(self, action: np.ndarray) -> np.ndarray:
        action = action.copy()
        action[-1] = 2 * action[-1] - 1
        if self.binarize_gripper:
            action[-1] = np.sign(action[-1])
        return action

    def invert_gripper_action(self, action: np.ndarray) -> np.ndarray:
        action = action.copy()
        action[-1] = -1.0 * action[-1]
        return action
