
import sys
import os
import numpy as np
from typing import Dict, Any, Optional, Union, List
import torch
from torch import Tensor
from dataclasses import dataclass

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from robots.geometry import quat2axisangle




class LiberoRobotArm:
    """LIBERO Franka Panda Arm """

    def __init__(
            self,
            action_scale: float = 1.0,
            normalize_gripper: bool = True,
            invert_gripper: bool = True,
            binarize_gripper: bool = True
    ):
        self.action_scale = action_scale
        self.normalize_gripper = normalize_gripper
        self.invert_gripper = invert_gripper
        self.binarize_gripper = binarize_gripper

    def preprocess_proprio(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        预处理 Franka Panda 机械臂的本体感知数据
        """
        # 提取位置、四元数、夹爪状态
        eef_pos = obs["robot0_eef_pos"]
        eef_quat = obs["robot0_eef_quat"]
        gripper_pos = obs["robot0_gripper_qpos"]

        # 将四元数转换为轴角表示
        axis_angle = quat2axisangle(eef_quat)

        # 拼接所有状态
        state = np.concatenate([eef_pos, axis_angle, gripper_pos])
        return state

    def postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        后处理动作数据
        """
        if isinstance(raw_action, torch.Tensor):
            action = raw_action.detach().cpu().numpy().copy()
        else:
            action = raw_action.copy()

        # 应用动作缩放
        action[:6] = action[:6] * self.action_scale

        # 应用夹爪动作处理
        if self.normalize_gripper:
            action = self.normalize_gripper_action(action)

        if self.invert_gripper:
            action = self.invert_gripper_action(action)

        return action


    def normalize_gripper_action(self, action: np.ndarray) -> np.ndarray:
        """将夹爪动作从[0,1]归一化到[-1,+1]"""
        action = action.copy()
        orig_low, orig_high = 0.0, 1.0
        action[-1] = 2 * (action[-1] - orig_low) / (orig_high - orig_low) - 1

        if self.binarize_gripper:
            action[-1] = np.sign(action[-1])

        return action

    def invert_gripper_action(self, action: np.ndarray) -> np.ndarray:
        """翻转夹爪动作的符号"""
        action = action.copy()
        action[-1] = action[-1] * -1.0
        return action
    
    def unnormalize_gripper_action(self, action: np.ndarray) -> np.ndarray:
        """将夹爪动作从[-1,+1]反归一化到[0,1]"""
        action = action.copy()
        unnorm_low, unnorm_high = 0.0, 1.0
        action[-1] = 0.5 * (action[-1] + 1) * (unnorm_high - unnorm_low) + unnorm_low
        return action