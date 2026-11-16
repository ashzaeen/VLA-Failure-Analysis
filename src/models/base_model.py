
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseVLAModel(ABC):
    """VLA 模型抽象基类"""

    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """选择动作"""
        pass

    @abstractmethod
    def reset(self):
        """重置模型状态"""
        pass