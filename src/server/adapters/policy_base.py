from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePolicyAdapter(ABC):
    def __init__(self, policy_id: str, spec: Dict[str, Any]):
        self.policy_id = policy_id
        self.spec = spec

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

