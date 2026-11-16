from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class CanonicalObservation:
    images_np: Dict[str, np.ndarray]
    images: Dict[str, str]
    state: List[float]
    task: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictRequest:
    policy_id: str
    observation: CanonicalObservation
    include_attentions: bool = False
    batch_size: int = 1
    include_nll: bool = False
    nll_hutchinson_samples: int = 1
    include_obs_embs: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'policy_id': self.policy_id,
            'observation': {
                'images': self.observation.images,
                'state': self.observation.state,
                'task_description': [self.observation.task],
                'metadata': self.observation.metadata,
            },
            'include_attentions': self.include_attentions,
            'batch_size': self.batch_size,
            'include_nll': self.include_nll,
            'nll_hutchinson_samples': self.nll_hutchinson_samples,
            'include_obs_embs': self.include_obs_embs,
        }


@dataclass
class PredictResponse:
    action: List[float]
    raw_action: List[float]
    did_replan: bool
    chunk_remaining: int
    action_attentions: Optional[Any] = None
    v_ts: Optional[Any] = None
    nll_res: Optional[float] = None
    nll_steps: Optional[List[float]] = None
    obs_embs: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'PredictResponse':
        return cls(
            action=payload['action'],
            raw_action=payload.get('raw_action', payload['action']),
            did_replan=payload.get('did_replan', False),
            chunk_remaining=payload.get('chunk_remaining', 0),
            action_attentions=payload.get('action_attentions'),
            v_ts=payload.get('v_ts'),
            nll_res=payload.get('nll_res'),
            nll_steps=payload.get('nll_steps'),
            obs_embs=payload.get('obs_embs'),
        )
