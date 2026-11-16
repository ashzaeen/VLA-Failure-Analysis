from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ServerRuntimeConfig:
    host: str
    port: int
    device: str
    dtype: str
    log_level: str


@dataclass
class PolicyConfig:
    policy_id: str
    policy_type: str
    checkpoint_path: str
    model_id: str
    unnorm_key: str
    output_attentions: bool
    input: Dict[str, Any]
    output: Dict[str, Any]

@dataclass
class ServerConfig:
    server: ServerRuntimeConfig
    policies: List[PolicyConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ServerConfig':
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        return cls(
            server=ServerRuntimeConfig(**raw['server']),
            policies=[PolicyConfig(**p) for p in raw['policies']],
        )
