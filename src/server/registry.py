from __future__ import annotations

from typing import Dict

from src.server.adapters.pi0_adapter import PI0Adapter
from src.server.configs import PolicyConfig, ServerConfig

from src.utils.logger import get_logger
logger = get_logger(__name__)

class PolicyRegistry:
    def __init__(self):
        self._policies: Dict[str, object] = {}

    def register_from_config(self, config: ServerConfig):
        for policy_cfg in config.policies:
            if policy_cfg.policy_type == 'pi0':
                logger.info(f"Registering PI0 policy: {policy_cfg}")
                image_keys = policy_cfg.input.get("image_keys", [])
                image_key_map = {}
                if len(image_keys) > 0:
                    image_key_map["agentview"] = image_keys[0]
                if len(image_keys) > 1:
                    image_key_map["wristview"] = image_keys[1]
                adapter = PI0Adapter(
                    cfg=policy_cfg,
                    image_key_map=image_key_map,
                    device=config.server.device,
                )
            else:
                raise ValueError(f'Unsupported policy_type: {policy_cfg.policy_type}')
            self._policies[policy_cfg.policy_id] = adapter

    def get(self, policy_id: str):
        if policy_id not in self._policies:
            raise KeyError(f'Unknown policy_id: {policy_id}')
        return self._policies[policy_id]

    def list_policy_ids(self):
        return sorted(self._policies.keys())
