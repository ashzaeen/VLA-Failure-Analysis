import os
import torch
from collections import deque
from typing import Any, Dict

from src.models.base_model import BaseVLAModel
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PI0Model(BaseVLAModel):
    """Thin wrapper around LeRobot PI0Policy."""

    def __init__(
        self,
        model_path: str,
        model_id: str,
        unnorm_key: str = "libero_object",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.unnorm_key = unnorm_key
        self.torch_dtype = torch_dtype

        self.vla = self._load_pi0_model(model_path)
        logger.info(f"Loaded PI0 model from {model_path}")

        # Offline/env-compat: when model_id resolves to a hub repo whose processor
        # pins the gated google/paligemma-3b-pt-224 tokenizer, redirect it to a local
        # ungated copy via PI0_TOKENIZER_PATH (unset => use the processor's own value).
        preprocessor_overrides = {"device_processor": {"device": str(device)}}
        tokenizer_override = os.environ.get("PI0_TOKENIZER_PATH")
        if tokenizer_override:
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_override}
            logger.info(f"Overriding tokenizer_name -> {tokenizer_override}")

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.vla.config,
            model_id,
            preprocessor_overrides=preprocessor_overrides,
        )
        logger.info(f"Loaded PI0 preprocessors for {model_id}")

        self._action_queue = deque()
        self.attention_maps = None
        self.reset()

    def _load_pi0_model(self, model_path: str):
        logger.info(f"Loading PI0 model from {model_path}")
        return PI0Policy.from_pretrained(
            pretrained_name_or_path=model_path,
            local_files_only=True,
            device_map={"": str(self.device)},
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).eval()

    def prepare_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:

        return self.preprocess(dict(observation))

    def predict_action_chunk(self, observation: Dict[str, Any],
                             include_attentions: bool = False,
                             include_nll: bool = False,
                             num_hutchinson_samples: int = 1) -> Dict[str, Any]:
        processed_obs = self.prepare_observation(observation)

        with torch.no_grad():
            if include_attentions:
                # fork returns 6: actions, language_attentions, action_attentions, v_ts, nll_res, obs_embs
                actions, _, action_attentions, v_ts, nll_res, obs_embs = self.vla.predict_action_chunk(
                    processed_obs,
                    output_attentions=True,
                    include_nll=include_nll,
                    num_hutchinson_samples = num_hutchinson_samples,
                )
            else:
                result = self.vla.predict_action_chunk(
                    processed_obs,
                    output_attentions=False,
                    include_nll=include_nll,
                    num_hutchinson_samples = num_hutchinson_samples,
                )
                if isinstance(result, tuple):
                    actions = result[0]
                    v_ts = result[3] if len(result) > 3 else []
                    nll_res = result[4] if len(result) > 4 else None
                    obs_embs = result[5] if len(result) > 5 else None
                else:
                    actions = result
                action_attentions = None
        # print(f"Raw model output actions: {actions[0,0]}")
        out_actions = self.postprocess(actions)
        # logger.info(f"Predicted action chunk: {out_actions[0,:10]}")
        # logger.info(f"Predicted rawaction chunk: {actions[0,:10]}")

        return {
            "raw_actions": actions,
            "actions": out_actions,
            "action_attentions": action_attentions,
            "v_ts": v_ts,
            "nll_res": nll_res,
            "obs_embs": obs_embs,
        }

    def _fill_action_queue(self, observation: Dict[str, Any],
                           include_attentions: bool = False) -> None:
        out = self.predict_action_chunk(observation,include_attentions=include_attentions)
        actions = out["actions"]

        if isinstance(actions, torch.Tensor):
            actions_cpu = actions.detach().cpu()
        else:
            actions_cpu = actions

        if actions_cpu.ndim == 3:
            # [B, T, D]
            for t in range(actions_cpu.shape[1]):
                self._action_queue.append(actions_cpu[0, t])
        elif actions_cpu.ndim == 2:
            # [T, D]
            for t in range(actions_cpu.shape[0]):
                self._action_queue.append(actions_cpu[t])
        else:
            raise ValueError(f"Unexpected action shape: {actions_cpu.shape}")

        self.attention_maps = out["action_attentions"]

    def select_action(self, observation: Dict[str, Any], output_attentions: bool = False) -> Dict[str, Any]:
        self.vla.eval()

        did_replan = False
        if len(self._action_queue) == 0:
            self._fill_action_queue(observation, include_attentions=output_attentions)
            did_replan = True

        action = self._action_queue.popleft()

        return {
            "actions": action,
            "action_attentions": self.attention_maps if output_attentions else None,
            "did_replan": did_replan,
        }

    def reset(self, seed: int = 666):
        self._action_queue.clear()
        self.attention_maps = None
        if hasattr(self.vla, "reset"):
            self.vla.reset()