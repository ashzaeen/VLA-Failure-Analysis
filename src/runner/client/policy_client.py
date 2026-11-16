from collections import deque

import requests
import numpy as np
from src.common.schemas import PredictRequest, PredictResponse
from src.runner.configs import PolicyClientConfig

from src.utils.logger import get_logger
logger = get_logger(__name__)

class RemotePolicyClient:
    def __init__(self, cfg: PolicyClientConfig):
        self.cfg = cfg
        self._chunk_queue = deque()
        self._chunk_queue_raw = deque()
        self._last_attentions = None
        self._v_ts_queue = deque()
        self._last_nll_total = None
        self._nll_step_queue = deque()
        self._embeddings = deque()
        

    def health(self) -> dict:
        response = requests.get(f'{self.cfg.base_url}/health', timeout=self.cfg.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def reset(self) -> None:
        self._chunk_queue.clear()
        self._last_attentions = None
        self._chunk_queue_raw.clear()
        self._v_ts_queue.clear()
        self._last_nll_total = None
        self._nll_step_queue.clear()

    def predict(self, observation ) -> PredictResponse:
        did_replan = False
        v_ts = None
        # nll_res = None
        if (not self.cfg.use_chunk) or len(self._chunk_queue) <= self.cfg.replan_threshold:
            req = PredictRequest(
                policy_id=self.cfg.policy_id,
                observation=observation,
                include_attentions=self.cfg.include_attentions,
                batch_size=self.cfg.batch_size,
                include_nll=self.cfg.include_nll,
                nll_hutchinson_samples=self.cfg.nll_hutchinson_samples,
                include_obs_embs=self.cfg.include_obs_embs,
            )
            # logger.info(f"Sending predict request: {req.to_dict()}")
            response = requests.post(
                f'{self.cfg.base_url}/predict',
                json=req.to_dict(),
                timeout=self.cfg.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            self._chunk_queue.clear()
            step = self.cfg.run_steps if self.cfg.use_chunk else 1
            v_ts = np.array(payload.get('v_ts')) if 'v_ts' in payload else None
            logger.info(f"Predicted action chunk: {payload['actions'][2]}")
            logger.info(f"Received v_ts shape: {v_ts.shape if v_ts is not None else 'None'}")  # Debug print
            v_ts =  np.transpose(v_ts,(2,0,1,3))
            if self.cfg.include_nll:
                # logger.info(f"Received nll_res: {payload.get('nll_res')}")  # Debug print
                nll_steps = payload.get('nll_res', {}).get('nll_per_step', [None]*step)
                nll_steps = np.array(nll_steps)
                logger.info(f"Received nll_steps: {nll_steps.shape}")      # Debug print
                nll_steps = np.transpose(nll_steps, (1,0))


            if self.cfg.include_obs_embs:
                obs_embs = payload.get('obs_embs')
                logger.info(f"Received obs_embs shape: {np.array(obs_embs).shape if obs_embs is not None else 'None'}")  # Debug print
            
            for i in range(step):
                action = payload['actions'][i]
                self._chunk_queue.append(action)
                raw_action = payload.get('raw_actions', payload['actions'])[i]
                self._chunk_queue_raw.append(raw_action)
                self._v_ts_queue.append(v_ts[i] if v_ts is not None else [None])
                self._nll_step_queue.append(nll_steps[i] if self.cfg.include_nll  else [None])

                self._embeddings.append(obs_embs[i] if self.cfg.include_obs_embs else [None])
                
            self._last_attentions = payload.get('attention_last_layer_mean')
            self._last_nll_total = payload.get('nll_res', {}).get('nll_total', None) if self.cfg.include_nll else None
            did_replan = True

            
        # src/analysisprint(f"Chunk queue length: {self._chunk_queue}, did_replan: {did_replan}")
        action = self._chunk_queue.popleft()
        raw_action = self._chunk_queue_raw.popleft()
        v_ts = self._v_ts_queue.popleft()

        nll_step = self._nll_step_queue.popleft() if self.cfg.include_nll else None
        obs_embs = self._embeddings.popleft() if self.cfg.include_obs_embs else None

        # logger.info(f"Using action from chunk queue: {action}, raw_action: {raw_action},did_replan: {did_replan}")
        return PredictResponse(
            action=action,
            raw_action=raw_action,
            did_replan=did_replan,
            chunk_remaining=len(self._chunk_queue),
            action_attentions=self._last_attentions,
            v_ts=v_ts,
            nll_res=self._last_nll_total,
            nll_steps=nll_step ,
            obs_embs=obs_embs,
        )
