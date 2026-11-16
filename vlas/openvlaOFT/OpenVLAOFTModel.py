import os
import sys
import numpy as np
import torch
from typing import Dict, Any

from PIL import Image

from src.utils.config import read_yaml
from src.utils.logger import get_logger
from src.models.base_model import BaseVLAModel
from vlas.vla_utils import set_seed_everywhere

from vlas.openvlaOFT.modeling_prismatic import OpenVLAForActionPrediction
from vlas.openvlaOFT.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from vlas.openvlaOFT.constants import PROPRIO_DIM

logger = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_config = read_yaml(os.path.join(PROJECT_ROOT, "configs", "model.yaml"))
config = read_yaml(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OpenVLAOFTModel(BaseVLAModel):
    """OpenVLA-OFT model wrapper."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device = DEVICE):
        self.device = device
        self.cfg = cfg

        self.vla = None
        self.processor = None
        self.action_head = None
        self.proprio_projector = None

        self.unnorm_key = cfg.unnorm_key
        self.pretrained_checkpoint = cfg.pretrained_checkpoint

        self._load_openvla_model(cfg)
        self.reset()

    def _load_openvla_model(self, cfg: Dict[str, Any]):
        logger.info(
            f"Loading OpenVLA-OFT model from {self.pretrained_checkpoint}"
        )

        self.processor = get_processor(cfg)
        self.vla = get_vla(cfg, device=self.device)

        self.action_head = get_action_head(
            cfg, llm_dim=self.vla.llm_dim, device=self.device
        )
        self.proprio_projector = get_proprio_projector(
            cfg,
            llm_dim=self.vla.llm_dim,
            proprio_dim=PROPRIO_DIM,
            device=self.device,
        )

        logger.info("OpenVLA-OFT model loaded")

    def select_action(self, observation) -> Dict[str, Any]:
        actions, hidden_states, normalized_actions, logits,_ = get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            observation,
            observation["task"][0],
            self.action_head,
            self.proprio_projector,
        )

        tokens, _ = torch.topk(logits, k=200, dim=2)

        return {
            "action": actions,
            "embedding": hidden_states.detach()
            .cpu()
            .to(torch.float32)
            .numpy(),
            "token": tokens.detach().cpu().to(torch.float32).numpy(),
            "normal_action": normalized_actions,
        }

    def reset(self, seed: int = config["seed"]):
        set_seed_everywhere(seed)

    def preprocess_image(self, image: np.ndarray) -> Image.Image:
        return Image.fromarray(image.astype("uint8")).convert("RGB")
