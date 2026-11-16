from __future__ import annotations

from typing import Any, Dict, List, Optional

from matplotlib import image
import numpy as np
import torch
from PIL import Image

from src.server.adapters.policy_base import BasePolicyAdapter
from src.models.pi0.pi0_model import PI0Model
from src.common.serialization import decode_images
from src.utils.logger import get_logger
from src.common.observation_builder import resize_image
from src.server.configs import PolicyConfig

logger = get_logger(__name__)
    


class PI0Adapter(BasePolicyAdapter):
    """
    Adapter layer between canonical server observation format and PI0Model.

    Input canonical observation example:
    {
        "images": {
            "agentview": np.ndarray,   # HWC uint8
            "wristview": np.ndarray,   # HWC uint8
        },
        "state": np.ndarray,           # [D]
        "task_description": str,
    }

    Output:
    {
        "action": np.ndarray,                          # [A]
        "attention_last_layer_mean": np.ndarray|None, # [B, 1, Q, K]
        "did_replan": bool,
    }
    """

    def __init__(
        self,
        cfg: PolicyConfig,
        device: str = "cuda",
        image_key_map: Optional[Dict[str, str]] = None,
    ):  
   
        self.model = PI0Model(
            model_path=cfg.checkpoint_path,
            model_id=cfg.model_id,
            unnorm_key=cfg.unnorm_key,
            device=device,
        )
        
        self.state_key = cfg.input["state_key"]
        self.task_key = cfg.input["task_key"]

        self.device = device

        logger.info(f"Initialized PI0Adapter with image_key_map={image_key_map} on device={device}")
        # canonical key -> PI0 expected key
        self.image_key_map = image_key_map or {
            "agentview": "observation.images.image",
            "wristview": "observation.images.image2",
        }

    def reset(self) -> None:
        self.model.reset()
    
    def _to_list_safe(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [self._to_list_safe(v) for v in x]
        return x

    def _to_pil_rgb(self, image: np.ndarray) -> Image.Image:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image as np.ndarray, got {type(image)}")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

        return Image.fromarray(image).convert("RGB")

    def build_model_observation(
        self,
        canonical_observation: Dict[str, Any],
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        if "images" not in canonical_observation:
            raise KeyError("canonical_observation missing 'images'")
        if "state" not in canonical_observation:
            raise KeyError("canonical_observation missing 'state'")
        if "task_description" not in canonical_observation:
            raise KeyError("canonical_observation missing 'task_description'")

        images = decode_images(canonical_observation["images"])
        state = canonical_observation["state"]
        task_description = canonical_observation["task_description"]

        model_observation: Dict[str, Any] = {}

        # -------------------------
        # Images
        # -------------------------
        for canonical_key, pi0_key in self.image_key_map.items():
            if canonical_key not in images:
                raise KeyError(f"canonical_observation['images'] missing '{canonical_key}'")

            tensor = self._image_to_chw_tensor(images[canonical_key])

            # batch_size == 1: keep [C, H, W]
            # batch_size > 1 : convert to [B, C, H, W]
            if batch_size > 1:
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                elif tensor.ndim == 4:
                    if tensor.shape[0] == 1:
                        tensor = tensor.repeat(batch_size, 1, 1, 1)
                    elif tensor.shape[0] != batch_size:
                        raise ValueError(
                            f"Batch size mismatch for image {canonical_key}: "
                            f"got shape {tuple(tensor.shape)}, expected batch_size={batch_size}"
                        )
                else:
                    raise ValueError(
                        f"Unexpected image tensor shape for {canonical_key}: "
                        f"{tuple(tensor.shape)}. Expected [C,H,W] or [B,C,H,W]."
                    )
            else:
                # batch_size == 1
                # keep [C, H, W] if image is single image
                # if already [1, C, H, W], squeeze to [C, H, W]
                if tensor.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                elif tensor.ndim != 3:
                    raise ValueError(
                        f"Unexpected image tensor shape for batch_size=1: "
                        f"{tuple(tensor.shape)}. Expected [C,H,W] or [1,C,H,W]."
                    )

            logger.info(
                f"tensor[{pi0_key}]: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                f"device={tensor.device}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}"
            )

            model_observation[pi0_key] = tensor

        # -------------------------
        # State
        # -------------------------
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        if batch_size > 1:
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0).repeat(batch_size, 1)
            elif state_tensor.ndim == 2:
                if state_tensor.shape[0] == 1:
                    state_tensor = state_tensor.repeat(batch_size, 1)
                elif state_tensor.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch size mismatch for state: "
                        f"got shape {tuple(state_tensor.shape)}, expected batch_size={batch_size}"
                    )
            else:
                raise ValueError(
                    f"Unexpected state shape: {tuple(state_tensor.shape)}. "
                    "Expected [state_dim] or [B,state_dim]."
                )
        else:
            # batch_size == 1
            # keep [state_dim]
            # if already [1, state_dim], squeeze to [state_dim]
            if state_tensor.ndim == 2 and state_tensor.shape[0] == 1:
                state_tensor = state_tensor.squeeze(0)
            elif state_tensor.ndim != 1:
                raise ValueError(
                    f"Unexpected state shape for batch_size=1: "
                    f"{tuple(state_tensor.shape)}. Expected [state_dim] or [1,state_dim]."
                )

        model_observation[self.state_key] = state_tensor

        logger.info(
            f"Built model state: shape={tuple(state_tensor.shape)}, "
            f"dtype={state_tensor.dtype}, device={state_tensor.device}"
        )

        # -------------------------
        # Task description
        # -------------------------
        if isinstance(task_description, str):
            if batch_size == 1:
                model_observation[self.task_key] = [task_description]
            else:
                model_observation[self.task_key] = [task_description] * batch_size

        elif isinstance(task_description, list):
            if batch_size == 1:
                if len(task_description) == 1:
                    model_observation[self.task_key] = task_description
                else:
                    raise ValueError(
                        f"For batch_size=1, task_description list should have length 1, "
                        f"got len={len(task_description)}"
                    )
            else:
                if len(task_description) == batch_size:
                    model_observation[self.task_key] = task_description
                elif len(task_description) == 1:
                    model_observation[self.task_key] = task_description * batch_size
                else:
                    raise ValueError(
                        f"Batch size mismatch for task_description: "
                        f"got len={len(task_description)}, expected batch_size={batch_size}"
                    )
        else:
            raise TypeError(
                f"task_description must be str or list[str], got {type(task_description)}"
            )

        return model_observation
    
    def _to_float_chw_tensor(self, image: np.ndarray) -> torch.Tensor:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(image)}")

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

        # ✅ 正确 resize
        # image_np = resize_image(image, (256, 256))

        # ✅ 再 normalize
        image_np = image.astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        image_tensor = image_tensor.to(self.device)
        return image_tensor

    def _image_to_chw_tensor(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        """Convert input image to CHW torch tensor."""
        image_np = self._to_numpy_image(image)
        
        image_np = image_np.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(torch.float32).to(self.device)
        return tensor

    def _to_numpy_image(self, image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif hasattr(image, "numpy") and not isinstance(image, np.ndarray):
            try:
                image = image.numpy()
            except Exception:
                pass
        elif "Image" in str(type(image)):
            image = np.array(image)

        image = np.asarray(image)

        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image
    def _reduce_last_layer_attention(
        self,
        action_attentions: Any,
    ) -> Optional[List[np.ndarray]]:
        """
        action_attentions is expected to be a sequence of layer attentions.
        last layer shape: [B, H, Q, K]

        Returns:
            np.ndarray with shape [B, 1, Q, K], dtype float16
        """
        if action_attentions is None:
            return None
        if len(action_attentions) == 0:
            return None

        last_layer = action_attentions[-1]

        if not isinstance(last_layer, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for last layer attention, got {type(last_layer)}")

        if last_layer.ndim != 4:
            raise ValueError(f"Expected attention shape [B,H,Q,K], got {tuple(last_layer.shape)}")

        reduced = last_layer.detach().mean(dim=1, keepdim=True)  # [B,1,Q,K]
        return reduced


    def format_model_output(
        self,
        out: Dict[str, Any],
        include_attentions: bool = False,
        batch_size: int = 1,
        include_nll: bool = False,
        include_obs_embs: bool = False,
    ) -> Dict[str, Any]:

        action = out["actions"]
        did_replan = bool(out.get("did_replan", False))

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        action = np.asarray(action, dtype=np.float32)

        raw_action = out.get("raw_actions", action)
        v_ts = out.get("v_ts", None)

        if isinstance(raw_action, torch.Tensor):
            raw_action = raw_action.detach().cpu().numpy()

        raw_action = np.asarray(raw_action, dtype=np.float32)

        if v_ts is not None:
            if isinstance(v_ts, list):
                if len(v_ts) > 0 and isinstance(v_ts[0], torch.Tensor):
                    v_ts = torch.stack(v_ts, dim=0).detach().cpu().numpy()
            elif isinstance(v_ts, torch.Tensor):
                v_ts = v_ts.detach().cpu().numpy()
            elif isinstance(v_ts, np.ndarray):
                pass
            else:
                v_ts = None

        if batch_size > 1:
            response = {
                "actions": np.transpose(action, (1, 0, 2)).tolist(),
                "raw_actions": np.transpose(raw_action, (1, 0, 2)).tolist(),
                "did_replan": did_replan,
                "v_ts": v_ts.tolist() if v_ts is not None else None,
            }
        else:
            response: Dict[str, Any] = {
                "actions": action[0].tolist(),
                "raw_actions": raw_action[0].tolist(),
                "did_replan": did_replan,
                "v_ts": v_ts.tolist() if v_ts is not None else None,
            }

        if include_attentions:
            attn = self._reduce_last_layer_attention(out.get("action_attentions"))

            if isinstance(attn, torch.Tensor):
                attn = attn.detach().to(torch.float32).cpu().numpy()

            if isinstance(attn, np.ndarray):
                attn = attn.astype(np.float32).tolist()

            response["attention_last_layer_mean"] = [attn]
        else:
            response["attention_last_layer_mean"] = None

        if include_nll:
            nll_res = out.get("nll_res", None)

            if nll_res is not None:
                nll_total, nll_per_step, z, div_integral_per_step, logp_z_per_step = nll_res

                response["nll_res"] = {
                    "nll_total": self._to_list_safe(nll_total),
                    "nll_per_step": self._to_list_safe(nll_per_step),
                    "z": self._to_list_safe(z),
                    "div_integral_per_step": self._to_list_safe(div_integral_per_step),
                    "logp_z_per_step": self._to_list_safe(logp_z_per_step),
                }

                logger.info(f"NLL total: {response['nll_res']['nll_total']}")
                logger.info(f"NLL per step: {response['nll_res']['nll_per_step']}")
                logger.info(
                    f"Div integral per step: "
                    f"{response['nll_res']['div_integral_per_step']}"
                )
                logger.info(f"logp_z_per_step: {response['nll_res']['logp_z_per_step']}")
                logger.info(f"z: {z[0,:,:7]}")
            else:
                response["nll_res"] = None
        else:
            response["nll_res"] = None
        
        if include_obs_embs:
            obs_embs = out["obs_embs"][0] #[action_chunk,obs_emb_dim]

            if isinstance(obs_embs, torch.Tensor):
                obs_embs = obs_embs.detach().cpu().numpy()

            if isinstance(obs_embs, np.ndarray):
                obs_embs = obs_embs.astype(np.float32).tolist()

            response["obs_embs"] = obs_embs
        else:            
            response["obs_embs"] = None

        return response
    
    def predict(
        self,
        canonical_observation: Dict[str, Any],
        include_attentions: bool = False,
        batch_size: int = 1,
        include_nll: bool = False,
        nll_hutchinson_samples: int = 1,
        include_obs_embs: bool = False,
    ) -> Dict[str, Any]:

        model_observation = self.build_model_observation(
            canonical_observation,
            batch_size=batch_size,
        )

        out = self.model.predict_action_chunk(
            model_observation,
            include_attentions=include_attentions,
            include_nll=include_nll,
            num_hutchinson_samples=nll_hutchinson_samples,
        )

        return self.format_model_output(
            out=out,
            include_attentions=include_attentions,
            batch_size=batch_size,
            include_nll=include_nll,
            include_obs_embs=include_obs_embs,
        )