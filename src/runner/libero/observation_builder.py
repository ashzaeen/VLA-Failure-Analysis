from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from PIL import Image

from src.common.schemas import CanonicalObservation
from src.common.serialization import encode_images
from src.runner.configs import ObservationConfig
from src.runner.libero.action_postprocess import LiberoActionPostprocessor

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None
from src.common.mapping import LIBERO_ENV_IMAGE_MAPPING

# Per-dimension reference scale for the 8-dim LIBERO proprio state
# [eef_pos(x,y,z) in metres, eef axis-angle(3) in radians, gripper_qpos(2)].
# proprio_noise_std multiplies this vector, so std=1.0 injects ~2 cm of positional
# noise, ~0.05 rad of orientation noise, and ~0.005 of gripper noise (1 sigma) — a
# meaningful but not destructive perturbation — while a single swept number keeps the
# relative magnitude comparable across dimensions with very different native units.
PROPRIO_NOISE_REF = np.array(
    [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.005, 0.005], dtype=np.float32
)


@dataclass
class LiberoObservationBuilder:
    cfg: ObservationConfig
    action_postprocessor: LiberoActionPostprocessor

    def __post_init__(self) -> None:
        # Dedicated RNG so the injected proprio noise is reproducible and independent of
        # the global seed (which also drives the env/policy). Seeding this the same way
        # across a sweep means only proprio_noise_std changes the perturbation magnitude.
        self._proprio_rng = np.random.default_rng(self.cfg.proprio_noise_seed)

    def _perturb_proprio(self, state: np.ndarray) -> np.ndarray:
        std = float(self.cfg.proprio_noise_std)
        if std <= 0.0:
            return state
        ref = PROPRIO_NOISE_REF
        if state.shape[0] != ref.shape[0]:
            # Fall back to a flat scale if the proprio layout is unexpected, so an
            # unusual state_dim never silently drops the perturbation.
            ref = np.full(state.shape[0], 0.02, dtype=np.float32)
        noise = self._proprio_rng.standard_normal(state.shape[0]).astype(np.float32) * std * ref
        return state + noise

    def _resize_image(self, image: np.ndarray, resize_size: tuple[int, int]) -> np.ndarray:

        """
        Takes numpy array corresponding to a single image and returns resized image as numpy array.

        NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                        the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
        """
        assert isinstance(resize_size, tuple)
        # Resize to image size expected by model
        img = tf.image.encode_jpeg(image)  # Encode as JPEG, as done in RLDS dataset builder
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
        img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        img = img.numpy()
        return img
    
    def adjust_brightness_contrast(self,image):
        # alpha: 对比度 (1.0 不变)
        # beta: 亮度
        bright = image.astype(np.float32) * self.cfg.alpha + self.cfg.beta
        bright = np.clip(bright, 0, 255).astype(np.uint8)
        return bright

    def _extract_images(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        resize_size = (self.cfg.image_size, self.cfg.image_size)
        images: Dict[str, np.ndarray] = {}
        for key in self.cfg.image_keys:
            obs_key = f'{key}_image'
            image = np.array(obs[obs_key])
            if self.cfg.rotate_180:
                image = np.ascontiguousarray(image[::-1, ::-1])
            # image = self._resize_image(image, resize_size)
            image = self.adjust_brightness_contrast(image)
            env_key = LIBERO_ENV_IMAGE_MAPPING.get(key, key)
            images[env_key] = image
        return images

    def build(self, obs: Dict[str, np.ndarray], task_description: str, metadata: Dict | None = None) -> CanonicalObservation:
        images = self._extract_images(obs)
        encoded_images = encode_images(images)
        state_arr = self.action_postprocessor.preprocess_proprio(obs)
        state_arr = self._perturb_proprio(state_arr)
        state = state_arr.tolist()
        if self.cfg.test_zero_proprio:
            state = [-0.14997, 0.00026115, 0.26816, 3.1401, -0.0014931, -0.083721, 0.038787, -0.038789]
        # print(f'images = {images["agentview"].shape},{images["agentview"]}')
        return CanonicalObservation(
            images_np=images,
            images=encoded_images,
            state=state,
            task=task_description,
            metadata=metadata or {},
        )

