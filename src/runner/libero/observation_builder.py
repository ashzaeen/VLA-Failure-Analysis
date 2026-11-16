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

@dataclass
class LiberoObservationBuilder:
    cfg: ObservationConfig
    action_postprocessor: LiberoActionPostprocessor

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
        state = self.action_postprocessor.preprocess_proprio(obs).tolist()
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

