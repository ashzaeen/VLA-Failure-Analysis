from __future__ import annotations

import base64
import io
from typing import Any,Dict

import cv2
import numpy as np
from PIL import Image


def encode_rgb_to_base64(image: np.ndarray, image_format: str = ".jpg", quality: int = 95) -> str:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    params = []
    if image_format.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(image_format, bgr, params)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def decode_base64_to_rgb(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image from base64")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def encode_images(images: Dict[str, np.ndarray]) -> Dict[str, str]:
    return {key: encode_rgb_to_base64(value) for key, value in images.items()}


def decode_images(images: Dict[str, str]) -> Dict[str, np.ndarray]:
    return {key: decode_base64_to_rgb(value) for key, value in images.items()}

def rgb_to_pil(image: np.ndarray) -> Image.Image:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def tensor_to_jsonable(value: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: tensor_to_jsonable(v) for k, v in value.items()}
    return value
