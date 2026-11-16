
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from itertools import product
from typing import Literal
from collections import defaultdict

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = Any

import torch


@dataclass
class DatasetBuildConfig:
    resize_size: Tuple[int, int] = (224, 224)
    image_key: str = "observation.images.image"
    wrist_image_key: str = "observation.images.wrist_image"
    output_base_image_key: str = "observation.images.image"
    output_wrist_image_key: str = "observation.images.image2"
    state_key: str = "observation.state"
    task_key: str = "task"
    object_name: Optional[str] = None
    instruction_template: str = "pick up the {object} and place it in the basket"
    normalize_to_unit: bool = True
    include_raw_images: bool = True
    
def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img

def image_to_chw_tensor(
    image: Any,
    *,
    resize_size: Tuple[int, int] = (224, 224),
    normalize_to_unit: bool = True,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert input image to CHW torch tensor."""
    image_np = _to_numpy_image(image)
    image_np = resize_image(image_np, resize_size)
    if normalize_to_unit:
        image_np = image_np.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def _to_numpy_image(image: Any) -> np.ndarray:
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

def prepare_observation(
    observation: Dict[str, Any],
    *,
    batch: int = 1,
    device: Optional[torch.device | str] = None,
) -> Dict[str, Any]:
    """
    Convert a single-sample observation dict into a batched observation dict.
    Expected image tensors are CHW and state is 1D.
    """
    result = dict(observation)

    for key, value in list(result.items()):
        if key.startswith("observation.images"):
            if isinstance(value, torch.Tensor):
                img_tensor = value
            elif isinstance(value, np.ndarray):
                img_tensor = torch.from_numpy(value)
            elif "Image" in str(type(value)):
                img_tensor = torch.from_numpy(np.array(value))
            else:
                continue

            if img_tensor.ndim == 3 and img_tensor.shape[-1] in (1, 3):
                img_tensor = img_tensor.permute(2, 0, 1)

            if device is not None:
                img_tensor = img_tensor.to(device)

            result[key] = img_tensor.unsqueeze(0).repeat(batch, 1, 1, 1)

    state_value = result.get("observation.state")
    if state_value is not None:
        if not isinstance(state_value, torch.Tensor):
            state_tensor = torch.tensor(state_value)
        else:
            state_tensor = state_value
        state_tensor = state_tensor.float()
        if device is not None:
            state_tensor = state_tensor.to(device)
        result["observation.state"] = state_tensor.unsqueeze(0).repeat(batch, 1)

    task_value = result.get("task")
    if isinstance(task_value, str):
        result["task"] = [task_value] * batch
    elif isinstance(task_value, list):
        if len(task_value) == 1:
            result["task"] = task_value * batch
        else:
            result["task"] = task_value

    return result


def build_task_text(
    *,
    sample: Optional[Mapping[str, Any]] = None,
    object_name: Optional[str] = None,
    task_text: Optional[str] = None,
    task_builder: Optional[Callable[[Mapping[str, Any]], str]] = None,
    template: str = "pick up the {object} and place it in the basket",
) -> str:
    if task_text is not None:
        return task_text
    if task_builder is not None and sample is not None:
        return task_builder(sample)
    if object_name is not None:
        return template.format(object=object_name)
    if sample is not None and "task" in sample and isinstance(sample["task"], str):
        return sample["task"]
    raise ValueError("Unable to build task text. Provide task_text, task_builder, or object_name.")


def build_observation_from_sample(
    sample: Mapping[str, Any],
    *,
    device: Optional[torch.device | str] = None,
    config: Optional[DatasetBuildConfig] = None,
    object_name: Optional[str] = None,
    task_text: Optional[str] = None,
    task_builder: Optional[Callable[[Mapping[str, Any]], str]] = None,
) -> Dict[str, Any]:
    """
    Build a single observation dict from one dataset sample.
    """
    cfg = config or DatasetBuildConfig()
    object_name = object_name if object_name is not None else cfg.object_name

    base_tensor = image_to_chw_tensor(
        sample[cfg.image_key],
        resize_size=cfg.resize_size,
        normalize_to_unit=cfg.normalize_to_unit,
        device=device,
    )
    wrist_tensor = image_to_chw_tensor(
        sample[cfg.wrist_image_key],
        resize_size=cfg.resize_size,
        normalize_to_unit=cfg.normalize_to_unit,
        device=device,
    )

    state_tensor = torch.tensor(sample[cfg.state_key], dtype=torch.float32)
    if device is not None:
        state_tensor = state_tensor.to(device)

    final_task = build_task_text(
        sample=sample,
        object_name=object_name,
        task_text=task_text,
        task_builder=task_builder,
        template=cfg.instruction_template,
    )

    return {
        cfg.state_key: state_tensor,
        cfg.output_base_image_key: base_tensor,
        cfg.output_wrist_image_key: wrist_tensor,
        cfg.task_key: [final_task],
    }


def build_analysis_item_from_sample(
    sample: Mapping[str, Any],
    *,
    sample_id: Any,
    device: Optional[torch.device | str] = None,
    config: Optional[DatasetBuildConfig] = None,
    object_name: Optional[str] = None,
    task_text: Optional[str] = None,
    task_builder: Optional[Callable[[Mapping[str, Any]], str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or DatasetBuildConfig()
    observation = build_observation_from_sample(
        sample,
        device=device,
        config=cfg,
        object_name=object_name,
        task_text=task_text,
        task_builder=task_builder,
    )

    images = {
        "base_img": sample[cfg.image_key],
        "wrist_img": sample[cfg.wrist_image_key],
    }

    out_metadata = dict(metadata or {})
    if cfg.include_raw_images:
        out_metadata.setdefault("task_index", sample.get("task_index"))

    return {
        "sample_id": sample_id,
        "observation": observation,
        "images": images,
        "metadata": out_metadata,
    }


def build_analysis_items_from_dataset(
    dataset_split: Any,
    *,
    indices: Optional[Sequence[int]] = None,
    device: Optional[torch.device | str] = None,
    config: Optional[DatasetBuildConfig] = None,
    object_name: Optional[str] = None,
    object_names: Optional[Sequence[str]] = None,
    pair_mode: Literal["product", "zip"] = "product",
    task_builder: Optional[Callable[[Mapping[str, Any]], str]] = None,
    metadata_builder: Optional[Callable[[Mapping[str, Any], int], Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Batch-build analysis items from a Hugging Face dataset split.

    Supported modes:
    - object_name + indices: single object for all indices
    - object_names + indices, pair_mode="product": every index with every object
    - object_names + indices, pair_mode="zip": object_names[i] matches indices[i]
    """
    cfg = config or DatasetBuildConfig()

    if indices is None:
        indices = list(range(len(dataset_split)))

    # Normalize object list
    if object_names is not None and object_name is not None:
        raise ValueError("Provide either object_name or object_names, not both.")

    if object_names is None:
        if object_name is not None:
            object_names = [object_name]
        elif cfg.object_name is not None:
            object_names = [cfg.object_name]
        else:
            object_names = [None]

    items: List[Dict[str, Any]] = []

    if pair_mode == "zip":
        if len(indices) != len(object_names):
            raise ValueError(
                f"When pair_mode='zip', len(indices) must equal len(object_names). "
                f"Got {len(indices)} and {len(object_names)}."
            )

        pairs = zip(indices, object_names)
    elif pair_mode == "product":
        pairs = product(indices, object_names)
    else:
        raise ValueError(f"Unsupported pair_mode: {pair_mode}")

    for idx, obj_name in pairs:
        sample = dataset_split[idx]
        metadata = dict(metadata_builder(sample, idx) if metadata_builder else {})
        metadata["object_name"] = obj_name

        items.append(
            build_analysis_item_from_sample(
                sample,
                # sample_id=f"{idx}:{obj_name}" if obj_name is not None else idx,
                sample_id=idx,
                device=device,
                config=cfg,
                object_name=obj_name,
                task_builder=task_builder,
                metadata=metadata,
            )
        )

    return items



def find_episode_field(sample: Dict[str, Any]) -> str:
    """
    自动猜测 episode 字段名。
    """
    candidates = [
        "episode_index",
        "episode_id",
        "episode",
        "traj_index",
        "trajectory_index",
        "sequence_index",
    ]
    for key in candidates:
        if key in sample:
            return key
    raise KeyError(
        f"Cannot find episode field in sample keys: {list(sample.keys())}"
    )


def find_step_field(sample: Dict[str, Any]) -> Optional[str]:
    """
    自动猜测 step 字段名。
    """
    candidates = [
        "frame_index",
        "step_index",
        "step",
        "timestep",
        "index_in_episode",
    ]
    for key in candidates:
        if key in sample:
            return key
    return None


def build_episode_to_indices(dataset) -> Dict[Any, List[int]]:
    """
    把 dataset 中每个 episode 对应的全局 sample index 收集起来。
    并按 step 顺序排序（如果有 step 字段）。
    """
    if len(dataset) == 0:
        return {}

    first_sample = dataset[0]
    episode_field = find_episode_field(first_sample)
    step_field = find_step_field(first_sample)

    episode_to_items = defaultdict(list)

    for global_idx in range(len(dataset)):
        sample = dataset[global_idx]
        ep_id = sample[episode_field]
        step_val = sample.get(step_field, global_idx) if step_field else global_idx
        episode_to_items[ep_id].append((step_val, global_idx))

    episode_to_indices = {}
    for ep_id, pairs in episode_to_items.items():
        pairs = sorted(pairs, key=lambda x: x[0])
        episode_to_indices[ep_id] = [idx for _, idx in pairs]

    return episode_to_indices

