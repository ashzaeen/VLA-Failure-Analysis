
import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import requests

from src.common.serialization import encode_rgb_to_base64
from src.utils.logger import get_logger
from matplotlib import pyplot as plt
logger = get_logger(__name__)

import tensorflow as tf
from datasets import load_dataset
from typing import Dict, Any
from PIL import Image
import matplotlib.pyplot as plt

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

def center_crop_and_resize( image: np.ndarray, crop_size: int) -> np.ndarray:
        h, w = image.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = image[y0:y0 + side, x0:x0 + side]
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
        return crop

def process_image(image, save_path=None, verbose=True):
    """
    将输入图片统一处理成 HWC 格式、uint8 类型、范围 [0, 255]。

    支持输入：
    - torch.Tensor
    - numpy.ndarray
    - CHW: [C, H, W]
    - HWC: [H, W, C]
    - float: [0, 1] 或 [0, 255]
    - uint8: [0, 255]

    Args:
        image: 输入图片 tensor 或 numpy array
        save_path: 是否保存图片，例如 "agentview.png"
        verbose: 是否打印 shape / dtype / max / min

    Returns:
        img: 处理后的 numpy 图片，格式为 HWC + uint8
    """

    # ---------- tensor → numpy ----------
    if hasattr(image, "detach"):
        img = image.detach().cpu().numpy()
    elif hasattr(image, "numpy"):
        img = image.numpy()
    else:
        img = np.asarray(image)

    if verbose:
        print("raw:", img.shape, img.dtype, img.max(), img.min())

    # ---------- CHW → HWC ----------
    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))

    if verbose:
        print("after transpose:", img.shape, img.dtype, img.max(), img.min())

    # ---------- safe normalize ----------
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255.0

        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

    if verbose:
        print("final:", img.shape, img.dtype, img.max(), img.min())

    # ---------- save image ----------
    if save_path is not None:
        plt.imsave(save_path, img)

    return img

def make_dummy_observation():
    states = np.load("test/episode_008/states.npy", allow_pickle=True)
    img = cv2.imread("test/episode_008/obs/rgb_0000.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    agentview_np = center_crop_and_resize(img, 480)
    plt.imsave("agentview.png", agentview_np)
    # plt.show()
    # agentview_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    wristview_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    agentview = encode_rgb_to_base64(agentview_np)
    wristview = encode_rgb_to_base64(wristview_np)
    state = np.random.randn(8).astype(np.float32)

    payload = {
        "policy_id": "pi0",
        'observation': {
            "images": {
                "agentview": agentview,
                # "wristview": wristview,
            },
            "state": states[0].tolist(),
            # "task_description": ["pick up the orange juice and place it in the basket"],
            "task_description": ["pick up the red cup and place it in the black container"],
        
        },
        'include_attentions': False,
    }
    return payload

def make_dummy_observation_hug():

    ds = LeRobotDataset(
        repo_id="amylingchen/rvl_ur5_task1_2_v30",
        revision="main",
    )

    data = ds[0]
    print("Data keys:", data.keys())

    img = data['observation.images.image'].numpy()

    print("raw:", img.shape, img.dtype, img.max(), img.min())

    # ---------- CHW → HWC ----------
    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))

    print("after transpose:", img.shape, img.dtype, img.max(), img.min())

    # ---------- safe normalize ----------
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255
        img = img.astype(np.uint8)

    print("final:", img.shape, img.dtype, img.max(), img.min())

    agentview_np = img

    plt.imsave("agentview.png", agentview_np)

    agentview = encode_rgb_to_base64(agentview_np)

    state = data['observation.state'].numpy()

    payload = {
        "policy_id": "pi0",
        "observation": {
            "images": {
                "agentview": agentview,
                # "wristview": wristview,
            },
            "state": state.tolist(),
            "task_description": [
                "pick up the red cup and place it in the black container"
            ],
        },
        "include_attentions": False,
    }

    return payload


def make_dummy_observation_hug_1():
    

    ds = LeRobotDataset(
        repo_id="amylingchen/rvl_ur5_task1_2_v30",
        revision="main",
    )

    data = ds[0]
    print("Data keys:", data.keys())

    img1 = data["observation.images.image1"]
    img2 = data["observation.images.image2"]

    agentview_np = process_image(img1, save_path="agentview.png")
    wristview_np = process_image(img2, save_path="wristview.png")

    agentview = encode_rgb_to_base64(agentview_np)
    wristview = encode_rgb_to_base64(wristview_np)

    state = data['observation.state'].numpy()

    payload = {
        "policy_id": "pi0",
        "observation": {
            "images": {
                "agentview": agentview,
                "wristview": wristview,
            },
            "state": state.tolist(),
            "task_description": [
                "pick up the red cup and place it in the black container"
            ],
        },
        "include_attentions": False,
    }

    return payload


def test_predict_api():
    url = "http://0.0.0.0:8001/predict"

    payload = make_dummy_observation_hug_1()
    response = requests.post(url, json=payload, timeout=120)
    print("Response status code:", response.status_code)
    

    assert response.status_code == 200, response.text

    result = response.json()
    # print("Predict API response:", result)
    for i in range(10):
        logger.info(f"Response content:action = {result['actions'][i]}, raw_action = {result.get('raw_actions')[i]}")

    assert "actions" in result
    # assert "did_replan" in result
    assert "attention_last_layer_mean" in result

    action = result["actions"]
    assert isinstance(action, list)
    # assert len(action) == 7

    attn = result["attention_last_layer_mean"]
    if attn is not None:
        # 期望格式: [batch, 1, action_token, input_token]
        assert isinstance(attn, list)
        assert len(attn) == 1
        assert len(attn[0]) == 1

if __name__ == "__main__":
    test_predict_api()