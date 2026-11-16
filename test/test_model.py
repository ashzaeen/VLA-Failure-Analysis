import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import requests
from src.common.serialization import encode_rgb_to_base64
from src.utils.logger import get_logger
from matplotlib import pyplot as plt
logger = get_logger(__name__)
import matplotlib.pyplot as plt

from pathlib import Path
import json
import time
import numpy as np
import requests

def call_policy_api(payload, api_url):
    """
    调用 policy server / API，返回 action response。
    根据你的接口地址修改 api_url。
    """
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    return response.json()
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


def make_payload_from_data(
    data,
    task_description="pick up the red cup and place it in the black container",
    save_dir=None,
    include_attentions=False,
):
    """
    从 LeRobotDataset 的单个 data 构造 API payload。
    同时可以保存图片和 state。

    Args:
        data: ds[i]
        task_description: 任务文本
        save_dir: 当前样本保存目录，例如 outputs/sample_000000
        include_attentions: 是否返回 attention

    Returns:
        payload, extra_data
    """

    img1 = data["observation.images.image1"]
    img2 = data["observation.images.image2"]

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        agentview_path = save_dir / "agentview.png"
        wristview_path = save_dir / "wristview.png"
    else:
        agentview_path = None
        wristview_path = None

    agentview_np = process_image(
        img1,
        save_path=str(agentview_path) if agentview_path is not None else None,
        verbose=False,
    )

    wristview_np = process_image(
        img2,
        save_path=str(wristview_path) if wristview_path is not None else None,
        verbose=False,
    )

    agentview = encode_rgb_to_base64(agentview_np)
    wristview = encode_rgb_to_base64(wristview_np)

    state = data["observation.state"].detach().cpu().numpy()
    action = data["action"].detach().cpu().numpy()
    payload = {
        "policy_id": "pi0",
        "observation": {
            "images": {
                "agentview": agentview,
                "wristview": wristview,
            },
            "state": state.tolist(),
            "task_description": [
                task_description
            ],
        },
        "include_attentions": include_attentions,
    }

    extra_data = {
        "agentview_np": agentview_np,
        "wristview_np": wristview_np,
        "state": state,
        "action":action,
    }

    return payload, extra_data

def extract_action_from_response(response):
    """
    根据 API 返回格式提取 action。
    """
    if "action" in response:
        return response["action"]
    elif "actions" in response:
        return response["actions"]
    else:
        return None


import plotly.graph_objects as go
import numpy as np
from pathlib import Path


def save_3d_trajectory_html(
    original_states,
    predicted_states,
    save_path="trajectory_3d.html",
    title="Original vs Generated Trajectory",
):
    """
    生成可交互 3D 轨迹 HTML，而不是静态图片。

    Args:
        original_states: [T, D], 前3维为 x,y,z
        predicted_states: [T, D], 前3维为 x,y,z
        save_path: 输出 html 文件路径
    """

    original_states = np.array(original_states, dtype=float)
    predicted_states = np.array(predicted_states, dtype=float)

    min_len = min(len(original_states), len(predicted_states))
    original_states = original_states[:min_len]
    predicted_states = predicted_states[:min_len]

    fig = go.Figure()

    # 原始轨迹
    fig.add_trace(
        go.Scatter3d(
            x=original_states[:, 0],
            y=original_states[:, 1],
            z=original_states[:, 2],
            mode="lines+markers",
            name="Original Trajectory",
            marker=dict(size=3),
            line=dict(width=5),
        )
    )

    # 生成 action 累积后的轨迹
    fig.add_trace(
        go.Scatter3d(
            x=predicted_states[:, 0],
            y=predicted_states[:, 1],
            z=predicted_states[:, 2],
            mode="lines+markers",
            name="Generated-Action Trajectory",
            marker=dict(size=3),
            line=dict(width=5),
        )
    )

    # 起点
    fig.add_trace(
        go.Scatter3d(
            x=[original_states[0, 0]],
            y=[original_states[0, 1]],
            z=[original_states[0, 2]],
            mode="markers",
            name="Start",
            marker=dict(size=8, symbol="circle"),
        )
    )

    # 原始终点
    fig.add_trace(
        go.Scatter3d(
            x=[original_states[-1, 0]],
            y=[original_states[-1, 1]],
            z=[original_states[-1, 2]],
            mode="markers",
            name="Original End",
            marker=dict(size=8, symbol="diamond"),
        )
    )

    # 生成轨迹终点
    fig.add_trace(
        go.Scatter3d(
            x=[predicted_states[-1, 0]],
            y=[predicted_states[-1, 1]],
            z=[predicted_states[-1, 2]],
            mode="markers",
            name="Generated End",
            marker=dict(size=8, symbol="square"),
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(
            x=0.02,
            y=0.98,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    save_path = Path(save_path)
    fig.write_html(str(save_path))

    print(f"Saved interactive 3D trajectory to: {save_path}")

    return fig


def run_policy_on_multiple_data(
    repo_id="amylingchen/rvl_ur5_task1_2_v30",
    local_path ="",
    revision="main",
    api_url="http://127.0.0.1:8000/predict",
    output_dir="./policy_outputs",
    start_idx=0,
    num_samples=10,
    task_description="pick up the red cup and place it in the black container",
    sleep_time=0.0,
):
    """
    批量读取 LeRobotDataset 中的数据，调用 policy API，并保存图片、state、action 和 response。

    保存结构示例：

    policy_outputs/
        sample_000000/
            agentview.png
            wristview.png
            state.npy
            action.npy
            payload_meta.json
            response.json
        sample_000001/
            ...
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset(
        repo_id=repo_id,
        root=local_path,
        revision=revision,
    )

    print("Dataset length:", len(ds))

    end_idx = min(start_idx + num_samples, len(ds))

    all_actions = []
    all_states = []
    all_indices = []

    for idx in range(start_idx, end_idx):
        print(f"\nProcessing data index: {idx}")

        sample_dir = output_dir / f"sample_{idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        data = ds[idx]

        payload, extra_data = make_payload_from_data(
            data=data,
            task_description=task_description,
            save_dir=sample_dir,
            include_attentions=False,
        )

        state = extra_data["state"]
        raw_action = extra_data["action"]

        # ---------- save lightweight payload meta ----------
        # 注意：不要把 base64 图片完整保存进 json，会非常大
        payload_meta = {
            "policy_id": payload["policy_id"],
            "state_shape": list(state.shape),
            "task_description": task_description,
            "include_attentions": payload["include_attentions"],
            "dataset_index": idx,
        }

        # ---------- call API ----------
        try:
            response = call_policy_api(payload, api_url=api_url)

            # ---------- extract action ----------
            # 这里需要根据你的 API 返回格式调整
            action = extract_action_from_response(response)
            result_json = {
                "dataset_index": idx,
                "task_description": task_description,

                "state": state.tolist(),
                "state_shape": list(state.shape),
                "data_action": raw_action.tolist(),
                "action": action,

                "images": {
                        "agentview": str(sample_dir / "agentview.png"),
                        "wristview": str(sample_dir / "wristview.png"),
                },
                }
            with open(sample_dir / "response.json", "w") as f:
                json.dump(result_json, f, indent=2)

        except Exception as e:
            print(f"Failed at index {idx}: {e}")

            error_info = {
                "dataset_index": idx,
                "error": str(e),
            }

            with open(sample_dir / "error.json", "w") as f:
                json.dump(error_info, f, indent=2)

        if sleep_time > 0:
            time.sleep(sleep_time)

    # ---------- save merged results ----------

    print("\nFinished.")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    run_policy_on_multiple_data(
        repo_id="amylingchen/rvl_ur5_task1_2_v30",
        local_path="../mydatasets/amylingchen/rvl_ur5_task1_2_v30",
        revision="main",
        api_url="http://0.0.0.0:8001/predict",
        output_dir="./outputs/rvl/policy_outputs_4_15000/datas",
        start_idx=0,
        num_samples=1,
        task_description="pick up the red cup and place it in the black container",
        )