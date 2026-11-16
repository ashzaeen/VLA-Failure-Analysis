import cv2
import os
import re
from typing import Optional, Sequence
import time
import imageio
import numpy as np
from src.utils.logger import get_logger
logger = get_logger(__name__)

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def images_to_video(
    image_folder: str,
    output_video: str,
    fps: int = 10,
    image_exts: Sequence[str] = (".png", ".jpg", ".jpeg"),
    sort_by_number: bool = True,
    resize_to_first_frame: bool = True,
    codec: Optional[str] = None,
) -> str:
    """
    将文件夹中的图片合成为视频。

    Args:
        image_folder: 图片文件夹路径
        output_video: 输出视频路径，如 "/path/to/out.avi" 或 "/path/to/out.mp4"
        fps: 帧率
        image_exts: 允许的图片后缀
        sort_by_number: 是否按文件名中的数字排序
        resize_to_first_frame: 是否将所有帧统一到第一帧尺寸
        codec: 视频编码器。
               若为 None，则根据输出后缀自动选择：
               - .avi -> XVID
               - .mp4 -> mp4v

    Returns:
        输出视频路径

    Raises:
        FileNotFoundError: 图片文件夹不存在
        ValueError: 没有图片 / 第一张图片无法读取 / VideoWriter 初始化失败
    """

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

    def extract_number(filename: str) -> int:
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else 0

    images = [
        img for img in os.listdir(image_folder)
        if os.path.splitext(img)[1].lower() in image_exts
    ]

    if not images:
        raise ValueError(f"No images found in folder: {image_folder}")

    if sort_by_number:
        images = sorted(images, key=extract_number)
    else:
        images = sorted(images)

    logger.info("Sorted images:", images)

    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    if first_frame is None:
        raise ValueError(f"Cannot read first image: {first_image_path}")

    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    if codec is None:
        ext = os.path.splitext(output_video)[1].lower()
        if ext == ".avi":
            codec = "XVID"
        elif ext == ".mp4":
            codec = "mp4v"
        else:
            codec = "XVID"

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not video.isOpened():
        raise ValueError(
            f"Failed to open VideoWriter. output={output_video}, codec={codec}"
        )

    written_count = 0

    for image_name in images:
        img_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(img_path)

        if frame is None:
            logger.info(f"Skip unreadable image: {img_path}")
            continue

        if resize_to_first_frame:
            frame = cv2.resize(frame, (width, height))

        video.write(frame)
        written_count += 1

    video.release()

    if written_count == 0:
        raise ValueError("No valid frames were written to the video.")

    logger.info(f"Video saved: {output_video}")
    logger.info(f"Frames written: {written_count}")
    return output_video


def save_rollout_video(rollout_images,episode_idx, success, task_description, log_file=None,rollout_dir=None) -> str:
    """Saves an MP4 replay of an episode."""
    if rollout_dir is None:
        rollout_dir = os.path.join(project_root,"rollouts",f"{DATE}")
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={episode_idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for i in range(len(rollout_images)):
        video_writer.append_data(rollout_images[i])
    video_writer.close()
    logger.info(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

def save_heatmap_video(heatmap_images,
                       log_file=None, 
                       rollout_dir=None, mp4_path=None) -> str:
    """Saves an MP4 replay of heatmap images."""
    if rollout_dir is None:
        rollout_dir = os.path.join(project_root,"heatmaps",f"{DATE}")
    os.makedirs(rollout_dir, exist_ok=True)
    # processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    # mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={episode_idx}--task={processed_task_description}.mp4"
    mp4_path = f"{rollout_dir}/{mp4_path}"
    video_writer = imageio.get_writer(mp4_path, fps=20)
    for i in range(len(heatmap_images)):
        heatmap_image = heatmap_images[i]
        # heatmap_image = cv2.cvtColor(heatmap_images[i], cv2.COLOR_BGR2RGB)
        video_writer.append_data(heatmap_image)
    video_writer.close()
    logger.info(f"Saved heatmap MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved heatmap MP4 at path {mp4_path}\n")
    return mp4_path

def fig_to_frame(fig):
    fig.canvas.draw()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    try:
        fig.set_constrained_layout(False)
    except:
        pass

    
    # 获取 RGBA buffer
    buf = np.asarray(fig.canvas.buffer_rgba())

    # 转成 RGB（去掉 alpha）
    frame = buf[..., :3].copy()
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame