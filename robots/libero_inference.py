import os
import sys
import numpy as np
from typing import Dict, Any, Optional, Union, List
import torch
import cv2
import shutil
from pathlib import Path

sys.path.append("../..")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


from robots.libero_arm import LiberoRobotArm
from src.runner.libero.libero_utils import get_libero_env, resize_image, save_rollout_video, resize_with_pad, \
    convert_to_uint8
import gc
from src.utils.logger import get_logger
from src.models.base_model import BaseVLAModel
from src.models.pi0.pi0_model import PI0Model
from PIL import Image


from src.utils.pi0_attention_utils import AnalysisConfig, analyze_single_image_attention
from src.utils.video_utils import images_to_video
from datasets import load_dataset
from robots.libero_inference_cfg import LiberoConfig


logger = get_logger(__name__)
class LiberoInference:
    """LIBERO 推理引擎"""

    def __init__(
            self,
            cfg: LiberoConfig,
            robot_arm: Optional[LiberoRobotArm] = None,
            vla_model: Optional[BaseVLAModel] = None,

    ):
        self.cfg = cfg
        self.robot_arm = robot_arm or LiberoRobotArm(
            action_scale=cfg.action_scale,
            normalize_gripper=cfg.normalize_gripper,
            invert_gripper=cfg.invert_gripper,
            binarize_gripper=cfg.binarize_gripper
        )
        self.vla_model = vla_model or self._load_vla_model(cfg)
        self.image_size = cfg.image_size
        self.num_steps_wait = cfg.num_steps_wait

        self.batch_size = 1

        self.reset()


    def _load_vla_model(self, cfg: LiberoConfig) -> BaseVLAModel:
        """根据配置加载 VLA 模型"""
        return PI0Model(model_path=cfg.pretrained_checkpoint,
                        model_id=cfg.model_id,
                        unnorm_key=cfg.unnorm_key,
                        output_attentions = cfg.output_attentions,
                        device=cfg.device)


    def reset(self):
        """重置推理状态"""
        self.vla_model.reset(seed = self.cfg.seed)
        self.task_description = None
        

    def preprocess_image(self, pil_image: np.ndarray) -> np.ndarray:
        """预处理图像"""

        pil_image = Image.fromarray(pil_image.astype('uint8'))
        pil_image = pil_image.convert("RGB")
            
        return pil_image

    def get_action(
            self,
            obs: Dict[str, np.ndarray],
            task_description: str,
            image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        获取动作

        参数:
            obs: 观测字典，包含机器人状态
            task_description: 任务描述
            image: 可选图像输入

        返回:
            处理后的动作
        """
        # 如果任务描述变化，重置模型
        if task_description != self.task_description:
            self.reset()
            self.task_description = task_description

        # 预处理本体感知数据
        state = self.robot_arm.preprocess_proprio(obs)

        # 预处理图像
        if image is not None:
            processed_image = self.preprocess_image(image[0])
            processed_wrist_image = self.preprocess_image(image[1])
        else:
            # 如果没有提供图像，尝试从观测中获取
            processed_image = self.preprocess_image(obs.get("image", np.zeros((self.image_size, self.image_size, 3))))
            processed_wrist_image = self.preprocess_image(obs.get("image", np.zeros((self.image_size, self.image_size, 3))))

        # 准备模型输入

        observation = {
            'observation.images.image': processed_image,
            'observation.images.image2': processed_wrist_image,
            "observation.state": state,
            "task": [task_description],
        }

        state = observation["observation.state"]
        # 获取模型预测
        raw_action = self._select_action(observation,batch_size=self.batch_size)

        # 后处理动作
        raw_action['raw_action'] = raw_action['action'].copy()
        raw_action['action'] = self.robot_arm.postprocess_action(raw_action['action'] )

        return raw_action

    def _select_action(self, observation,batch_size=1,uq = None):
        raw_action =  self.vla_model.select_action(observation,seed =self.cfg.seed, batch_size=batch_size)
        return raw_action    


    
    def run_episode(
        self,
        env,
        task_description: str,
        initial_states: Dict[str, np.ndarray],
        max_steps: int = 300,
        render: bool = False,
        save_heatmap_video: bool = True,
        episode_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        运行一个完整的 episode，并保存整段 heatmap 视频
        """
        self.reset()

        env.reset()
        obs = env.set_init_state(initial_states)

        success = False
        actions = []

        # 输出目录
        episode_name = task_description.replace(" ", "_")[:60]
        episode_dir = Path(self.cfg.out_dir) / f"episode_{episode_idx}_{episode_name}"
        frame_dir = episode_dir / "heatmap_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        last_frame_rgb = None
        frame_idx = 0

        logger.info("start run episode for heatmap video...")

        for t in range(max_steps + self.num_steps_wait):
            if t < self.num_steps_wait:
                obs, _, _, _ = env.step(self.get_dummy_action())
                continue

            image = self.get_libero_image(obs, self.cfg.image_size)

            action_dict = self.get_action(
                obs,
                task_description,
                image=image,
            )

            action = action_dict["action"]
            action_attentions = action_dict["action_attentions"]
            did_replan = action_dict.get("did_replan", False)

            actions.append(action)

            if save_heatmap_video:
                # 只有新规划时重画；否则沿用上一帧
                if did_replan and action_attentions is not None:
                    last_frame_rgb = make_heatmap_frame(
                        base_img=image[0],
                        wrist_img=image[1],
                        action_attentions=action_attentions,
                        step_idx=t,
                        task_description=task_description,
                    )

                if last_frame_rgb is not None:
                    frame_bgr = cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(frame_dir / f"{frame_idx:04d}.png"), frame_bgr)
                    frame_idx += 1

            obs, _, done, _ = env.step(action.tolist())

            if done:
                success = True
                break

        heatmap_video_path = None
        if save_heatmap_video and frame_idx > 0:
            heatmap_video_path = str(episode_dir / "heatmap_episode.mp4")
            images_to_video(
                image_folder=str(frame_dir),
                output_video=heatmap_video_path,
                fps=10,
            )
            shutil.rmtree(frame_dir, ignore_errors=True)

        return {
            "success": success,
            "task_description": task_description,
            "steps": frame_idx,
            "heatmap_video_path": heatmap_video_path,
            "actions": np.array(actions),
        }

    def get_dummy_action(self):
        """获取虚拟动作（等待时使用）"""
        return [0, 0, 0, 0, 0, 0, -1]

    def get_libero_image(self,obs, resize_size):
        """Extracts image from observations and preprocesses it."""
        assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)
        img1 = obs["agentview_image"]
        img1 = np.ascontiguousarray(img1[::-1, ::-1])  # IMPORTANT: rotate 180 degrees to match train preprocessing
        img2 = obs["robot0_eye_in_hand_image"]
        img2 = np.ascontiguousarray(img2[::-1, ::-1])

        img1 = resize_image(img1, resize_size)
        img2 = resize_image(img2, resize_size)
        imgs = np.stack([img1, img2])
        return imgs

    def get_libero_image_debug(self, obs, resize_size, save_dir=None):
        """
        Debug version of get_libero_image().
        Visualizes preprocessing steps (flip, resize, normalization).

        Args:
            obs: dict, LIBERO observation (contains 'agentview_image' and 'robot0_eye_in_hand_image')
            resize_size: int or (h, w)
            save_dir: optional path to save debug images
        """
        assert isinstance(resize_size, (int, tuple)), "resize_size must be int or tuple"
        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)

        # 原始图像
        raw_agent = obs["agentview_image"]
        raw_hand = obs["robot0_eye_in_hand_image"]

        # 翻转
        flipped_agent = np.ascontiguousarray(raw_agent[::-1, ::-1])
        flipped_hand = np.ascontiguousarray(raw_hand[::-1, ::-1])

        # 拷贝对比用
        before = np.copy(raw_hand)


        resized_agent = resize_image(flipped_agent, resize_size)
        resized_hand = resize_image(flipped_hand, resize_size)
        # 打印对比信息
        print("\n[DEBUG] --- LIBERO Image Preprocessing ---")
        print(f"Original shape: {raw_hand.shape}, After resize: {resized_hand.shape}")
        print(f"Value range before: ({before.min()}, {before.max()}), after: ({resized_hand.min()}, {resized_hand.max()})")
        print(f"Flipped changed pixels: {not np.array_equal(before, flipped_hand)}")
        imgs = np.stack([resized_agent, resized_hand])
        return imgs
        
def make_heatmap_frame(base_img, wrist_img, action_attentions):
    config = AnalysisConfig(
        layer_idx=-1,
        action_token_idx=-1,   # 或者改成当前 action token idx
        head_idx=None,
        alpha=0.35,
        top_k=5,
        patch_grid_size=16,
    )

    base_result = analyze_single_image_attention(
        image=base_img,
        action_attentions=action_attentions,
        image_token_slice=slice(0, 256),
        config=config,
    )

    wrist_result = analyze_single_image_attention(
        image=wrist_img,
        action_attentions=action_attentions,
        image_token_slice=slice(256, 512),
        config=config,
    )

    base_overlay = base_result["overlay_rgb"]
    wrist_overlay = wrist_result["overlay_rgb"]

    frame = np.concatenate([base_overlay, wrist_overlay], axis=1)
    return frame