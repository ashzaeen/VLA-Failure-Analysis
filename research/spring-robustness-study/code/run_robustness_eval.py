import argparse
import csv
import sys
import collections
import dataclasses
import logging
import math
import pathlib
import imageio
import numpy as np
import tqdm
import tyro
from perturbations import LinguisticPerturber, VisualPerturber, SpatialPerturber

# LIBERO imports
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# Constants
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_goal" # Defaulting to libero_goal
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data/libero/videos"
    seed: int = 7

# Custom arguments for our perturbations
parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='151')
parser.add_argument('--pert_type', type=str, default='baseline')
parser.add_argument('--intensity', type=int, default=1)
custom_args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

ling_perturber = LinguisticPerturber('perturbations.json')

def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Suite-specific max steps
    suite_max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400
    }
    max_steps = suite_max_steps.get(args.task_suite_name, 300)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    
    # Selection logic for Task IDs
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        if 'HAS_RUN_MATCH' in globals(): break
        
        task_name_str = str(getattr(task, 'name', '')).lower()
        task_keywords = {'151': 'cream_cheese', '75': 'push_the_plate', '132': 'bowl_on_the_stove'}
        target_keyword = task_keywords.get(str(custom_args.task_id), 'UNKNOWN')
        
        if target_keyword not in task_name_str: continue
        globals()['HAS_RUN_MATCH'] = True

        current_instruction = ling_perturber.get_instruction(custom_args.task_id, custom_args.pert_type, custom_args.intensity)
        print("\n" + "!"*80)
        print(f"EXPERIMENT STARTING FOR TASK: {custom_args.task_id}")
        print(f"ACTIVE INSTRUCTION: {current_instruction}")
        print("!"*80 + "\n")

        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env.reset()
            action_plan = collections.deque()
            
            # --- APPLY SPATIAL PERTURBATION ---
            if custom_args.pert_type == 'spatial':
                targets = []
                if custom_args.task_id == '151': targets = ['cream_cheese', 'bowl']
                elif custom_args.task_id == '75': targets = ['plate', 'stove']
                elif custom_args.task_id == '132': targets = ['bowl', 'stove']
                SpatialPerturber.apply_xy_displacement(env, targets, custom_args.intensity)

            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    # Image preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
                    
                    replay_images.append(img)

                    if not action_plan:
                        # Prepare observations
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                            "prompt": str(current_instruction),
                        }

                        # --- APPLY VISUAL PERTURBATION TO ALL CAMERAS ---
                        if 'visual' in custom_args.pert_type:
                            for k in ["observation/image", "observation/wrist_image"]:
                                if 'gaussian' in custom_args.pert_type:
                                    element[k] = VisualPerturber.apply_gaussian_noise(element[k], custom_args.intensity)
                                elif 'brightness' in custom_args.pert_type:
                                    element[k] = VisualPerturber.apply_brightness(element[k], custom_args.intensity)

                            # Save debug view
                            import os
                            from PIL import Image
                            if not os.path.exists('AI_VIEW.png'):
                                Image.fromarray(element["observation/image"]).save('AI_VIEW.png')

                        action_chunk = client.infer(element)["actions"]
                        action_plan.extend(action_chunk[:args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            
            # Append to CSV
            with open('results.csv', mode='a', newline='') as f:
                csv.writer(f).writerow([custom_args.task_id, custom_args.pert_type, custom_args.intensity, episode_idx, done])

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_height": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)