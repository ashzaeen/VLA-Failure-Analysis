import os
import random
import sys
import numpy as np
import torch

sys.path.append("../..")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f"Project root: {project_root}")
from src.runner.libero.libero_utils import get_libero_env, save_rollout_video
from libero.libero import benchmark
import gc
from src.utils.logger import get_logger



from src.utils.pi0_attention_utils import AnalysisConfig, analyze_single_image_attention
from src.utils.video_utils import images_to_video
from robots.LiberoInference import LiberoInference
from robots.liberoInferenceCfg import LiberoConfig



logger = get_logger(__name__)
def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

# 使用示例
def setup_libero_evaluation(cfg: LiberoConfig):
    """设置 LIBERO 评估环境"""
    # 创建推理引擎
    inference = LiberoInference(cfg)
    print("Initialized LIBERO inference engine.")

    return inference

def get_task_suite(cfg: LiberoConfig):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    return task_suite

# 主要的评估函数
def evaluate_libero_tasks(cfg: LiberoConfig):
    """评估 LIBERO 任务套件"""
    task_suite = get_task_suite(cfg)
    inference = setup_libero_evaluation(cfg)
    results = []
    set_seed_everywhere(cfg.seed)

    num_tasks_in_suite = task_suite.n_tasks
    total_task_idx = 0
    total_success = 0
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        print(task)
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task,resolution=256,seed=cfg.seed)
        print(f"Evaluating task {task_id}/{num_tasks_in_suite}: {task_description}")
        # task_description ='pick up the orange juice'
        for episode_idx in range(cfg.num_trials_per_task):
            result = inference.run_episode(
                        env,
                        task_description,
                        initial_states[episode_idx],
                        max_steps=cfg.max_steps,
                        episode_idx=episode_idx
                        )
            
            total_task_idx+=1
            if cfg.render:
                # for i in range(result['images'].shape[1]):
                for i in range(1):
                    save_rollout_video(
                        result['images'][:,i],f'{total_task_idx}-{i}' , success=result['success'], task_description=task_description,
                    )
            if result['success']:
                total_success += 1
            result['episode_idx'] =episode_idx
            print(f"episode {episode_idx}, task ={task_description},success ={result['success']}")
            results.append({k:v for k, v in result.items() if k != "images"})
            torch.cuda.empty_cache()
            gc.collect()
    print(f"Overall Success Rate: {total_success}/{total_task_idx} = {total_success/total_task_idx:.2f}") 
    env.close()
    return results


if __name__ == "__main__":
    # 配置示例
    seed = 1
    model_path = os.path.join(project_root,'vlas','checkpoints','pi0_libero_finetuned_v044')
    model_id = "lerobot/pi0_libero_finetuned_v044"
    out_dir = os.path.join(project_root,'outputs','libero_evaluation_results')
    cfg = LiberoConfig(
        # model_family = "pi0",
        pretrained_checkpoint = model_path,
        model_id = model_id,
        image_size = 256,
        action_scale = 1.0,
        num_trials_per_task = 1,
        max_steps = 400,
        seed = seed,
        out_dir = out_dir,
        render = True,
        output_attentions = True,
        save_attention_maps = True,

    )
    for key, value in cfg.__dict__.items():
        print(f"{key}: {value}")

    res = evaluate_libero_tasks(cfg)
