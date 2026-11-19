import os
import sys
from dataclasses import dataclass
from typing import  Optional
import torch


sys.path.append("../..")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

@dataclass
class LiberoConfig:
    """LIBERO 配置参数"""
    # model_family: str = "openvla"  # "openvla" 或 "pi0" 或 "openvla_oft"
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-object"
    # model_path: str = "openvla/openvla-7b-finetuned-libero-object"
    model_id: str = "openvla/openvla-7b-finetuned-libero-object"
    image_size: int = 256
    action_scale: float = 1.0
    normalize_gripper: bool = True
    invert_gripper: bool = True
    binarize_gripper: bool = True
    num_steps_wait: int = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    center_crop: bool = True  # Center crop? (if trained w/ random crop image aug)
    action_dims : int = 7  # Number of action dimensions (default: 7 for Franka Panda arm)
    #################################################################################################################
    # openvla/prismatic-specific parameters
    #################################################################################################################
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)


    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_object"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    unnorm_key: str = 'libero_object'  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 4  # Number of rollouts per task
    max_steps : int = 300
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    
    
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None  # Extra note to add in run ID for logging
    local_log_dir: str = os.path.join(project_root, 'experiments', 'logs')  # Local directory for eval logs

    use_wandb: bool = False  # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"  # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"  # Name of entity to log under
    out_dir: str ="" # Path to save eval results

    seed: int = 7  # Random Seed (for reproducibility)
    render: bool = False  # Whether to render the environment during eval

    ############################################
    output_attentions: bool = False  # Whether to output attention maps from the model (if supported)
    save_attention_maps: bool = False  # Whether to save attention maps to disk 