import os
import sys
import torch
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors

import psutil

from robots.liberoInferenceCfg import LiberoConfig
from robots.LiberoInference import LiberoInference
from src.models.pi0.pi0_model import PI0Model

# os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from libero.libero import get_libero_path,benchmark
from libero.libero.envs import OffScreenRenderEnv


model_id = "lerobot/pi0_libero_finetuned_v044"


def report(tag=""):
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024**3
    print(f"[{tag}] RAM: {ram_gb:.2f} GB")
    if torch.cuda.is_available():
        print(f"[{tag}] CUDA allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"[{tag}] CUDA reserved : {torch.cuda.memory_reserved()/1024**3:.2f} GB")


pi0_path = os.path.abspath("vlas/checkpoints/pi0_libero_finetuned_v044")
print(f"pi0_path: {pi0_path}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", device)

if torch.cuda.is_available():
    print(torch.cuda.memory_summary(device=device, abbreviated=True))

report("before load")

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_object']()

task = task_suite.get_task(0)
task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file
)
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256,
    "camera_widths": 256,
}
report("before env")
env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=128,
    camera_widths=128,
)
report("after env")
out_dir = os.path.abspath('outputs/libero_evaluation_results')

cfg = LiberoConfig(
        # model_family = "pi0",
        pretrained_checkpoint = pi0_path,
        model_id = model_id,
        image_size = 256,
        action_scale = 1.0,
        num_trials_per_task = 1,
        max_steps = 400,
        seed = 1,
        out_dir = out_dir,
        render = True,
        output_attentions = True,
        save_attention_maps = True,

    )

print("before load")
pi_model = PI0Model(model_path=pi0_path,
                    model_id=model_id,
                     device=device)

print("after load")
print("before inference")
infer = LiberoInference(cfg= cfg,vla_model=pi_model)
print("after inference")

if torch.cuda.is_available():
    print(torch.cuda.memory_summary(device=device, abbreviated=True))
report("success")