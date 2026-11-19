import os
os.environ.setdefault("MPLBACKEND", "Agg")
import glob
import shutil
import numpy as np
import sys

REPO = os.path.expanduser("~/projects/vla-failure-analysis")
os.chdir(REPO)
sys.path.insert(0, REPO)
from scripts.analyze_attention import analyze_npz_folder_group_by_episode

ROOT = os.path.join(REPO, "outputs", "contrast_sweep_full")
STAGE = os.path.expanduser("~/deliver_stage_contrast")
DELIVER = os.path.expanduser("~/deliver_contrast")
ATTN_OUT = os.path.join(DELIVER, "attention")
ROLL_OUT = os.path.join(DELIVER, "rollouts")
for p in (ATTN_OUT, ROLL_OUT):
    os.makedirs(p, exist_ok=True)

# (condition, task_id, episode, object keyword) — OJ ep3 & choc ep0 succeed at all 3 alphas
SELECT = [
    ("alpha_0.2", 9, 3, "orange_juice"),
    ("alpha_1.0", 9, 3, "orange_juice"),
    ("alpha_2.0", 9, 3, "orange_juice"),
    ("alpha_0.2", 8, 0, "chocolate_pudding"),
    ("alpha_1.0", 8, 0, "chocolate_pudding"),
    ("alpha_2.0", 8, 0, "chocolate_pudding"),
]

def cs(c):
    return c.replace("alpha_", "a")

by_cond = {}
for cond, tid, ep, obj in SELECT:
    by_cond.setdefault(cond, []).append((tid, ep, obj))

for cond, items in by_cond.items():
    sdir = os.path.join(STAGE, cs(cond))
    if os.path.exists(sdir):
        shutil.rmtree(sdir)
    os.makedirs(sdir)
    for tid, ep, obj in items:
        shutil.copy(os.path.join(ROOT, cond, "episodes_npz", f"task_{tid:03d}_episode_{ep:03d}.npz"), sdir)
    out = os.path.join(sdir, "attn_out")
    print(f"\n===== attention for {cs(cond)} ({len(items)} eps) =====", flush=True)
    analyze_npz_folder_group_by_episode(
        npz_dir=sdir, output_dir=out, head_idx=0, patch_grid_size=16, top_k=5,
        attention_key="attention_maps",
    )

print("\n===== collecting =====", flush=True)
for cond, tid, ep, obj in SELECT:
    npz = os.path.join(ROOT, cond, "episodes_npz", f"task_{tid:03d}_episode_{ep:03d}.npz")
    tag = "success" if bool(np.load(npz, allow_pickle=True)["success"]) else "FAIL"
    name = f"{cs(cond)}__{obj}__{tag}.mp4"
    hits = glob.glob(os.path.join(STAGE, cs(cond), "attn_out", "heatmaps", f"task_idx_{tid}_episode_{ep}_*.mp4"))
    if hits:
        shutil.copy(hits[0], os.path.join(ATTN_OUT, name)); print(f"  attention -> {name}")
    else:
        print(f"  !! no attention mp4 {cs(cond)} tid={tid} ep={ep}")
    rhits = glob.glob(os.path.join(ROOT, cond, "rollouts", f"*episode={ep}--*{obj}*.mp4"))
    if rhits:
        shutil.copy(rhits[0], os.path.join(ROLL_OUT, name)); print(f"  rollout   -> {name}")
    else:
        print(f"  !! no rollout mp4 {cs(cond)} {obj} ep={ep}")

print("\nDONE.")
for p in (ATTN_OUT, ROLL_OUT):
    print(p, "->", sorted(os.listdir(p)))
