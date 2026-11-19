import glob
import os
import numpy as np

ROOT = os.path.expanduser("~/projects/vla-failure-analysis/outputs/contrast_sweep_full")
CONDS = ["alpha_0.2", "alpha_0.8", "alpha_1.0", "alpha_2.0"]

base = os.path.join(ROOT, "alpha_1.0", "episodes_npz")
id2obj = {}
for f in sorted(glob.glob(os.path.join(base, "task_*_episode_000.npz"))):
    d = np.load(f, allow_pickle=True)
    id2obj[int(d["task_id"])] = str(d["task_description"])

print("success of episode_000 per task, per contrast condition")
print("tid  " + "  ".join(c.replace("alpha_", "a") for c in CONDS) + "    object")
for tid in sorted(id2obj):
    cells = []
    for c in CONDS:
        f = os.path.join(ROOT, c, "episodes_npz", f"task_{tid:03d}_episode_000.npz")
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            cells.append("T" if bool(d["success"]) else "F")
        else:
            cells.append("?")
    short = id2obj[tid].replace("pick up the ", "").replace(" and place it in the basket", "")
    print(f"  {tid:2d}   " + "     ".join(cells) + f"     {short}")
