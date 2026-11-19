import os
import numpy as np
import pandas as pd

STAGE = os.path.expanduser("~/deliver_stage_contrast")
conds = ["a0.2", "a1.0", "a2.0"]
rows = []
for c in conds:
    csv = os.path.join(STAGE, c, "attn_out", "sample_level_attention_metrics.csv")
    if not os.path.exists(csv):
        continue
    df = pd.read_csv(csv)
    for fp, g in df.groupby("file_path"):
        rows.append({
            "cond": c,
            "npz": os.path.basename(str(fp)),
            "steps": len(g),
            "base_max_mean": round(g["max_attention_value(base_img)"].mean(), 4),
            "base_entropy_mean": round(g["entropy(base_img)"].mean(), 3),
            "base_gini_mean": round(g["gini(base_img)"].mean(), 3),
            "base_top5_mean": round(g["top5_share(base_img)"].mean(), 3),
        })
out = pd.DataFrame(rows).sort_values(["npz", "cond"])
pd.set_option("display.width", 200)
print(out.to_string(index=False))
