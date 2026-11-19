#!/usr/bin/env bash
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla-libero
BS="$HOME/projects/vla-failure-analysis/outputs/beta_sweep_attn"
python - "$BS" <<'PY'
import pandas as pd, json, os, sys
bs=sys.argv[1]; rows=[]
bm,be='max_attention_value(base_img)','entropy(base_img)'
for beta in [0,10,20,30,40,50,60,70,80,90,100]:
    d=f"{bs}/beta_{beta}"
    res=json.load(open(f"{d}/results.json"))
    succ=sum(bool(r['success']) for r in res); n=len(res)
    csv=f"{d}/attention_analysis/sample_level_attention_metrics.csv"
    df=pd.read_csv(csv)
    rows.append(dict(beta=beta, success=f"{succ}/{n}", success_rate=round(succ/n,3),
                     base_peak=round(df[bm].mean(),3), base_entropy=round(df[be].mean(),3)))
out=pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(f"{bs}/SWEEP_SUMMARY.csv", index=False)
print("\nsaved:", f"{bs}/SWEEP_SUMMARY.csv")
PY