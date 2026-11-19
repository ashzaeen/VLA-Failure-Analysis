from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

src_path = Path("./vlas/checkpoints/rvl_ur5_task1_3_7500/model.safetensors")
dst_dir = Path("./vlas/checkpoints/rvl_ur5_task1_3_7500_fixed")
dst_path = dst_dir / "model.safetensors"

dst_dir.mkdir(parents=True, exist_ok=True)

new_prefix = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model."
old_prefix = "model.paligemma_with_expert.paligemma.model.vision_tower."

# 1. 读取 tensor 和 metadata
state_dict = {}
with safe_open(str(src_path), framework="pt", device="cpu") as f:
    metadata = f.metadata() or {}

    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

print("Original tensor keys:", len(state_dict))
print("Original metadata:")
for mk, mv in metadata.items():
    print(f"  {mk} -> {mv}")

# 2. remap tensor keys
new_state_dict = {}
changed_tensor_keys = 0

for k, v in state_dict.items():
    new_k = k.replace(old_prefix, new_prefix)

    if new_k != k:
        changed_tensor_keys += 1

    if new_k in new_state_dict:
        raise RuntimeError(f"Duplicate tensor key after remap: {new_k}")

    new_state_dict[new_k] = v

# 3. remap metadata keys 和 values
# 这里主要是为了保险：如果 metadata 里也有 vision_tower.vision_model，就一起改。
new_metadata = {}
changed_metadata = 0

for mk, mv in metadata.items():
    new_mk = mk.replace(old_prefix, new_prefix)
    new_mv = mv.replace(old_prefix, new_prefix)

    if new_mk != mk or new_mv != mv:
        changed_metadata += 1

    new_metadata[new_mk] = new_mv

# 4. 确认 embed_tokens metadata 没丢
embed_key = (
    "model.paligemma_with_expert.paligemma.model."
    "language_model.embed_tokens.weight"
)

lm_head_key = (
    "model.paligemma_with_expert.paligemma.lm_head.weight"
)

if embed_key not in new_state_dict:
    print(f"{embed_key} is not a real tensor key, checking metadata...")

    if embed_key in new_metadata:
        print(f"Found tied weight metadata: {embed_key} -> {new_metadata[embed_key]}")
    else:
        print("WARNING: embed_tokens metadata missing. Adding it back.")
        new_metadata[embed_key] = lm_head_key

# 5. 保存 fixed checkpoint，并保留 metadata
save_file(new_state_dict, str(dst_path), metadata=new_metadata)

print("\nSaved fixed checkpoint to:", dst_path)
print("Changed tensor keys:", changed_tensor_keys)
print("Changed metadata entries:", changed_metadata)
print("New tensor keys:", len(new_state_dict))
print("New metadata:")
for mk, mv in new_metadata.items():
    print(f"  {mk} -> {mv}")