from safetensors import safe_open

path = "./vlas/checkpoints/rvl_ur5_task1_4_pick/policy_preprocessor_step_5_normalizer_processor.safetensors"

with safe_open(path, framework="pt", device="cpu") as f:
    print("Keys:")
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(
            key,
            "shape=", tuple(tensor.shape),
            "dtype=", tensor.dtype,
            "min=", tensor.min().item() if tensor.numel() > 0 else None,
            "max=", tensor.max().item() if tensor.numel() > 0 else None,
            "mean=", tensor.float().mean().item() if tensor.numel() > 0 else None,
        )