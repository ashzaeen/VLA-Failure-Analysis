"""
Minimal usage example for pi0_attention_utils.py
Adjust `build_item(...)` to match your notebook variables.
"""
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(),'..'))
print(project_root)
sys.path.append(project_root)

from src.utils.pi0_attention_utils import AnalysisConfig, batch_analyze_pi0_attention

# Example config
config = AnalysisConfig(
    layer_idx=-1,
    action_token_idx=-1,  # average over all action queries
    head_idx=None,        # average over all heads
    alpha=0.35,
    top_k=5,
)


def build_item(data_index, observation, image_pil, wrist_image, object_name=None):
    return {
        "sample_id": data_index,
        "observation": observation,
        "images": {
            "base_img": image_pil,
            "wrist_img": wrist_image,
        },
        "metadata": {
            "object": object_name,
        },
    }


# items = [
#     build_item(data_index, observation, image_pil, wrist_image, object),
#     ...
# ]
#
# df = batch_analyze_pi0_attention(
#     items,
#     policy=policy,
#     preprocess=preprocess,
#     output_dir="pi0_attention_outputs",
#     config=config,
#     save_csv=True,
#     csv_name="attention_metrics.csv",
# )
# print(df.head())
