"""Constants for attention analysis.

RECONSTRUCTED 2026-06-24 from how scripts/analyze_attention.py and
src/runner/libero/evaluator.py use them.
"""
from __future__ import annotations

# NPZ key under which per-step attention is stored by the evaluator
# (see LiberoEvaluator._save_episode_npz -> "attention_maps").
ATTENTION_KEY = "attention_maps"

# Candidate NPZ keys for the two camera views (front / wrist). The episode NPZ
# written by the evaluator uses "front_images" / "wrist_images".
BASE_IMG_KEY_CANDIDATES = ["front_images", "base_img", "agentview_images"]
WRIST_IMG_KEY_CANDIDATES = ["wrist_images", "wrist_img", "robot0_eye_in_hand_images"]

# Optional task_id -> target-object fallback for the LIBERO-Object suite.
# get_object_names() parses the saved task_description first (reliable); this map
# is only a fallback. NOTE: index order follows the libero_object benchmark task
# list and should be verified against your LIBERO version if used directly.
TASK_TO_OBJECT_LIBERO = {
    0: "alphabet soup",
    1: "cream cheese",
    2: "salad dressing",
    3: "bbq sauce",
    4: "ketchup",
    5: "tomato sauce",
    6: "butter",
    7: "milk",
    8: "orange juice",
    9: "chocolate pudding",
}
