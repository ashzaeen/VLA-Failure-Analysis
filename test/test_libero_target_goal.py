import os
import re
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path


def normalize_name(name: str) -> str:
    return (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def read_bddl_file(bddl_path: str) -> str:
    with open(bddl_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_goal_section(bddl_text: str) -> str:
    """
    粗略提取 (:goal ...) 部分。
    不做完整 Lisp parser，但对 LIBERO BDDL 调试基本够用。
    """
    idx = bddl_text.find("(:goal")
    if idx == -1:
        return ""

    text = bddl_text[idx:]
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if ch == "(":
            if start is None:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if start is not None and depth == 0:
                return text[start : i + 1]

    return text


def parse_goal_predicates(bddl_text: str) -> List[Tuple[str, List[str]]]:
    """
    从 goal section 中抽取类似：
        (On chocolate_pudding_1 basket_1)
        (In mug_1 drawer_1)
        (Open drawer_1)
    返回：
        [("On", ["chocolate_pudding_1", "basket_1"])]
    """
    goal = extract_goal_section(bddl_text)
    if not goal:
        return []

    # 去掉一些逻辑关键字
    ignored = {
        "and",
        "or",
        "not",
        "exists",
        "forall",
        ":goal",
    }

    patterns = re.findall(r"\(([A-Za-z_][A-Za-z0-9_\-]*)\s+([A-Za-z0-9_\-\s]+?)\)", goal)

    predicates = []
    for pred, args_str in patterns:
        pred_norm = pred.strip()
        if pred_norm.lower() in ignored:
            continue

        args = [a.strip() for a in args_str.split() if a.strip()]
        if len(args) == 0:
            continue

        # 过滤掉一些明显不是 object predicate 的项
        if pred_norm.startswith(":"):
            continue

        predicates.append((pred_norm, args))

    return predicates


def infer_target_and_goal_from_predicates(
    predicates: List[Tuple[str, List[str]]]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    返回：
        predicate, target_object, goal_object

    常见情况：
        On target reference
        In target reference
        Inside target reference
        Open target
        Close target
    """
    if not predicates:
        return None, None, None

    # 优先找二元空间关系
    preferred_binary = [
        "on",
        "in",
        "inside",
        "under",
        "nextto",
        "next_to",
        "leftof",
        "rightof",
        "behind",
        "frontof",
        "contains",
    ]

    for pred, args in predicates:
        if len(args) >= 2 and pred.lower() in preferred_binary:
            return pred, args[0], args[1]

    # 否则找任意二元 predicate
    for pred, args in predicates:
        if len(args) >= 2:
            return pred, args[0], args[1]

    # 一元 predicate，比如 Open drawer_1
    pred, args = predicates[0]
    return pred, args[0], None


def find_matching_names(
    query: Optional[str],
    names: List[str],
    extra_keywords: Optional[List[str]] = None,
) -> List[str]:
    """
    在 MuJoCo body/site names 里找和 query 匹配的名字。
    query 可以是 chocolate_pudding_1，也可以是 chocolate pudding。
    """
    if query is None:
        return []

    query_norm = normalize_name(query)
    query_parts = [p for p in query_norm.split("_") if p and not p.isdigit()]

    keywords = set(query_parts)
    if extra_keywords:
        for k in extra_keywords:
            keywords.add(normalize_name(k))

    matched = []
    for name in names:
        name_norm = normalize_name(name)

        # 完整包含
        if query_norm in name_norm or name_norm in query_norm:
            matched.append(name)
            continue

        # 关键词部分匹配
        if keywords and all(k in name_norm for k in keywords):
            matched.append(name)
            continue

    return matched


def get_body_pose(env, body_name: str) -> Dict[str, np.ndarray]:
    sim = env.sim
    body_id = sim.model.body_name2id(body_name)
    return {
        "name": body_name,
        "pos": sim.data.body_xpos[body_id].copy(),
        "quat": sim.data.body_xquat[body_id].copy(),
    }


def get_site_pose(env, site_name: str) -> Dict[str, np.ndarray]:
    sim = env.sim
    site_id = sim.model.site_name2id(site_name)
    return {
        "name": site_name,
        "pos": sim.data.site_xpos[site_id].copy(),
        "mat": sim.data.site_xmat[site_id].copy(),
    }


def get_first_pose_by_name(
    env,
    query: Optional[str],
    object_type: str = "body",
) -> Optional[Dict[str, Any]]:
    """
    优先从 body 里找，找不到可以从 site 里找。
    """
    sim = env.sim

    if query is None:
        return None

    if object_type == "body":
        names = list(sim.model.body_names)
        matched = find_matching_names(query, names)
        if matched:
            return get_body_pose(env, matched[0])

    if object_type == "site":
        names = list(sim.model.site_names)
        matched = find_matching_names(query, names)
        if matched:
            return get_site_pose(env, matched[0])

    return None


def get_object_pose_auto(env, query: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    先找 body，再找 site。
    """
    pose = get_first_pose_by_name(env, query, "body")
    if pose is not None:
        pose["source"] = "body"
        return pose

    pose = get_first_pose_by_name(env, query, "site")
    if pose is not None:
        pose["source"] = "site"
        return pose

    return None


def get_gripper_pose(env, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    尽量兼容不同 robosuite / LIBERO 版本。
    优先：
        obs["robot0_eef_pos"]
    其次：
        env.robots[0].eef_site_id
    再其次：
        搜索 site name 中的 gripper/eef/right_hand
    """
    result = {}

    if obs is not None:
        for key in [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ]:
            if key in obs:
                result[key] = np.array(obs[key]).copy()

        if "robot0_eef_pos" in result:
            result["source"] = "obs"
            return result

    sim = env.sim

    # robosuite 常见方式
    try:
        robot = env.robots[0]
        if hasattr(robot, "eef_site_id"):
            eef_site_id = robot.eef_site_id

            # 有的版本是 int，有的可能是 dict
            if isinstance(eef_site_id, dict):
                first_key = list(eef_site_id.keys())[0]
                eef_site_id = eef_site_id[first_key]

            result["robot0_eef_pos"] = sim.data.site_xpos[eef_site_id].copy()
            result["robot0_eef_mat"] = sim.data.site_xmat[eef_site_id].copy()
            result["source"] = "env.robots[0].eef_site_id"
            return result
    except Exception as e:
        result["eef_site_error"] = repr(e)

    # fallback: 搜索 site name
    candidate_keywords = ["eef", "gripper", "right_hand", "hand"]
    matched_sites = []
    for name in sim.model.site_names:
        lname = name.lower()
        if any(k in lname for k in candidate_keywords):
            matched_sites.append(name)

    if matched_sites:
        site_name = matched_sites[0]
        site_id = sim.model.site_name2id(site_name)
        result["site_name"] = site_name
        result["robot0_eef_pos"] = sim.data.site_xpos[site_id].copy()
        result["robot0_eef_mat"] = sim.data.site_xmat[site_id].copy()
        result["source"] = "site_name_search"
        result["matched_sites"] = matched_sites
        return result

    result["source"] = "not_found"
    return result


def print_related_names(env, keywords: List[str]):
    sim = env.sim
    norm_keywords = [normalize_name(k) for k in keywords if k]

    print("\n[Matched body names]")
    for name in sim.model.body_names:
        lname = normalize_name(name)
        if any(k in lname for k in norm_keywords):
            print("  BODY:", name)

    print("\n[Matched site names]")
    for name in sim.model.site_names:
        lname = normalize_name(name)
        if any(k in lname for k in norm_keywords):
            print("  SITE:", name)


def make_env_for_task(task, camera_size: int = 128):
    bddl_file_path = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_heights=camera_size,
        camera_widths=camera_size,
    )

    return env, bddl_file_path


def debug_one_task(
    suite,
    task_id: int,
    init_state_id: int = 0,
    camera_size: int = 128,
):
    print("\n" + "=" * 100)
    print(f"[Task {task_id}]")

    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)

    print("language:", task.language)
    print("bddl_file:", task.bddl_file)
    print("problem_folder:", task.problem_folder)

    env, bddl_path = make_env_for_task(task, camera_size=camera_size)

    print("bddl_path:", bddl_path)

    bddl_text = read_bddl_file(bddl_path)
    predicates = parse_goal_predicates(bddl_text)
    predicate, target_obj, goal_obj = infer_target_and_goal_from_predicates(predicates)

    print("\n[Parsed BDDL goal predicates]")
    for p, args in predicates:
        print(f"  ({p} {' '.join(args)})")

    print("\n[Inferred]")
    print("predicate:", predicate)
    print("target_obj:", target_obj)
    print("goal_obj/reference_obj:", goal_obj)

    env.seed(0)
    obs = env.reset()

    if initial_states is not None and len(initial_states) > 0:
        state = initial_states[init_state_id]
        obs = env.set_init_state(state)

    # 有些版本 set_init_state 不返回 obs
    if obs is None:
        obs = env.get_observation()

    # 搜索相关 body/site 名称
    related_keywords = []
    if target_obj:
        related_keywords.extend(target_obj.split("_"))
    if goal_obj:
        related_keywords.extend(goal_obj.split("_"))

    # 也从 language 中拿一些关键词，方便 debug
    language_tokens = re.findall(r"[a-zA-Z]+", task.language.lower())
    related_keywords.extend(language_tokens)

    print_related_names(env, related_keywords)

    target_pose = get_object_pose_auto(env, target_obj)
    goal_pose = get_object_pose_auto(env, goal_obj)
    gripper_pose = get_gripper_pose(env, obs)

    print("\n[Target pose]")
    if target_pose is None:
        print("target pose: NOT FOUND")
    else:
        print("source:", target_pose.get("source"))
        print("name:", target_pose.get("name"))
        print("pos:", target_pose.get("pos"))
        if "quat" in target_pose:
            print("quat:", target_pose.get("quat"))

    print("\n[Goal / reference pose]")
    if goal_pose is None:
        print("goal pose: NOT FOUND")
    else:
        print("source:", goal_pose.get("source"))
        print("name:", goal_pose.get("name"))
        print("pos:", goal_pose.get("pos"))
        if "quat" in goal_pose:
            print("quat:", goal_pose.get("quat"))

    print("\n[Gripper pose]")
    for k, v in gripper_pose.items():
        print(f"{k}: {v}")

    if target_pose is not None and goal_pose is not None:
        target_pos = target_pose["pos"]
        goal_pos = goal_pose["pos"]
        delta = goal_pos - target_pos
        dist = np.linalg.norm(delta)

        print("\n[Relative target -> goal]")
        print("delta_xyz:", delta)
        print("distance:", dist)

    if target_pose is not None and "robot0_eef_pos" in gripper_pose:
        eef_pos = gripper_pose["robot0_eef_pos"]
        target_pos = target_pose["pos"]
        delta_eef_target = target_pos - eef_pos
        dist_eef_target = np.linalg.norm(delta_eef_target)

        print("\n[Relative gripper -> target]")
        print("delta_xyz:", delta_eef_target)
        print("distance:", dist_eef_target)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_object",
        help="例如 libero_object, libero_spatial, libero_goal, libero_10",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="指定 task id。如果不指定，则遍历所有 task。",
    )
    parser.add_argument(
        "--start_task_id",
        type=int,
        default=0,
        help="遍历时的起始 task id。",
    )
    parser.add_argument(
        "--init_state_id",
        type=int,
        default=0,
        help="使用第几个 initial state。",
    )
    parser.add_argument(
        "--camera_size",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.task_suite]()

    print("task_suite:", args.task_suite)
    print("n_tasks:", suite.n_tasks)

    if args.task_id is not None:
        debug_one_task(
            suite=suite,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
            camera_size=args.camera_size,
        )
    else:
        for task_id in range(args.start_task_id, suite.n_tasks):
            debug_one_task(
                suite=suite,
                task_id=task_id,
                init_state_id=args.init_state_id,
                camera_size=args.camera_size,
            )


if __name__ == "__main__":
    main()