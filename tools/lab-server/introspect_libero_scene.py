"""Introspect LIBERO-Object scene structure.

Groundwork for Stage 3 object-position perturbations: we need to know, for each
task, (a) which movable objects exist, (b) where their free-joint qpos lives in
the flattened sim state used by set_init_state, and (c) the spread of object
positions across the provided init states (which bounds how far we can move
things and stay on the table).

Writes a JSON report; prints a human-readable digest.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITE = "libero_object"
OUT = os.path.expanduser("~/libero_object_scene_report.json")


def free_joint_map(sim):
    """name -> (qpos_addr, qvel_addr) for every free (6-dof) joint in the model."""
    out = {}
    model = sim.model
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != 0:  # 0 == mjJNT_FREE
            continue
        name = model.joint_id2name(jid)
        out[name] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
    return out


def main():
    suite = benchmark.get_benchmark_dict()[SUITE]()
    report = {"suite": SUITE, "n_tasks": suite.n_tasks, "tasks": []}

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        init_states = np.asarray(init_states)

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(7)
        env.reset()

        sim = env.env.sim
        nq, nv = int(sim.model.nq), int(sim.model.nv)
        fj = free_joint_map(sim)

        # robosuite object handles (name -> joint names)
        obj_joints = {}
        try:
            for obj in env.env.objects:
                obj_joints[obj.name] = list(obj.joints)
        except Exception as exc:  # pragma: no cover
            obj_joints = {"__error__": repr(exc)}

        # Decode object xyz from every init state to see the spread.
        spread = {}
        for jname, (qadr, _) in fj.items():
            xyz = init_states[:, 1 + qadr: 1 + qadr + 3]
            spread[jname] = {
                "qpos_addr": qadr,
                "state_slice": [1 + qadr, 1 + qadr + 7],
                "mean": np.round(xyz.mean(axis=0), 4).tolist(),
                "min": np.round(xyz.min(axis=0), 4).tolist(),
                "max": np.round(xyz.max(axis=0), 4).tolist(),
                "range": np.round(xyz.max(axis=0) - xyz.min(axis=0), 4).tolist(),
            }

        entry = {
            "task_id": task_id,
            "name": task.name,
            "language": task.language,
            "bddl_file": task.bddl_file,
            "init_states_shape": list(init_states.shape),
            "nq": nq,
            "nv": nv,
            "expected_state_len": 1 + nq + nv,
            "object_joints": obj_joints,
            "free_joints": spread,
        }
        report["tasks"].append(entry)

        print(f"[{task_id}] {task.language}")
        print(f"     state={init_states.shape} nq={nq} nv={nv} (1+nq+nv={1+nq+nv})")
        print(f"     objects={list(obj_joints)}")
        for jname, s in spread.items():
            print(f"       {jname:<34} adr={s['qpos_addr']:<4} "
                  f"mean={s['mean']} range={s['range']}")
        sys.stdout.flush()
        env.close()

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
