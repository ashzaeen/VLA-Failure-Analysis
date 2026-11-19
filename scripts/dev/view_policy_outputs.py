from pathlib import Path
import json
import numpy as np
import plotly.graph_objects as go


def load_policy_outputs(output_dir, steps=None):
    """
    从 policy_outputs/sample_xxxxxx/response.json 中读取:
    - state
    - action

    返回:
        states: [T, D]
        actions: [T, A]
    """
    output_dir = Path(output_dir)
    sample_dirs = sorted(output_dir.glob("sample_*"))

    states = []
    actions = []

    for idx, sample_dir in enumerate(sample_dirs):
        print(f"process: {sample_dir}")

        json_path = sample_dir / "response.json"
        if not json_path.exists():
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        if "state" in data and data["state"] is not None:
            states.append(data["state"])

        if "action" in data and data["action"] is not None:
            if steps is None or steps <= 1:
                actions.append(data["action"][0])
            elif idx % steps == 0:
                actions += data["action"][:steps]

    states = np.array(states, dtype=float)
    actions = np.array(actions, dtype=float)

    return states, actions


def reconstruct_trajectory_from_actions(initial_state, actions, state_dim=None):
    """
    用初始 state + action 增量累积生成新轨迹。

    action 格式假设:
    - action[0:3] = dx, dy, dz
    - action[3:6] = dRx, dRy, dRz
    - action[6]   = grip command
    """
    initial_state = np.array(initial_state, dtype=float)
    actions = np.array(actions, dtype=float)

    if state_dim is None:
        state_dim = len(initial_state)

    pred_states = np.zeros((len(actions) + 1, state_dim), dtype=float)
    pred_states[0] = initial_state.copy()

    for t in range(len(actions)):
        pred_states[t + 1] = pred_states[t].copy()

        # 累积 xyz
        if actions.shape[1] >= 3 and state_dim >= 3:
            pred_states[t + 1, :3] = pred_states[t, :3] + actions[t, :3]

        # 累积 rotation
        if actions.shape[1] >= 6 and state_dim >= 6:
            pred_states[t + 1, 3:6] = pred_states[t, 3:6] + actions[t, 3:6]

        # grip 不是增量，直接记录当前 action 的 grip command
        if actions.shape[1] >= 7 and state_dim >= 7:
            pred_states[t + 1, 6] = actions[t, 6]

    return pred_states


def make_grip_for_states(actions, states_len):
    """
    actions 长度是 T，predicted_states 长度通常是 T+1。
    为了让每个轨迹点都有颜色，给初始点补一个 grip。
    """
    if actions is None:
        return None

    actions = np.array(actions, dtype=float)

    if actions.ndim != 2 or actions.shape[1] < 7:
        return None

    grip = actions[:, 6]

    if len(grip) == 0:
        return None

    grip_for_states = np.concatenate([[grip[0]], grip])

    if len(grip_for_states) < states_len:
        pad_len = states_len - len(grip_for_states)
        grip_for_states = np.concatenate(
            [grip_for_states, np.full(pad_len, grip_for_states[-1])]
        )
    else:
        grip_for_states = grip_for_states[:states_len]

    return grip_for_states


def make_state_lastdim_for_colors(states):
    """
    用 state 的最后一维作为颜色值。
    """
    if states is None:
        return None

    states = np.array(states, dtype=float)

    if states.ndim != 2 or states.shape[1] < 1:
        return None

    return states[:, -1]


def make_hover_text(states, color_values=None, color_name=None, name="Trajectory"):
    texts = []

    for i in range(len(states)):
        base = (
            f"{name}<br>"
            f"step={i}<br>"
            f"x={states[i, 0]:.5f}<br>"
            f"y={states[i, 1]:.5f}<br>"
            f"z={states[i, 2]:.5f}"
        )

        if color_values is not None and color_name is not None:
            base += f"<br>{color_name}={color_values[i]:.5f}"

        texts.append(base)

    return texts


def save_3d_trajectory_html(
    states,
    predicted_states,
    actions,
    save_path="trajectory_3d.html",
    title="State Trajectory vs Generated Action Rollout",
):
    """
    生成可交互 3D 轨迹 HTML。

    包含:
    1. State Trajectory: response.json 里的 state
       - marker 颜色根据 state 最后一维
    2. Generated Action Rollout: 用 action 累积出来的轨迹
       - marker 颜色根据 action[:, 6] 的 grip
    """
    states = np.array(states, dtype=float)
    predicted_states = np.array(predicted_states, dtype=float)
    actions = np.array(actions, dtype=float)

    min_len = min(len(states), len(predicted_states))

    states = states[:min_len]
    predicted_states = predicted_states[:min_len]

    grip_for_states = make_grip_for_states(actions, states_len=len(predicted_states))
    if grip_for_states is not None:
        grip_for_states = grip_for_states[:len(predicted_states)]

    state_lastdim = make_state_lastdim_for_colors(states)

    fig = go.Figure()

    # 1) 真实 state 轨迹：按最后一维着色
    fig.add_trace(
        go.Scatter3d(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            mode="lines+markers",
            name="State Trajectory | color = state[-1]",
            marker=dict(
                size=5,
                color=state_lastdim,
                colorscale="Cividis",
                showscale=True,
                colorbar=dict(
                    title="State Last Dim",
                    x=1.02,
                ),
            ),
            line=dict(width=5),
            text=make_hover_text(
                states,
                color_values=state_lastdim,
                color_name="state[-1]",
                name="State Trajectory",
            ),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # 2) action rollout：按 grip 着色
    fig.add_trace(
        go.Scatter3d(
            x=predicted_states[:, 0],
            y=predicted_states[:, 1],
            z=predicted_states[:, 2],
            mode="lines+markers",
            name="Generated Action Rollout | color = grip",
            marker=dict(
                size=5,
                color=grip_for_states,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Grip",
                    x=1.14,
                ),
            ),
            line=dict(width=5),
            text=make_hover_text(
                predicted_states,
                color_values=grip_for_states,
                color_name="grip",
                name="Generated Action Rollout",
            ),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # 起点
    fig.add_trace(
        go.Scatter3d(
            x=[states[0, 0]],
            y=[states[0, 1]],
            z=[states[0, 2]],
            mode="markers",
            name="Start",
            marker=dict(size=9, symbol="circle"),
            text=["Start"],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # state 终点
    state_end_text = "State End"
    if state_lastdim is not None:
        state_end_text += f"<br>state[-1]={state_lastdim[-1]:.5f}"

    fig.add_trace(
        go.Scatter3d(
            x=[states[-1, 0]],
            y=[states[-1, 1]],
            z=[states[-1, 2]],
            mode="markers",
            name="State End",
            marker=dict(size=9, symbol="diamond"),
            text=[state_end_text],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # generated action 终点
    generated_end_text = "Generated Action End"
    if grip_for_states is not None:
        generated_end_text += f"<br>grip={grip_for_states[-1]:.5f}"

    fig.add_trace(
        go.Scatter3d(
            x=[predicted_states[-1, 0]],
            y=[predicted_states[-1, 1]],
            z=[predicted_states[-1, 2]],
            mode="markers",
            name="Generated Action End",
            marker=dict(size=9, symbol="square"),
            text=[generated_end_text],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(
            x=0.02,
            y=0.98,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))

    print(f"Saved interactive 3D trajectory to: {save_path}")
    return fig


def visualize_generated_trajectory_3d_html(
    output_dir="./policy_outputs",
    steps=None,
    save_path="trajectory_3d.html",
):
    states, actions = load_policy_outputs(
        output_dir=output_dir,
        steps=steps,
    )

    print("states shape:", states.shape)
    print("actions shape:", actions.shape)

    if len(states) == 0:
        raise ValueError("No states found. Please check response.json state field.")

    if len(actions) == 0:
        raise ValueError("No actions found. Please check response.json action field.")

    predicted_states = reconstruct_trajectory_from_actions(
        initial_state=states[0],
        actions=actions,
        state_dim=states.shape[1],
    )

    fig = save_3d_trajectory_html(
        states=states,
        predicted_states=predicted_states,
        actions=actions,
        save_path=save_path,
        title="State Trajectory (color=state[-1]) vs Generated Action Rollout (color=grip)",
    )

    return states, actions, predicted_states, fig


if __name__ == "__main__":
    steps = 50
    
    model_steps ='5_10000'
    states, actions, predicted_states, fig = visualize_generated_trajectory_3d_html(
        output_dir=f"./outputs/rvl/policy_outputs_{model_steps}/datas",
        steps=steps,
        save_path=f"./outputs/rvl/policy_outputs_{model_steps}/trajectory_3d_{model_steps}_{steps}steps.html",
    )