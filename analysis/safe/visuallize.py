import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math

def plot_score_distribution(res, bins=50, title="Distribution of SAFE scores"):
    """
    res:
        res["max_scores"]: shape [N]
        res["labels"]: shape [N], failure=1, success=0
    """

    max_scores = res["max_scores"]
    labels = res["labels"]

    success_scores = max_scores[labels == 0]
    failure_scores = max_scores[labels == 1]

    plt.figure(figsize=(7, 5))
    plt.hist(success_scores, bins=bins, alpha=0.6, label="Success")
    plt.hist(failure_scores, bins=bins, alpha=0.6, label="Failure")

    plt.xlabel("SAFE Score (lower is safer)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

def compute_valid_max_scores(scores, valid_masks):
    """
    scores:
        shape [N, T]

    valid_masks:
        shape [N, T], 1 表示有效 timestep，0 表示 padding

    return:
        max_scores: shape [N]
        valid_scores: list，每个元素是一条 rollout 的有效 score 序列
    """

    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()

    if hasattr(valid_masks, "detach"):
        valid_masks = valid_masks.detach().cpu().numpy()

    max_scores = []
    valid_scores = []

    for i in range(scores.shape[0]):
        scores_i = scores[i][valid_masks[i] == 1]

        max_score = scores_i.max()
        max_scores.append(max_score)
        valid_scores.append(scores_i)

    max_scores = np.asarray(max_scores)

    return max_scores, valid_scores

def compute_limited_max_scores(scores, rollouts):
    """
    只看所有 rollout 共有的前 min_len 个 timestep。

    scores:
        shape [N, T]

    datas:
        list of rollout data，每个 data 里面有 data["hidden_states"]

    return:
        max_scores: shape [N]
        min_len: int
    """

    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()

    min_len = min(len(rollout.hidden_states) for rollout in rollouts)

    limited_scores = scores[:, :min_len]
    max_scores = limited_scores.max(axis=1)

    return max_scores, min_len


def plot_failure_roc(labels, max_scores, title="Failure Detection ROC Curve", save_path=None):
    """
    labels:
        shape [N], failure=1, success=0

    max_scores:
        shape [N], score 越大越像 failure
    """

    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    if hasattr(max_scores, "detach"):
        max_scores = max_scores.detach().cpu().numpy()

    labels = np.asarray(labels)
    max_scores = np.asarray(max_scores)

    fpr, tpr, thresholds = roc_curve(labels, max_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    print("ROC-AUC:", roc_auc)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": roc_auc,
    }

def plot_youden_j_vs_threshold(labels, scores, alpha=2, save_path=None):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores =  tpr - alpha *fpr

    best_idx = np.argmax(j_scores)

    # sklearn 的第一个 threshold 可能是 inf，所以去掉第一个点更好看
    valid = np.isfinite(thresholds)
    print(f"Best threshold:{best_idx}  {thresholds[best_idx]:.4f}, J={j_scores[best_idx]:.3f}")

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds[valid], j_scores[valid])
    plt.scatter(
        thresholds[best_idx],
        j_scores[best_idx],
        label=f"Best threshold={thresholds[best_idx]:.4f}, J={j_scores[best_idx]:.3f}"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Youden's J score")
    plt.title("Youden's J Score vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()

    return thresholds, j_scores


def plot_safe_scores_subplots(scores,label, n=6, cols=3,save_path=None, ylim=(0, 1)):
    """
    scores: shape [num_rollouts, T] 或 list of arrays
    n: 画前 n 条 rollout
    cols: 每行几个子图
    """
    n = min(n, len(scores))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(n):
        axes[i].plot(scores[i], label=f"Rollout {i}")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("SAFE Score")
        axes[i].set_title(f"SAFE Score - Rollout {i} failed: {label[i]}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        axes[i].set_ylim(ylim)

    # 删除多余空子图
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()