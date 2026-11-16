import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def draw_patch_grid(image, grid_size=16, color="white", linewidth=0.5):
    """
    Draw patch grid on image
    """
    h, w = image.shape[:2]

    plt.imshow(image)

    grid_h = h / grid_size
    grid_w = w / grid_size

    for i in range(grid_size + 1):
        plt.axhline(i * grid_h, color=color, linewidth=linewidth)

    for j in range(grid_size + 1):
        plt.axvline(j * grid_w, color=color, linewidth=linewidth)

    plt.axis("off")

def extract_action_image_attention(attentions,
                                   layer_idx=-1,
                                   action_token_idx=-1,
                                   image_token_slice=slice(0,256),
                                   head_idx=None):
    """
    Extract attention from action token to image tokens
    """

    attn = attentions[layer_idx]  # [B, heads, seq, seq]

    if head_idx is None:
        attn = attn.mean(dim=1)
    else:
        attn = attn[:, head_idx]

    attn = attn.squeeze(0)

    action_to_image = attn[action_token_idx, image_token_slice]

    return action_to_image

def attention_to_patch_map(attention_1d, grid_size=16):
    """
    Convert 256 attention vector -> 16x16 patch map
    """

    if isinstance(attention_1d, torch.Tensor):
        attention_1d = attention_1d.detach().cpu().numpy()

    patch_map = attention_1d.reshape(grid_size, grid_size)

    return patch_map

def overlay_attention(image,
                      patch_map,
                      alpha=0.35,
                      cmap="jet"):
    """
    Overlay attention heatmap on image
    """

    H, W = image.shape[:2]

    heatmap = cv2.resize(patch_map, (W, H))

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heatmap_color = plt.get_cmap(cmap)(heatmap)[:, :, :3]

    overlay = (1 - alpha) * image / 255 + alpha * heatmap_color

    return heatmap, overlay

def visualize_attention(image,
                        attentions,
                        action_token_idx=-1,
                        image_token_slice=slice(0,256),
                        layer_idx=-1,
                        head_idx=None):
    """
    Visualize action->image attention
    """

    attn = extract_action_image_attention(
        attentions,
        layer_idx,
        action_token_idx,
        image_token_slice,
        head_idx
    )

    patch_map = attention_to_patch_map(attn)

    heatmap, overlay = overlay_attention(image, patch_map)

    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    draw_patch_grid(image)
    plt.axis("off")

    return patch_map

def compute_attention_metrics(patch_map, top_k=5):

    p = patch_map.flatten()
    p = p / (p.sum() + 1e-12)

    entropy = -(p * np.log(p + 1e-12)).sum()
    sorted_p = np.sort(p)[::-1]
    topk_mass = sorted_p[:top_k].sum()

    n = len(p)
    gini = (np.sum((2*np.arange(1,n+1)-n-1)*np.sort(p))) / (n*np.sum(p))

    return {
        "entropy": float(entropy),
        "gini": float(gini),
        "topk_mass": float(topk_mass)
    }


def visualize_patch_grid(
        image,
        grid_size=16,
        show_index=True,
        line_color="white",
        text_color="yellow",
        linewidth=0.5,
        fontsize=6,
        figsize=(6,6)
    ):
    """
    Draw grid and optional patch index on image.

    Args:
        image: PIL / numpy / torch image
        grid_size: number of patches per side (default 16)
        show_index: whether to display patch index
    """

    # convert torch tensor
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    image = np.array(image)

    H, W = image.shape[:2]

    grid_h = H / grid_size
    grid_w = W / grid_size

    plt.figure(figsize=figsize)
    plt.imshow(image)

    # draw horizontal lines
    for i in range(grid_size + 1):
        plt.axhline(i * grid_h, color=line_color, linewidth=linewidth)

    # draw vertical lines
    for j in range(grid_size + 1):
        plt.axvline(j * grid_w, color=line_color, linewidth=linewidth)

    # draw patch index
    if show_index:
        for i in range(grid_size):
            for j in range(grid_size):

                idx = i * grid_size + j

                plt.text(
                    j * grid_w + grid_w/2,
                    i * grid_h + grid_h/2,
                    str(idx),
                    color=text_color,
                    fontsize=fontsize,
                    ha="center",
                    va="center"
                )

    plt.axis("off")
    plt.show()

def visualize_mean_attention_matrix(attentions, layer_idx=-1, save_path=None, cmap="turbo"):
    """
    Visualize mean attention matrix of one layer.
    
    Args:
        attentions: list of attention tensors, each [B, heads, seq, seq]
        layer_idx: which layer to visualize
        save_path: optional path to save
    """
    mean_attention = attentions[layer_idx].mean(dim=1).squeeze(0).detach().cpu().float().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(mean_attention, cmap=cmap)
    plt.colorbar()
    plt.title(f"Mean Attention Matrix (Layer {layer_idx})")
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()

    return mean_attention