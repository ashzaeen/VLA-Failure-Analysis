from dataclasses import dataclass
import torch
from torch.utils.data import Dataset



@dataclass
class SimpleRollout:
    hidden_states: torch.Tensor   # [T, D]
    episode_success: int          # 1 = success, 0 = failure
    success_labels: torch.Tensor | None = None
    task_id: int | None = None

    def __post_init__(self):
        self.episode_success = int(self.episode_success)

class SimpleRolloutDataset(Dataset):
    def __init__(self, rollouts, device="cpu"):
        self.rollouts = rollouts

        self.features, self.valid_masks, self.episode_success_labels, self.success_labels = pad_rollouts(
            rollouts,
            device=device,
        )
        

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "valid_masks": self.valid_masks[idx],
            "episode_success_labels": self.episode_success_labels[idx],
            "success_labels": self.success_labels[idx],
        }
    

def pad_rollouts(rollouts, device="cpu"):
    """
    rollouts: list[SimpleRollout]

    return:
        features: [B, T_max, D]
        valid_masks: [B, T_max]
        success_labels: [B, T_max]
    """
    batch_size = len(rollouts)
    max_len = max(r.hidden_states.shape[0] for r in rollouts)
    hidden_dim = rollouts[0].hidden_states.shape[-1]

    dtype = rollouts[0].hidden_states.dtype

    features = torch.zeros(
        batch_size, max_len, hidden_dim,
        dtype=dtype,
        device=device,
    )

    valid_masks = torch.zeros(
        batch_size, max_len,
        dtype=torch.float32,
        device=device,
    )

    all_success_labels = torch.zeros(
        (batch_size,max_len),    
        dtype=torch.float32,
        device=device,
    )
    success_labels = torch.zeros(
        batch_size,    
        dtype=torch.float32,
        device=device,
    )

    for i, r in enumerate(rollouts):
        T = r.hidden_states.shape[0]

        features[i, :T] = r.hidden_states.to(device)
        valid_masks[i, :T] = 1.0
        if r.success_labels is None:
            all_success_labels[i, :T] = torch.tensor(r.episode_success, dtype=torch.float32, device=device).expand(T)
        else:
            all_success_labels[i,:T] = r.success_labels[:T].to(device)
        success_labels[i] = r.episode_success
    return features, valid_masks, success_labels,all_success_labels

def normalize_rollouts_hidden_states(
    rollouts: list[SimpleRollout],
    mean: torch.Tensor = None,
    std: torch.Tensor = None,
):
    """
    Normalize the hidden states of the rollouts to have zero mean and unit variance.
    This is done in place, modifying the original rollouts.
    """
    # Stack all hidden states into a single tensor
    all_hidden_states = torch.cat([r.hidden_states for r in rollouts], dim=0)
    
    # Compute mean and std
    if mean is None:
        mean = all_hidden_states.mean(dim=0)
    if std is None:
        std = all_hidden_states.std(dim=0)
    
    # Normalize each rollout's hidden states
    for r in rollouts:
        r.hidden_states = (r.hidden_states - mean) / std
    
    return rollouts, mean, std
