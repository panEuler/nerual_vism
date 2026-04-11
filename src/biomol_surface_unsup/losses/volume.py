from __future__ import annotations

import torch

# volmue项可能暂时用不到
def smooth_heaviside(x: torch.Tensor, eps: float) -> torch.Tensor:
    eps = max(float(eps), 1e-6)
    return 0.5 * (1.0 + (2.0 / torch.pi) * torch.atan(x / eps))


def volume_loss(
    pred_sdf: torch.Tensor,
    mask: torch.Tensor | None = None,
    target_volume_fraction: float = 0.5,
    eps: float = 0.1,
) -> torch.Tensor:
    """Toy volume-fraction penalty on masked batched SDF samples."""
    if mask is not None:
        if not torch.any(mask):
            return pred_sdf.new_zeros(())
        pred_sdf = pred_sdf[mask]

    inside = smooth_heaviside(-pred_sdf, eps).mean()
    target = pred_sdf.new_tensor(float(target_volume_fraction))
    return (inside - target).pow(2)
