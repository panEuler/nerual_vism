from __future__ import annotations

import torch

from .volume import smooth_heaviside


def pressure_volume_loss(
    pred_sdf: torch.Tensor,
    mask: torch.Tensor | None = None,
    pressure: float = 0.01,
    eps: float = 0.1,
) -> torch.Tensor:
    """Linear pressure-volume penalty on the solvent-accessible exterior."""
    exterior = smooth_heaviside(pred_sdf, eps)
    if mask is not None:
        if not torch.any(mask):
            return pred_sdf.new_zeros(())
        exterior = exterior[mask]
    return pred_sdf.new_tensor(float(pressure)) * exterior.mean()
