from __future__ import annotations

import torch

from .area import _masked_monte_carlo_integral
from .volume import smooth_heaviside


def pressure_volume_loss(
    pred_sdf: torch.Tensor,
    mask: torch.Tensor | None = None,
    pressure: float = 0.01,
    eps: float = 0.1,
    domain_volume: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear pressure-volume penalty on the solute cavity (interior)."""
    # SDF < 0 inside the protein, so -pred_sdf > 0 inside.
    interior = smooth_heaviside(-pred_sdf, eps)
    if domain_volume is None:
        if mask is not None:
            if not torch.any(mask):
                return pred_sdf.new_zeros(())
            interior = interior[mask]
        return pred_sdf.new_tensor(float(pressure)) * interior.mean()
    integral = _masked_monte_carlo_integral(interior, domain_volume=domain_volume, mask=mask)
    return pred_sdf.new_tensor(float(pressure)) * integral
