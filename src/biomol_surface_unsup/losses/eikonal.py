from __future__ import annotations

import torch

from .area import _safe_query_grads, _stable_grad_norm


def eikonal_loss(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-based eikonal penalty on masked batched query groups."""
    if mask is not None and not torch.any(mask):
        return pred_sdf.new_zeros(())

    grads = _safe_query_grads(pred_sdf, query_points)
    penalty = (_stable_grad_norm(grads) - 1.0).abs()
    if mask is not None:
        penalty = penalty[mask]
    return penalty.mean()
