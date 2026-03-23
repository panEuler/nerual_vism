from __future__ import annotations

import torch

from .area import _safe_query_grads


def eikonal_loss(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-based eikonal penalty on masked query groups.

    Shapes:
    - pred_sdf: [Q]
    - query_points: [Q, 3]
    - mask: [Q] or None
    - grads: [Q, 3]

    The implementation keeps the gradient computation over the full query tensor and only
    masks the per-sample penalties afterward, which avoids breaking the autograd graph.
    """
    if mask is not None and not torch.any(mask):
        return pred_sdf.new_zeros(())

    grads = _safe_query_grads(pred_sdf, query_points)
    penalty = (grads.norm(dim=-1) - 1.0).pow(2)  # [Q]
    if mask is not None:
        penalty = penalty[mask]  # [Qm]
    return penalty.mean()
