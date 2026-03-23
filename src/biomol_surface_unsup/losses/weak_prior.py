from __future__ import annotations

import torch

from biomol_surface_unsup.geometry.sdf_ops import atomic_union_field


def weak_prior_loss(
    coords: torch.Tensor,
    radii: torch.Tensor,
    query_points: torch.Tensor,
    pred_sdf: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Toy weak prior against the atomic-union proxy.

    Shapes:
    - coords: [N, 3]
    - radii: [N]
    - query_points: [Q, 3]
    - pred_sdf: [Q]
    - mask: [Q] or None

    The intended scope is the surface-band group, where this toy proxy is most aligned
    with the sampled narrow band. TODO: replace with a stronger geometry prior once the
    non-toy surface objective is available.
    """
    if mask is not None:
        if not torch.any(mask):
            return pred_sdf.new_zeros(())
        query_points = query_points[mask]  # [Qm, 3]
        pred_sdf = pred_sdf[mask]  # [Qm]

    target = atomic_union_field(coords, radii, query_points)  # [Qm] or [Q]
    return (pred_sdf - target).abs().mean()
