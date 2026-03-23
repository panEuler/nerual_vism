from __future__ import annotations

import torch


def smooth_delta(phi: torch.Tensor, eps: float) -> torch.Tensor:
    return eps / (torch.pi * (eps**2 + phi**2))


def area_loss(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 0.1,
) -> torch.Tensor:
    """Toy surface-area surrogate on masked queries.

    Shapes:
    - pred_sdf: [Q]
    - query_points: [Q, 3]
    - mask: [Q] or None
    - grads: [Q, 3]

    We compute gradients on all queries first, then reduce only over the selected mask so
    autograd still has a valid graph into the original query_points tensor.
    """
    if mask is not None and not torch.any(mask):
        return pred_sdf.new_zeros(())

    grads = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=query_points,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    integrand = smooth_delta(pred_sdf, eps) * grads.norm(dim=-1)  # [Q]
    if mask is not None:
        integrand = integrand[mask]  # [Qm]
    return integrand.mean()
