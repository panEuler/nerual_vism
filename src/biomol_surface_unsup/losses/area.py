from __future__ import annotations

import torch


def smooth_delta(phi: torch.Tensor, eps: float) -> torch.Tensor:
    eps = max(float(eps), 1e-6)
    x = phi / eps
    return torch.where(
        x.abs() <= 1.0,
        0.5 / eps * (1.0 + torch.cos(torch.pi * x)),
        torch.zeros_like(phi),
    )


def _safe_query_grads(pred_sdf: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    """Return d(pred_sdf.sum()) / d(query_points) with batched support.

    Shapes:
    - pred_sdf: [Q] or [B, Q]
    - query_points: [Q, 3] or [B, Q, 3]
    - return: same leading shape as query_points
    """
    grads = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=query_points,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    if grads is None:
        grads = torch.zeros_like(query_points)
    return grads


def _stable_grad_norm(grads: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable gradient norm used inside higher-order losses."""
    return torch.sqrt(grads.pow(2).sum(dim=-1) + float(eps))


def area_loss(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 0.1,
    query_grads: torch.Tensor | None = None,
) -> torch.Tensor:
    """Toy surface-area surrogate on masked batched queries.

    Shapes:
    - pred_sdf: [Q] or [B, Q]
    - query_points: [Q, 3] or [B, Q, 3]
    - mask: [Q] or [B, Q] or None
    """
    if mask is not None and not torch.any(mask):
        return pred_sdf.new_zeros(())

    grads = _safe_query_grads(pred_sdf, query_points) if query_grads is None else query_grads
    integrand = smooth_delta(pred_sdf, eps) * _stable_grad_norm(grads)
    if mask is not None:
        integrand = integrand[mask]
    return integrand.mean()
