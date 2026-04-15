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


def _masked_monte_carlo_integral(
    integrand: torch.Tensor,
    domain_volume: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if integrand.ndim == 1:
        integrand = integrand.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)

    if mask is None:
        mask = torch.ones_like(integrand, dtype=torch.bool)
    if not torch.any(mask):
        return integrand.new_zeros(())

    if domain_volume is None:
        domain_volume = torch.ones((integrand.shape[0],), dtype=integrand.dtype, device=integrand.device)
    else:
        domain_volume = torch.as_tensor(domain_volume, dtype=integrand.dtype, device=integrand.device).reshape(-1)

    masked_integrand = integrand * mask.to(integrand.dtype)
    counts = mask.sum(dim=-1).clamp_min(1).to(integrand.dtype)
    per_sample_means = masked_integrand.sum(dim=-1) / counts
    per_sample_integrals = per_sample_means * domain_volume
    valid_samples = (mask.sum(dim=-1) > 0).to(integrand.dtype)
    return (per_sample_integrals * valid_samples).sum() / valid_samples.sum().clamp_min(1.0)


def _mean_curvature_from_level_set(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    query_grads: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    grads = _safe_query_grads(pred_sdf, query_points) if query_grads is None else query_grads
    grad_norm = _stable_grad_norm(grads)
    unit_normals = grads / grad_norm.unsqueeze(-1)

    divergence = torch.zeros_like(pred_sdf)
    for axis in range(query_points.shape[-1]):
        component = unit_normals[..., axis]
        component_grad = torch.autograd.grad(
            outputs=component,
            inputs=query_points,
            grad_outputs=torch.ones_like(component),
            create_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        if component_grad is None:
            continue
        divergence = divergence + component_grad[..., axis]
    mean_curvature = 0.5 * divergence
    return mean_curvature, grad_norm


def area_loss(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 0.1,
    query_grads: torch.Tensor | None = None,
    domain_volume: torch.Tensor | None = None,
) -> torch.Tensor:
    """Toy surface-area surrogate on masked batched queries.

    Shapes:
    - pred_sdf: [Q] or [B, Q]
    - query_points: [Q, 3] or [B, Q, 3]
    - mask: [Q] or [B, Q] or None
    """
    grads = _safe_query_grads(pred_sdf, query_points) if query_grads is None else query_grads
    integrand = smooth_delta(pred_sdf, eps) * _stable_grad_norm(grads)
    if domain_volume is None:
        if mask is not None and not torch.any(mask):
            return pred_sdf.new_zeros(())
        if mask is not None:
            integrand = integrand[mask]
        return integrand.mean()
    return _masked_monte_carlo_integral(integrand, domain_volume=domain_volume, mask=mask)


def mean_curvature_integral(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 0.1,
    query_grads: torch.Tensor | None = None,
    domain_volume: torch.Tensor | None = None,
) -> torch.Tensor:
    if domain_volume is None:
        raise ValueError("mean_curvature_integral requires domain_volume for physical Monte Carlo normalization")
    mean_curvature, grad_norm = _mean_curvature_from_level_set(
        pred_sdf,
        query_points,
        query_grads=query_grads,
    )
    integrand = smooth_delta(pred_sdf, eps) * grad_norm * mean_curvature
    return _masked_monte_carlo_integral(integrand, domain_volume=domain_volume, mask=mask)
