from __future__ import annotations

import torch

from .volume import smooth_heaviside


def lj_body_integral(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    coords: torch.Tensor,
    epsilon_lj: torch.Tensor,
    sigma_lj: torch.Tensor,
    atom_mask: torch.Tensor,
    mask: torch.Tensor | None = None,
    rho_0: float = 0.0334,
    eps_h: float = 0.1,
    dist_eps: float = 1.5,
    potential_clip: float = 100.0,
) -> torch.Tensor:
    if pred_sdf.ndim == 1:
        pred_sdf = pred_sdf.unsqueeze(0)
        query_points = query_points.unsqueeze(0)
        coords = coords.unsqueeze(0)
        epsilon_lj = epsilon_lj.unsqueeze(0)
        sigma_lj = sigma_lj.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)

    dists = torch.cdist(query_points, coords).clamp_min(float(dist_eps))
    sigma_r6 = (sigma_lj.unsqueeze(1) / dists).pow(6)
    potential = 4.0 * epsilon_lj.unsqueeze(1) * (sigma_r6.pow(2) - sigma_r6)
    potential = potential.clamp(min=-float(potential_clip), max=float(potential_clip))
    lj_energy = (potential * atom_mask.unsqueeze(1).to(potential.dtype)).sum(dim=-1)
    exterior = smooth_heaviside(pred_sdf, eps_h)
    integrand = lj_energy * exterior
    if mask is not None:
        if not torch.any(mask):
            return pred_sdf.new_zeros(())
        integrand = integrand[mask]
    return pred_sdf.new_tensor(float(rho_0)) * integrand.mean()
