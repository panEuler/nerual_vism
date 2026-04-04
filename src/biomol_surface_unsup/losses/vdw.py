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
    """Compute the Lennard-Jones (LJ) body integral over the exterior solvent region.

    Shapes:
    - pred_sdf: [Q] or [B, Q] (predicted SDF fields for query points)
    - query_points: [Q, 3] or [B, Q, 3] (3D coordinates of query probes)
    - coords: [N, 3] or [B, N, 3] (3D coordinates of all atoms)
    - epsilon_lj: [N] or [B, N] (LJ epsilon parameter per atom)
    - sigma_lj: [N] or [B, N] (LJ sigma parameter per atom)
    - atom_mask: [N] or [B, N] (mask for valid atoms in padded batches)
    - mask: [Q] or [B, Q] or None (mask for valid query points)
    - returns: [], a single scalar Tensor representing the averaged loss.
    """
    # Handle single sample inputs by adding a batch dimension
    if pred_sdf.ndim == 1:
        pred_sdf = pred_sdf.unsqueeze(0)
        query_points = query_points.unsqueeze(0)
        coords = coords.unsqueeze(0)
        epsilon_lj = epsilon_lj.unsqueeze(0)
        sigma_lj = sigma_lj.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)

    # 1. Compute pairwise distances between space probes and atom centers.
    # Clip the minimum distance to 'dist_eps' to prevent division-by-zero singularities.
    dists = torch.cdist(query_points, coords).clamp_min(float(dist_eps))
    
    # 2. Compute the (sigma / r)^6 term for standard L-J 12-6 potential.
    sigma_r6 = (sigma_lj.unsqueeze(1) / dists).pow(6)
    
    # 3. Compute the full L-J potential: V(r) = 4 * epsilon * ((sigma/r)^12 - (sigma/r)^6)
    potential = 4.0 * epsilon_lj.unsqueeze(1) * (sigma_r6.pow(2) - sigma_r6)
    
    # Clip extreme positive/negative potentials to avoid exploding gradients.
    potential = potential.clamp(min=-float(potential_clip), max=float(potential_clip))
    
    # 4. Sum the potential contributions from all valid atoms for each spatial probe.
    lj_energy = (potential * atom_mask.unsqueeze(1).to(potential.dtype)).sum(dim=-1)
    
    # 5. Extract the exterior solvent region mask using a smooth Heaviside step function.
    # (Inside protein: SDF < 0 -> 0; Outside protein: SDF > 0 -> 1)
    exterior = smooth_heaviside(pred_sdf, eps_h)
    
    # 6. Only integrate the L-J potential over the valid exterior solvent region.
    integrand = lj_energy * exterior
    
    # 7. Discard padded dummy space queries safely.
    if mask is not None:
        if not torch.any(mask):
            return pred_sdf.new_zeros(())
        integrand = integrand[mask]
        
    # 8. Scale the volumetric Monte Carlo integral by the bulk solvent density 'rho_0'
    return pred_sdf.new_tensor(float(rho_0)) * integrand.mean()
