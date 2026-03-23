from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None


QUERY_GROUP_GLOBAL = 0
QUERY_GROUP_CONTAINMENT = 1
QUERY_GROUP_SURFACE_BAND = 2


def _compute_bbox(coords: torch.Tensor, radii: torch.Tensor | None, padding: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bbox bounds with shape [3]."""
    if radii is not None:
        lower = (coords - radii.unsqueeze(-1)).amin(dim=0) - padding
        upper = (coords + radii.unsqueeze(-1)).amax(dim=0) + padding
    else:
        lower = coords.amin(dim=0) - padding
        upper = coords.amax(dim=0) + padding
    return lower, upper


def approximate_atomic_union_sdf(coords: torch.Tensor, radii: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    """Toy atomic-union SDF approximation with shape [Q].

    TODO: this is still a toy proxy for the molecular surface. It approximates the
    union-of-spheres field using per-atom Euclidean distance minus atom radius.
    """
    pairwise_dist = torch.cdist(query_points, coords)  # [Q, N]
    return (pairwise_dist - radii.unsqueeze(0)).amin(dim=1)  # [Q]


def sample_query_points(
    coords: Any,
    num_query_points: int,
    padding: float,
    radii: Any | None = None,
    containment_jitter: float = 0.15,
    surface_band_width: float = 0.25,
) -> dict[str, Any]:
    """Create hierarchical toy query samples.

    Shapes for torch path:
    - coords: [N, 3]
    - radii: [N]
    - query_points: [Q, 3]
    - query_group: [Q]
    - containment_points: [C, 3]

    Sampling groups:
    - global: bbox-uniform samples
    - containment: atom-centered / near-atom anchors
    - surface-band: samples near a toy atomic-union narrow band
    """
    if torch is None or not isinstance(coords, torch.Tensor):
        raise RuntimeError("sample_query_points requires torch in the current toy training path")

    if num_query_points <= 0:
        raise ValueError("num_query_points must be positive")
    if radii is None:
        radii = torch.full((coords.shape[0],), 1.0, dtype=coords.dtype, device=coords.device)

    num_global = max(1, num_query_points // 2)
    num_containment = max(1, num_query_points // 4)
    num_surface = max(1, num_query_points - num_global - num_containment)
    total = num_global + num_containment + num_surface
    if total != num_query_points:
        num_surface += num_query_points - total

    lower, upper = _compute_bbox(coords, radii, padding)

    # [Qg, 3]
    global_samples = lower.unsqueeze(0) + torch.rand(
        num_global,
        3,
        dtype=coords.dtype,
        device=coords.device,
    ) * (upper - lower).unsqueeze(0)

    atom_index = torch.arange(num_containment, device=coords.device) % coords.shape[0]
    base_centers = coords[atom_index]  # [Qc, 3]
    jitter_dir = torch.randn(num_containment, 3, dtype=coords.dtype, device=coords.device)
    jitter_dir = jitter_dir / jitter_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    jitter_scale = (radii[atom_index] * containment_jitter).unsqueeze(-1)
    containment_points = base_centers + jitter_dir * jitter_scale  # [Qc, 3]

    candidate_count = max(num_surface * 8, num_surface)
    candidate_points = lower.unsqueeze(0) + torch.rand(
        candidate_count,
        3,
        dtype=coords.dtype,
        device=coords.device,
    ) * (upper - lower).unsqueeze(0)  # [K, 3]
    candidate_field = approximate_atomic_union_sdf(coords, radii, candidate_points)  # [K]
    near_surface_mask = candidate_field.abs() <= surface_band_width
    if int(near_surface_mask.sum().item()) >= num_surface:
        surface_samples = candidate_points[near_surface_mask][:num_surface]  # [Qs, 3]
    else:
        sort_index = candidate_field.abs().argsort()
        surface_samples = candidate_points[sort_index[:num_surface]]  # [Qs, 3]

    query_points = torch.cat([global_samples, containment_points, surface_samples], dim=0)  # [Q, 3]
    query_group = torch.cat(
        [
            torch.full((num_global,), QUERY_GROUP_GLOBAL, dtype=torch.long, device=coords.device),
            torch.full((num_containment,), QUERY_GROUP_CONTAINMENT, dtype=torch.long, device=coords.device),
            torch.full((num_surface,), QUERY_GROUP_SURFACE_BAND, dtype=torch.long, device=coords.device),
        ],
        dim=0,
    )  # [Q]

    return {
        "query_points": query_points,
        "query_group": query_group,
        "containment_points": containment_points,
        "sampling_counts": {
            "global": int(num_global),
            "containment": int(num_containment),
            "surface_band": int(num_surface),
        },
    }


def sample_surface_band_points(coords: Any, num_points: int, radii: Any, padding: float = 0.0):
    """Return only the toy near-surface-band samples with shape [Q, 3]."""
    sampling = sample_query_points(
        coords=coords,
        num_query_points=num_points,
        padding=padding,
        radii=radii,
    )
    mask = sampling["query_group"] == QUERY_GROUP_SURFACE_BAND
    return sampling["query_points"][mask]
