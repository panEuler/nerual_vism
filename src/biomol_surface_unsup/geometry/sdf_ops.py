import torch

from biomol_surface_unsup.utils.pairwise import chunked_smooth_atomic_union_field

def sphere_sdf(query_points: torch.Tensor, center: torch.Tensor, radius: torch.Tensor):
    return (query_points - center).norm(dim=-1) - radius

def smooth_min(x: torch.Tensor, dim: int = -1, temperature: float = 10.0):
    return -torch.logsumexp(-temperature * x, dim=dim) / temperature

def atomic_union_field(coords: torch.Tensor, radii: torch.Tensor, query_points: torch.Tensor):
    return chunked_smooth_atomic_union_field(coords, radii, query_points)
