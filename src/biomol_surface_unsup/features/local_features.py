from __future__ import annotations

import torch
import torch.nn as nn

from biomol_surface_unsup.features.atom_features import AtomFeatureEmbedding


class LocalFeatureBuilder(nn.Module):
    """Build toy local neighbor features for each query point.

    Shapes:
    - coords: [N, 3]
    - atom_types: [N]
    - radii: [N]
    - query_points: [Q, 3]
    - features: [Q, K, F]
    - mask: [Q, K]
    - neighbor_indices: [Q, K]
    - neighbor_distances: [Q, K]
    """

    def __init__(
        self,
        num_atom_types: int,
        atom_embed_dim: int,
        rbf_dim: int,
        cutoff: float,
        max_neighbors: int,
    ) -> None:
        super().__init__()
        self.atom_embedding = AtomFeatureEmbedding(num_atom_types, atom_embed_dim)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.rbf_centers = nn.Parameter(torch.linspace(0.0, self.cutoff, rbf_dim), requires_grad=False)
        gamma = 1.0 / max(self.cutoff / max(rbf_dim, 1), 1e-6) ** 2
        self.rbf_gamma = float(gamma)
        # feature = relative xyz [3] + radius [1] + atom embedding [E] + distance RBF [R] + distance scalar [1]
        self.feature_dim = 3 + 1 + atom_embed_dim + rbf_dim + 1

    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        radii: torch.Tensor,
        query_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_queries = query_points.shape[0]
        num_atoms = coords.shape[0]
        k = min(self.max_neighbors, num_atoms)

        # dists: [Q, N]
        dists = torch.cdist(query_points, coords)
        sorted_dists, sorted_indices = torch.topk(dists, k=k, dim=-1, largest=False)

        # mask: [Q, K]
        mask = (sorted_dists <= self.cutoff).to(query_points.dtype)
        if num_atoms > 0:
            # Keep at least one neighbor per query in the toy path so the encoder sees a valid token.
            mask[:, 0] = 1.0

        flat_indices = sorted_indices.reshape(-1)
        neighbor_coords = coords.index_select(0, flat_indices).reshape(num_queries, k, 3)  # [Q, K, 3]
        neighbor_radii = radii.index_select(0, flat_indices).reshape(num_queries, k, 1)  # [Q, K, 1]
        neighbor_atom_types = atom_types.index_select(0, flat_indices).reshape(num_queries, k)  # [Q, K]
        neighbor_atom_emb = self.atom_embedding(neighbor_atom_types)  # [Q, K, E]

        # rel_pos: [Q, K, 3], rel_dist: [Q, K, 1]
        rel_pos = query_points.unsqueeze(1) - neighbor_coords
        rel_dist = sorted_dists.unsqueeze(-1)

        centers = self.rbf_centers.to(query_points.device, query_points.dtype).view(1, 1, -1)
        rbf = torch.exp(-self.rbf_gamma * (rel_dist - centers).pow(2))  # [Q, K, R]

        features = torch.cat([rel_pos, neighbor_radii, neighbor_atom_emb, rbf, rel_dist], dim=-1)  # [Q, K, F]
        features = features * mask.unsqueeze(-1)

        return {
            "features": features,
            "mask": mask,
            "neighbor_indices": sorted_indices,
            "neighbor_distances": sorted_dists,
        }


def build_local_features(sample: dict[str, object]) -> dict[str, object]:
    values = list(sample.get("values", []))
    return {"count": len(values), "sum": float(sum(values))}
