from __future__ import annotations

import torch
import torch.nn as nn

from biomol_surface_unsup.features.atom_features import AtomFeatureEmbedding


class LocalFeatureBuilder(nn.Module):
    """Build local neighbor features with explicit batch/query/atom masks.

    Batched shapes:
    - coords: [B, N, 3]
    - atom_types: [B, N]
    - radii: [B, N]
    - atom_mask: [B, N]
    - query_points: [B, Q, 3]
    - query_mask: [B, Q]
    - features: [B, Q, K, F]
    - neighbor_mask: [B, Q, K]
    - neighbor_indices: [B, Q, K]
    - neighbor_distances: [B, Q, K]

    Single-sample compatibility:
    - 2D/1D inputs are promoted to batch size 1 internally.
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
        self.feature_dim = 3 + 1 + atom_embed_dim + rbf_dim + 1

    @staticmethod
    def _stable_pairwise_distance(
        query_points: torch.Tensor,
        coords: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        diff = query_points.unsqueeze(2) - coords.unsqueeze(1)
        return torch.sqrt(diff.pow(2).sum(dim=-1) + float(eps))

    def forward(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        radii: torch.Tensor,
        query_points: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        squeeze_batch = coords.ndim == 2
        if squeeze_batch:
            coords = coords.unsqueeze(0)
            atom_types = atom_types.unsqueeze(0)
            radii = radii.unsqueeze(0)
            query_points = query_points.unsqueeze(0)
            atom_mask = None if atom_mask is None else atom_mask.unsqueeze(0)
            query_mask = None if query_mask is None else query_mask.unsqueeze(0)

        batch_size, num_atoms, _ = coords.shape
        _, num_queries, _ = query_points.shape
        if atom_mask is None:
            atom_mask = torch.ones((batch_size, num_atoms), dtype=torch.bool, device=coords.device)
        if query_mask is None:
            query_mask = torch.ones((batch_size, num_queries), dtype=torch.bool, device=query_points.device)

        k = min(self.max_neighbors, num_atoms) if num_atoms > 0 else 0
        if k == 0:
            feature_shape = (batch_size, num_queries, 0, self.feature_dim)
            empty_features = query_points.new_zeros(feature_shape)
            empty_mask = torch.zeros((batch_size, num_queries, 0), dtype=torch.bool, device=query_points.device)
            empty_index = torch.zeros((batch_size, num_queries, 0), dtype=torch.long, device=query_points.device)
            empty_dist = query_points.new_zeros((batch_size, num_queries, 0))
            result = {
                "features": empty_features,
                "mask": empty_mask,
                "neighbor_indices": empty_index,
                "neighbor_distances": empty_dist,
            }
            if squeeze_batch:
                return {k_: v.squeeze(0) for k_, v in result.items()}
            return result

        dists = self._stable_pairwise_distance(query_points, coords)  # [B, Q, N]
        masked_dists = dists.masked_fill(~atom_mask.unsqueeze(1), float("inf"))
        sorted_dists, sorted_indices = torch.topk(masked_dists, k=k, dim=-1, largest=False)

        neighbor_atom_mask = torch.gather(
            atom_mask.unsqueeze(1).expand(-1, num_queries, -1),
            2,
            sorted_indices,
        )
        neighbor_mask = neighbor_atom_mask & (sorted_dists <= self.cutoff) & query_mask.unsqueeze(-1)

        gather_index_xyz = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        gather_coords = coords.unsqueeze(1).expand(-1, num_queries, -1, -1)
        neighbor_coords = torch.gather(gather_coords, 2, gather_index_xyz)

        gather_radii = radii.unsqueeze(1).expand(-1, num_queries, -1)
        neighbor_radii = torch.gather(gather_radii, 2, sorted_indices).unsqueeze(-1)

        gather_atom_types = atom_types.unsqueeze(1).expand(-1, num_queries, -1)
        neighbor_atom_types = torch.gather(gather_atom_types, 2, sorted_indices)
        neighbor_atom_emb = self.atom_embedding(neighbor_atom_types)

        rel_pos = query_points.unsqueeze(2) - neighbor_coords
        rel_dist = sorted_dists.unsqueeze(-1)
        centers = self.rbf_centers.to(query_points.device, query_points.dtype).view(1, 1, 1, -1)
        rbf = torch.exp(-self.rbf_gamma * (rel_dist - centers).pow(2))

        features = torch.cat([rel_pos, neighbor_radii, neighbor_atom_emb, rbf, rel_dist], dim=-1)
        features = features.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)
        safe_indices = sorted_indices.masked_fill(~neighbor_atom_mask, -1)
        safe_dists = sorted_dists.masked_fill(~neighbor_mask, 0.0)

        result = {
            "features": features,
            "mask": neighbor_mask,
            "neighbor_indices": safe_indices,
            "neighbor_distances": safe_dists,
        }
        if squeeze_batch:
            return {k_: v.squeeze(0) for k_, v in result.items()}
        return result


def build_local_features(sample: dict[str, object]) -> dict[str, object]:
    values = list(sample.get("values", []))
    return {"count": len(values), "sum": float(sum(values))}
