from __future__ import annotations

from pathlib import Path

import torch

from biomol_surface_unsup.datasets.molecule_dataset import (
    _encode_atom_types,
    _find_sample_prefix,
    _load_npy,
    _required_sample_fields,
)


def load_processed_molecule(sample_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load one preprocessed protein sample for point-wise SDF inference."""
    sample_dir = Path(sample_dir)
    prefix = _find_sample_prefix(sample_dir)
    arrays = {
        name: _load_npy(sample_dir / filename)
        for name, filename in _required_sample_fields(prefix).items()
    }
    return {
        "coords": torch.as_tensor(arrays["coords"], dtype=torch.float32),
        "atom_types": torch.as_tensor(_encode_atom_types(arrays["atom_types"]), dtype=torch.long),
        "radii": torch.as_tensor(arrays["radii"], dtype=torch.float32),
        "charges": torch.as_tensor(arrays["charges"], dtype=torch.float32),
        "epsilon": torch.as_tensor(arrays["epsilon"], dtype=torch.float32),
        "sigma": torch.as_tensor(arrays["sigma"], dtype=torch.float32),
    }


def _validate_query_points(query_points: torch.Tensor) -> torch.Tensor:
    if query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError(f"query_points must have shape [Q, 3], got {tuple(query_points.shape)}")
    return query_points


@torch.no_grad()
def predict_sdf(
    model: torch.nn.Module,
    molecule: dict[str, torch.Tensor],
    query_points: torch.Tensor,
    device: str | torch.device = "cpu",
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Predict SDF values for arbitrary xyz query points on one molecule."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    query_points = _validate_query_points(torch.as_tensor(query_points, dtype=torch.float32))
    model = model.to(device)
    model.eval()

    coords = molecule["coords"].to(device)
    atom_types = molecule["atom_types"].to(device)
    radii = molecule["radii"].to(device)
    charges = molecule["charges"].to(device)
    epsilon = molecule["epsilon"].to(device)
    sigma = molecule["sigma"].to(device)

    sdf_chunks = []
    for start in range(0, query_points.shape[0], chunk_size):
        query_chunk = query_points[start : start + chunk_size].to(device)
        out = model(
            coords,
            atom_types,
            radii,
            query_chunk,
            charges=charges,
            epsilon=epsilon,
            sigma=sigma,
        )
        sdf_chunks.append(out["sdf"].detach().cpu())

    if not sdf_chunks:
        return query_points.new_zeros((0,))
    return torch.cat(sdf_chunks, dim=0)
