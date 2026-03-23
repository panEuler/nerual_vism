from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset
from biomol_surface_unsup.models.surface_model import SurfaceModel


def test_model_forward():
    dataset = MoleculeDataset(num_query_points=32)
    sample = dataset[0]

    # coords: [N, 3], atom_types: [N], radii: [N], query_points: [Q, 3]
    model = SurfaceModel(num_atom_types=16)
    out = model(
        sample["coords"],
        sample["atom_types"],
        sample["radii"],
        sample["query_points"],
    )
    assert out["sdf"].shape == (32,)
    assert out["features"].shape[0] == 32
    assert out["mask"].shape[0] == 32
