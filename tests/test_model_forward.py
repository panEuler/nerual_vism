from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset
from biomol_surface_unsup.features.global_features import GlobalFeatureEncoder
from biomol_surface_unsup.models.surface_model import SurfaceModel


def test_model_forward_single_sample_keeps_compatibility():
    dataset = MoleculeDataset(num_query_points=32)
    sample = dataset[0]
    expected_neighbors = min(sample["coords"].shape[0], 64)

    model = SurfaceModel(num_atom_types=dataset.num_atom_types)
    out = model(
        sample["coords"],
        sample["atom_types"],
        sample["radii"],
        sample["query_points"],
    )
    assert out["sdf"].shape == (32,)
    assert out["features"].shape[:2] == (32, expected_neighbors)
    assert out["mask"].shape[:2] == (32, expected_neighbors)


def test_model_forward_batched_uses_atom_and_query_masks():
    dataset = MoleculeDataset(num_samples=2, num_query_points=8)
    batch = collate_fn([dataset[0], dataset[1]])
    expected_neighbors = min(batch["coords"].shape[1], 64)

    model = SurfaceModel(num_atom_types=dataset.num_atom_types)
    out = model(
        batch["coords"],
        batch["atom_types"],
        batch["radii"],
        batch["query_points"],
        atom_mask=batch["atom_mask"],
        query_mask=batch["query_mask"],
    )
    assert out["sdf"].shape == (2, 8)
    assert out["features"].shape[:3] == (2, 8, expected_neighbors)
    assert out["mask"].shape == (2, 8, expected_neighbors)
    assert torch.all(out["sdf"][~batch["query_mask"]] == 0.0)
    assert torch.equal(out["mask"], out["mask"] & batch["query_mask"].unsqueeze(-1))


def test_global_feature_encoder_is_translation_invariant_with_atom_mask():
    encoder = GlobalFeatureEncoder(num_atom_types=16, atom_embed_dim=8, hidden_dim=32, out_dim=16)
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    atom_types = torch.tensor([[1, 6, 8, 0]], dtype=torch.long)
    radii = torch.tensor([[1.2, 1.3, 1.1, 0.0]], dtype=torch.float32)
    atom_mask = torch.tensor([[True, True, True, False]])
    shift = torch.tensor([[[3.2, -1.4, 0.7]]], dtype=torch.float32)

    base = encoder(coords, atom_types, radii, atom_mask=atom_mask)
    shifted = encoder(coords + shift, atom_types, radii, atom_mask=atom_mask)

    assert torch.allclose(base, shifted, atol=1e-5, rtol=1e-5)
