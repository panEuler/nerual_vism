from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset


def test_dataset_returns_toy_sample_with_sampling_metadata() -> None:
    torch.manual_seed(0)
    dataset = MoleculeDataset(num_query_points=8)
    sample = dataset[0]

    assert sample["id"] == "train-0"
    assert tuple(sample["coords"].shape) == (4, 3)
    assert tuple(sample["atom_types"].shape) == (4,)
    assert tuple(sample["radii"].shape) == (4,)
    assert tuple(sample["query_points"].shape) == (8, 3)
    assert tuple(sample["query_group"].shape) == (8,)
    assert tuple(sample["containment_points"].shape) == (2, 3)
    assert sample["sampling_counts"] == {"global": 4, "containment": 2, "surface_band": 2}
