from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import ATOM_TYPE_TO_ID, MoleculeDataset


def _write_processed_sample(sample_dir: Path, prefix: str, num_atoms: int) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / f"{prefix}_coords.npy", np.stack([np.arange(num_atoms), np.zeros(num_atoms), np.zeros(num_atoms)], axis=-1))
    atom_types = np.array(["C", "N", "O", "S"][:num_atoms], dtype="<U2")
    np.save(sample_dir / f"{prefix}_atom_types.npy", atom_types)
    np.save(sample_dir / f"{prefix}_radii.npy", np.linspace(1.2, 1.5, num_atoms))
    np.save(sample_dir / f"{prefix}_charges.npy", np.linspace(-0.2, 0.2, num_atoms))
    np.save(sample_dir / f"{prefix}_epsilon.npy", np.linspace(0.1, 0.4, num_atoms))
    np.save(sample_dir / f"{prefix}_sigma.npy", np.linspace(2.0, 2.3, num_atoms))
    np.save(sample_dir / f"{prefix}_res_ids.npy", np.arange(1, num_atoms + 1, dtype=np.int32))
    np.save(sample_dir / f"{prefix}_atom_names.npy", np.array(["CA", "N", "O", "CB"][:num_atoms], dtype="<U4"))
    np.save(sample_dir / f"{prefix}_res_names.npy", np.array(["GLY", "ALA", "SER", "VAL"][:num_atoms], dtype="<U3"))


def test_dataset_reads_processed_sample_with_sampling_metadata(tmp_path: Path) -> None:
    torch.manual_seed(0)
    _write_processed_sample(tmp_path / "1ABC_TEST", "1ABC_TEST_A", num_atoms=4)
    dataset = MoleculeDataset(root=str(tmp_path), num_query_points=8)
    sample = dataset[0]

    assert sample["id"] == "1ABC_TEST"
    assert tuple(sample["coords"].shape) == (4, 3)
    assert tuple(sample["atom_types"].shape) == (4,)
    assert tuple(sample["radii"].shape) == (4,)
    assert tuple(sample["charges"].shape) == (4,)
    assert tuple(sample["epsilon"].shape) == (4,)
    assert tuple(sample["sigma"].shape) == (4,)
    assert tuple(sample["res_ids"].shape) == (4,)
    assert tuple(sample["query_points"].shape) == (8, 3)
    assert tuple(sample["query_group"].shape) == (8,)
    assert tuple(sample["containment_points"].shape) == (2, 3)
    assert sample["atom_types"].tolist() == [
        ATOM_TYPE_TO_ID["C"],
        ATOM_TYPE_TO_ID["N"],
        ATOM_TYPE_TO_ID["O"],
        ATOM_TYPE_TO_ID["S"],
    ]
    assert sample["sampling_counts"] == {"global": 4, "containment": 2, "surface_band": 2}


def test_collate_fn_pads_atoms_and_queries_without_dropping_samples(tmp_path: Path) -> None:
    torch.manual_seed(0)
    _write_processed_sample(tmp_path / "1ABC_TEST", "1ABC_TEST_A", num_atoms=4)
    _write_processed_sample(tmp_path / "2XYZ_TEST", "2XYZ_TEST_A", num_atoms=3)
    dataset = MoleculeDataset(root=str(tmp_path), num_query_points=8)
    batch = collate_fn([dataset[0], dataset[1]])

    assert batch["id"] == ["1ABC_TEST", "2XYZ_TEST"]
    assert tuple(batch["coords"].shape) == (2, 4, 3)
    assert tuple(batch["atom_types"].shape) == (2, 4)
    assert tuple(batch["radii"].shape) == (2, 4)
    assert tuple(batch["charges"].shape) == (2, 4)
    assert tuple(batch["epsilon"].shape) == (2, 4)
    assert tuple(batch["sigma"].shape) == (2, 4)
    assert tuple(batch["res_ids"].shape) == (2, 4)
    assert tuple(batch["atom_mask"].shape) == (2, 4)
    assert tuple(batch["query_points"].shape) == (2, 8, 3)
    assert tuple(batch["query_group"].shape) == (2, 8)
    assert tuple(batch["query_mask"].shape) == (2, 8)
    assert batch["atom_mask"][0].sum().item() == 4
    assert batch["atom_mask"][1].sum().item() == 3
    assert batch["query_mask"][0].sum().item() == 8
    assert batch["query_mask"][1].sum().item() == 8
    assert batch["sampling_counts"] == {"global": 8, "containment": 4, "surface_band": 4}
