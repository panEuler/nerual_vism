from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - fallback when torch is unavailable
    class Dataset:  # type: ignore[override]
        pass

try:
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None

from biomol_surface_unsup.datasets.sampling import sample_query_points


ATOM_TYPE_TO_ID = {
    "PAD": 0,
    "UNK": 1,
    "H": 2,
    "C": 3,
    "N": 4,
    "O": 5,
    "S": 6,
    "P": 7,
    "F": 8,
    "CL": 9,
    "BR": 10,
    "I": 11,
    "MG": 12,
    "CA": 13,
    "ZN": 14,
    "FE": 15,
}


@dataclass(frozen=True)
class MoleculeRecord:
    sample_id: str
    directory: Path
    prefix: str


def _normalize_atom_type(atom_type: str) -> str:
    token = str(atom_type).strip().upper()
    return token if token else "UNK"


def _encode_atom_types(atom_types: np.ndarray) -> np.ndarray:
    encoded = [ATOM_TYPE_TO_ID.get(_normalize_atom_type(atom_type), ATOM_TYPE_TO_ID["UNK"]) for atom_type in atom_types]
    return np.asarray(encoded, dtype=np.int64)


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def _find_sample_prefix(sample_dir: Path) -> str:
    coords_files = sorted(sample_dir.glob("*_coords.npy"))
    if len(coords_files) != 1:
        raise FileNotFoundError(f"expected exactly one '*_coords.npy' in {sample_dir}, found {len(coords_files)}")
    return coords_files[0].name[: -len("_coords.npy")]


def _required_sample_fields(prefix: str) -> dict[str, str]:
    return {
        "coords": f"{prefix}_coords.npy",
        "atom_types": f"{prefix}_atom_types.npy",
        "radii": f"{prefix}_radii.npy",
        "charges": f"{prefix}_charges.npy",
        "epsilon": f"{prefix}_epsilon.npy",
        "sigma": f"{prefix}_sigma.npy",
        "res_ids": f"{prefix}_res_ids.npy",
        "atom_names": f"{prefix}_atom_names.npy",
        "res_names": f"{prefix}_res_names.npy",
    }


class MoleculeDataset(Dataset):
    """Dataset for processed protein-chain directories under `data/processed`.

    Per-sample tensor shapes before collation:
    - coords: [N, 3]
    - atom_types: [N]
    - radii: [N]
    - charges: [N]
    - epsilon: [N]
    - sigma: [N]
    - res_ids: [N]
    - query_points: [Q, 3]
    - query_group: [Q]
    - containment_points: [C, 3]
    """

    def __init__(
        self,
        root: str = "data/processed",
        split: str = "train",
        num_samples: int | None = None,
        num_atoms: int | None = None,
        num_query_points: int = 512,
        bbox_padding: float = 4.0,
        containment_jitter: float = 0.15,
        surface_band_width: float = 0.25,
    ) -> None:
        del num_atoms  # Real processed samples determine atom count from disk.
        self.root = Path(root)
        self.split = split
        self.num_query_points = int(num_query_points)
        self.bbox_padding = float(bbox_padding)
        self.containment_jitter = float(containment_jitter)
        self.surface_band_width = float(surface_band_width)
        self.num_atom_types = len(ATOM_TYPE_TO_ID)
        self.records = self._discover_records()
        if num_samples is not None:
            self.records = self.records[: int(num_samples)]
        if not self.records:
            raise FileNotFoundError(f"no processed molecule samples found under {self.root}")

    def _discover_records(self) -> list[MoleculeRecord]:
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")

        split_file = self.root / f"{self.split}.txt"
        if split_file.exists():
            sample_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            sample_dirs = [self.root / sample_id for sample_id in sample_ids]
        elif any(self.root.glob("*_coords.npy")):
            sample_dirs = [self.root]
        else:
            sample_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())

        records: list[MoleculeRecord] = []
        for sample_dir in sample_dirs:
            prefix = _find_sample_prefix(sample_dir)
            required = _required_sample_fields(prefix)
            missing = [name for name, filename in required.items() if not (sample_dir / filename).exists()]
            if missing:
                raise FileNotFoundError(f"sample {sample_dir.name} is missing required fields: {', '.join(missing)}")
            records.append(MoleculeRecord(sample_id=sample_dir.name, directory=sample_dir, prefix=prefix))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("MoleculeDataset requires torch in the current training path")

        record = self.records[idx]
        required = _required_sample_fields(record.prefix)
        arrays = {name: _load_npy(record.directory / filename) for name, filename in required.items()}

        coords = torch.as_tensor(arrays["coords"], dtype=torch.float32)
        radii = torch.as_tensor(arrays["radii"], dtype=torch.float32)
        charges = torch.as_tensor(arrays["charges"], dtype=torch.float32)
        epsilon = torch.as_tensor(arrays["epsilon"], dtype=torch.float32)
        sigma = torch.as_tensor(arrays["sigma"], dtype=torch.float32)
        res_ids = torch.as_tensor(arrays["res_ids"], dtype=torch.long)
        atom_types = torch.as_tensor(_encode_atom_types(arrays["atom_types"]), dtype=torch.long)

        sampling = sample_query_points(
            coords=coords,
            num_query_points=self.num_query_points,
            padding=self.bbox_padding,
            radii=radii,
            containment_jitter=self.containment_jitter,
            surface_band_width=self.surface_band_width,
        )

        return {
            "id": record.sample_id,
            "coords": coords,
            "atom_types": atom_types,
            "radii": radii,
            "charges": charges,
            "epsilon": epsilon,
            "sigma": sigma,
            "res_ids": res_ids,
            "atom_names": arrays["atom_names"],
            "res_names": arrays["res_names"],
            "query_points": sampling["query_points"],
            "query_group": sampling["query_group"],
            "containment_points": sampling["containment_points"],
            "sampling_counts": sampling["sampling_counts"],
        }
