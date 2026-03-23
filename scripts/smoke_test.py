from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover - script path
    raise SystemExit(f"smoke_test requires torch: {exc}")

from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset
from biomol_surface_unsup.losses.loss_builder import build_loss_fn
from biomol_surface_unsup.models.surface_model import SurfaceModel


def main() -> int:
    dataset = MoleculeDataset(num_query_points=16)
    sample = dataset[0]

    # coords: [N, 3], atom_types: [N], radii: [N], query_points: [Q, 3]
    assert sample["coords"].shape == (4, 3)
    assert sample["atom_types"].shape == (4,)
    assert sample["radii"].shape == (4,)
    assert sample["query_points"].shape == (16, 3)

    model = SurfaceModel(num_atom_types=16)
    query_points = sample["query_points"].requires_grad_(True)
    output = model(
        sample["coords"],
        sample["atom_types"],
        sample["radii"],
        query_points,
    )
    assert "sdf" in output
    assert output["sdf"].shape == (16,)
    assert output["features"].shape[0] == 16
    assert output["mask"].shape[0] == 16

    loss_fn = build_loss_fn({"loss": {}})
    losses = loss_fn(
        {
            "coords": sample["coords"],
            "atom_types": sample["atom_types"],
            "radii": sample["radii"],
            "query_points": query_points,
        },
        output,
    )
    assert "total" in losses
    assert torch.isfinite(losses["total"])
    print("smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
