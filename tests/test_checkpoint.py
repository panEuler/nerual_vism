from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.training.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"

    original_weight = model.weight.detach().clone()
    saved = save_checkpoint(path, model, optimizer=optimizer, epoch=3, step=42, metrics={"loss": 1.23})
    assert saved.exists()

    with torch.no_grad():
        model.weight.zero_()

    checkpoint = load_checkpoint(path, model, optimizer=optimizer)
    assert checkpoint["epoch"] == 3
    assert checkpoint["step"] == 42
    assert checkpoint["metrics"]["loss"] == pytest.approx(1.23)
    assert torch.allclose(model.weight, original_weight)
