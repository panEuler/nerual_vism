from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.losses.containment import containment_loss
from biomol_surface_unsup.losses.loss_builder import build_loss, build_loss_fn


def test_build_loss_and_call() -> None:
    loss_fn = build_loss("weak_prior")
    value = loss_fn({"sdf": 1.0}, {"values": [0.0]})
    assert isinstance(value, float)
    assert value == pytest.approx(1.0)


def test_containment_loss_uses_margin_penalty() -> None:
    pred_sdf = torch.tensor([-0.8, -0.6, 0.1], dtype=torch.float32)
    value = containment_loss(pred_sdf, margin=0.5)
    expected = torch.tensor([0.0, 0.0, 0.36], dtype=torch.float32).mean()
    assert torch.allclose(value, expected)


def test_build_loss_fn_returns_weighted_losses() -> None:
    query_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.2, 0.1, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    pred_sdf = query_points.pow(2).sum(dim=-1) - 0.2
    batch = {
        "coords": torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32),
        "radii": torch.tensor([1.2, 1.3], dtype=torch.float32),
        "query_points": query_points,
        "query_group": torch.tensor([0, 1, 2, 1], dtype=torch.long),
        "containment_points": query_points[torch.tensor([1, 3])],
    }
    loss_fn = build_loss_fn(
        {
            "loss": {
                "lambda_area": 1.0,
                "lambda_volume": 0.5,
                "lambda_containment": 2.0,
                "lambda_prior": 0.5,
                "lambda_eikonal": 0.1,
                "containment_margin": 0.5,
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})
    assert {"area", "volume", "containment", "weak_prior", "eikonal", "total"}.issubset(losses)
    assert losses["containment_count"].item() == 2.0
    assert losses["global_count"].item() == 1.0
    assert losses["surface_band_count"].item() == 1.0
    assert losses["weak_prior_count"].item() == 1.0
    assert losses["area_count"].item() == 1.0
    assert losses["volume_count"].item() == 1.0
    assert losses["eikonal_count"].item() == 2.0
    assert losses["containment"].ndim == 0
    assert losses["total"].ndim == 0
    assert float(losses["total"].detach().cpu()) >= 0.0


def test_build_loss_fn_handles_empty_surface_band_masks() -> None:
    query_points = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    pred_sdf = query_points[:, 0] - 0.1
    batch = {
        "coords": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "radii": torch.tensor([1.0], dtype=torch.float32),
        "query_points": query_points,
        "query_group": torch.tensor([0, 1], dtype=torch.long),
        "containment_points": query_points[torch.tensor([1])],
    }
    loss_fn = build_loss_fn({"loss": {}})

    losses = loss_fn(batch, {"sdf": pred_sdf})
    assert losses["area"].item() == pytest.approx(0.0)
    assert losses["weak_prior"].item() == pytest.approx(0.0)
    assert losses["area_count"].item() == pytest.approx(0.0)
    assert losses["weak_prior_count"].item() == pytest.approx(0.0)
    assert losses["volume_count"].item() == pytest.approx(1.0)
    assert losses["eikonal_count"].item() == pytest.approx(1.0)
