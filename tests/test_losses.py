from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.losses.loss_builder import build_loss, build_loss_fn


def test_build_loss_and_call() -> None:
    loss_fn = build_loss("weak_prior")
    value = loss_fn({"sdf": 1.0}, {"values": [0.0]})
    assert isinstance(value, float)
    assert value == pytest.approx(1.0)


def test_build_loss_fn_returns_weighted_losses() -> None:
    query_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    pred_sdf = (query_points.pow(2).sum(dim=-1) + 0.1).requires_grad_(True)
    batch = {
        "coords": torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32),
        "radii": torch.tensor([1.2, 1.3], dtype=torch.float32),
        "query_points": query_points,
    }
    loss_fn = build_loss_fn(
        {
            "loss": {
                "lambda_area": 1.0,
                "lambda_volume": 0.5,
                "lambda_containment": 2.0,
                "lambda_prior": 0.5,
                "lambda_eikonal": 0.1,
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})
    assert {"area", "volume", "containment", "prior", "eikonal", "total"}.issubset(losses)
    assert losses["total"].ndim == 0
    assert float(losses["total"].detach().cpu()) >= 0.0
