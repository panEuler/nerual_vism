from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.sampling import (
    QUERY_GROUP_CONTAINMENT,
    QUERY_GROUP_GLOBAL,
    QUERY_GROUP_SURFACE_BAND,
)
from biomol_surface_unsup.losses.containment import containment_loss
from biomol_surface_unsup.losses.loss_builder import build_loss, build_loss_fn
from biomol_surface_unsup.training.train_step import train_step
from biomol_surface_unsup.utils.config import normalize_loss_config


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


def _build_batch(query_group: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    query_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.2, 0.1, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    pred_sdf = query_points.pow(2).sum(dim=-1) - 0.2
    batch = {
        "coords": torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32),
        "atom_types": torch.tensor([1, 6], dtype=torch.long),
        "radii": torch.tensor([1.2, 1.3], dtype=torch.float32),
        "query_points": query_points,
        "query_group": query_group,
        "containment_points": query_points[query_group == QUERY_GROUP_CONTAINMENT],
    }
    return batch, pred_sdf


def test_normalize_loss_config_preserves_default_behavior() -> None:
    normalized = normalize_loss_config(
        {
            "lambda_area": 1.0,
            "lambda_volume": 0.5,
            "lambda_containment": 2.0,
            "lambda_prior": 0.5,
            "lambda_eikonal": 0.1,
        }
    )

    assert normalized["losses"]["containment"] == {"weight": 2.0, "groups": ["containment"]}
    assert normalized["losses"]["weak_prior"] == {"weight": 0.5, "groups": ["surface_band"]}
    assert normalized["losses"]["area"] == {"weight": 1.0, "groups": ["surface_band"]}
    assert normalized["losses"]["volume"] == {"weight": 0.5, "groups": ["global"]}
    assert normalized["losses"]["eikonal"] == {"weight": 0.1, "groups": ["global", "surface_band"]}


def test_build_loss_fn_returns_weighted_losses_from_default_mapping() -> None:
    batch, pred_sdf = _build_batch(
        torch.tensor(
            [QUERY_GROUP_GLOBAL, QUERY_GROUP_CONTAINMENT, QUERY_GROUP_SURFACE_BAND, QUERY_GROUP_CONTAINMENT],
            dtype=torch.long,
        )
    )
    loss_fn = build_loss_fn(
        {
            "loss": {
                "losses": {
                    "containment": {"weight": 2.0, "groups": ["containment"]},
                    "weak_prior": {"weight": 0.5, "groups": ["surface_band"]},
                    "area": {"weight": 1.0, "groups": ["surface_band"]},
                    "volume": {"weight": 0.5, "groups": ["global"]},
                    "eikonal": {"weight": 0.1, "groups": ["global", "surface_band"]},
                },
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


def test_build_loss_fn_supports_multi_group_union_masks() -> None:
    batch, pred_sdf = _build_batch(
        torch.tensor(
            [QUERY_GROUP_GLOBAL, QUERY_GROUP_CONTAINMENT, QUERY_GROUP_SURFACE_BAND, QUERY_GROUP_CONTAINMENT],
            dtype=torch.long,
        )
    )
    loss_fn = build_loss_fn(
        {
            "loss": {
                "losses": {
                    "containment": {"weight": 1.0, "groups": ["containment", "surface_band"]},
                    "weak_prior": {"weight": 1.0, "groups": ["surface_band"]},
                    "area": {"weight": 1.0, "groups": ["surface_band"]},
                    "volume": {"weight": 1.0, "groups": ["global"]},
                    "eikonal": {"weight": 1.0, "groups": ["global", "surface_band"]},
                }
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})
    assert losses["containment_count"].item() == pytest.approx(3.0)
    assert losses["surface_band_count"].item() == pytest.approx(1.0)
    assert losses["eikonal_count"].item() == pytest.approx(2.0)
    assert losses["containment"].item() >= 0.0
    assert losses["total"].ndim == 0


def test_build_loss_fn_handles_empty_masks_from_configured_groups() -> None:
    batch, pred_sdf = _build_batch(
        torch.tensor(
            [QUERY_GROUP_GLOBAL, QUERY_GROUP_CONTAINMENT, QUERY_GROUP_CONTAINMENT, QUERY_GROUP_GLOBAL],
            dtype=torch.long,
        )
    )
    loss_fn = build_loss_fn(
        {
            "loss": {
                "losses": {
                    "containment": {"weight": 1.0, "groups": []},
                    "weak_prior": {"weight": 1.0, "groups": ["surface_band"]},
                    "area": {"weight": 1.0, "groups": ["surface_band"]},
                    "volume": {"weight": 1.0, "groups": ["global"]},
                    "eikonal": {"weight": 1.0, "groups": []},
                }
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})
    assert losses["area"].item() == pytest.approx(0.0)
    assert losses["weak_prior"].item() == pytest.approx(0.0)
    assert losses["area_count"].item() == pytest.approx(0.0)
    assert losses["weak_prior_count"].item() == pytest.approx(0.0)
    assert losses["containment"].item() == pytest.approx(0.0)
    assert losses["containment_count"].item() == pytest.approx(0.0)
    assert losses["eikonal"].item() == pytest.approx(0.0)
    assert losses["eikonal_count"].item() == pytest.approx(0.0)
    assert losses["volume_count"].item() == pytest.approx(2.0)


def test_train_step_runs_backward_and_optimizer_step_on_toy_batch() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        def forward(self, coords, atom_types, radii, query_points):
            del coords, atom_types, radii
            sdf = self.scale * query_points.pow(2).sum(dim=-1) + self.bias
            return {"sdf": sdf}

    batch, _ = _build_batch(
        torch.tensor(
            [QUERY_GROUP_GLOBAL, QUERY_GROUP_CONTAINMENT, QUERY_GROUP_SURFACE_BAND, QUERY_GROUP_GLOBAL],
            dtype=torch.long,
        )
    )
    loss_fn = build_loss_fn({"loss": {}})
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = model.scale.detach().clone()
    metrics = train_step(model, batch, loss_fn, optimizer, device="cpu")
    after = model.scale.detach().clone()

    assert "total" in metrics
    assert metrics["total"] >= 0.0
    assert not torch.allclose(before, after)
