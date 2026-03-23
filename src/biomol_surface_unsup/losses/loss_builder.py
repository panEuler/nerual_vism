from __future__ import annotations

import torch

from biomol_surface_unsup.geometry.sdf_ops import atomic_union_field
from biomol_surface_unsup.losses.area import area_loss
from biomol_surface_unsup.losses.containment import containment_loss
from biomol_surface_unsup.losses.volume import volume_loss
from biomol_surface_unsup.losses.weak_prior import weak_prior_loss


def _eikonal_loss(pred_sdf: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    """Toy-but-real eikonal term computed from autograd.

    Shapes:
    - pred_sdf: [Q]
    - query_points: [Q, 3]
    - grads: [Q, 3]
    """
    grads = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=query_points,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (grads.norm(dim=-1) - 1.0).pow(2).mean()


def _toy_containment_loss(coords: torch.Tensor, radii: torch.Tensor, pred_sdf: torch.Tensor) -> torch.Tensor:
    """Placeholder containment proxy for the toy loop.

    Shapes:
    - coords: [N, 3]
    - radii: [N]
    - pred_sdf: [Q]

    TODO: replace this proxy with a real containment objective that evaluates the
    predicted field at atom centers or other guaranteed-inside anchor points.
    """
    approx_center_sdf = pred_sdf.mean().expand_as(radii)
    return containment_loss(approx_center_sdf)


def build_loss_fn(cfg: dict[str, object]):
    loss_cfg = dict(cfg.get("loss", {}))
    lambda_area = float(loss_cfg.get("lambda_area", 1.0))
    lambda_volume = float(loss_cfg.get("lambda_volume", 0.5))
    lambda_containment = float(loss_cfg.get("lambda_containment", 0.0))
    lambda_prior = float(loss_cfg.get("lambda_prior", 0.5))
    lambda_eikonal = float(loss_cfg.get("lambda_eikonal", 0.1))
    target_volume_fraction = float(loss_cfg.get("target_volume_fraction", 0.5))
    delta_eps = float(loss_cfg.get("delta_eps", 0.1))
    heaviside_eps = float(loss_cfg.get("heaviside_eps", 0.1))

    def loss_fn(batch: dict[str, torch.Tensor], model_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]  # [N, 3]
        radii = batch["radii"]  # [N]
        query_points = batch["query_points"]  # [Q, 3]
        pred_sdf = model_out["sdf"]  # [Q]

        losses = {
            "area": area_loss(pred_sdf, query_points, eps=delta_eps),
            "volume": volume_loss(pred_sdf, target_volume_fraction=target_volume_fraction, eps=heaviside_eps),
            "prior": weak_prior_loss(coords, radii, query_points, pred_sdf),
            "eikonal": _eikonal_loss(pred_sdf, query_points),
            "containment": _toy_containment_loss(coords, radii, pred_sdf),
        }
        losses["target_sdf"] = atomic_union_field(coords, radii, query_points).detach().mean()
        losses["total"] = (
            lambda_area * losses["area"]
            + lambda_volume * losses["volume"]
            + lambda_containment * losses["containment"]
            + lambda_prior * losses["prior"]
            + lambda_eikonal * losses["eikonal"]
        )
        return losses

    return loss_fn


def build_loss(name: str):
    if name != "weak_prior":
        raise ValueError(f"unsupported toy loss: {name}")

    def loss_fn(prediction: dict[str, object], target: dict[str, object]) -> float:
        pred_sdf = float(prediction.get("sdf", 0.0))
        values = list(target.get("values", [0.0]))
        target_value = float(values[0]) if values else 0.0
        return abs(pred_sdf - target_value)

    return loss_fn
