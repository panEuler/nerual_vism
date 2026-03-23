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


def _containment_from_model(
    model_out: dict[str, torch.Tensor],
    query_group: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Containment term evaluated on the explicit containment sampling group.

    Shapes:
    - model_out["sdf"]: [Q]
    - query_group: [Q]
    """
    containment_mask = query_group == 1  # [Q]
    if not torch.any(containment_mask):
        return model_out["sdf"].new_zeros(())
    containment_sdf = model_out["sdf"][containment_mask]  # [Qc]
    return containment_loss(containment_sdf, margin=margin)


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
    containment_margin = float(loss_cfg.get("containment_margin", 0.5))

    def loss_fn(batch: dict[str, torch.Tensor], model_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]  # [N, 3]
        radii = batch["radii"]  # [N]
        query_points = batch["query_points"]  # [Q, 3]
        query_group = batch["query_group"]  # [Q]
        pred_sdf = model_out["sdf"]  # [Q]

        losses = {
            "area": area_loss(pred_sdf, query_points, eps=delta_eps),
            "volume": volume_loss(pred_sdf, target_volume_fraction=target_volume_fraction, eps=heaviside_eps),
            "prior": weak_prior_loss(coords, radii, query_points, pred_sdf),
            "eikonal": _eikonal_loss(pred_sdf, query_points),
            "containment": _containment_from_model(model_out, query_group, margin=containment_margin),
        }
        losses["target_sdf"] = atomic_union_field(coords, radii, query_points).detach().mean()
        losses["containment_count"] = (query_group == 1).sum().to(pred_sdf.dtype)
        losses["global_count"] = (query_group == 0).sum().to(pred_sdf.dtype)
        losses["surface_band_count"] = (query_group == 2).sum().to(pred_sdf.dtype)
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
