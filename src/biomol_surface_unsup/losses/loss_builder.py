from __future__ import annotations

import torch

from biomol_surface_unsup.datasets.sampling import (
    QUERY_GROUP_CONTAINMENT,
    QUERY_GROUP_GLOBAL,
    QUERY_GROUP_SURFACE_BAND,
)
from biomol_surface_unsup.geometry.sdf_ops import atomic_union_field
from biomol_surface_unsup.losses.area import area_loss
from biomol_surface_unsup.losses.containment import containment_loss
from biomol_surface_unsup.losses.eikonal import eikonal_loss
from biomol_surface_unsup.losses.volume import volume_loss
from biomol_surface_unsup.losses.weak_prior import weak_prior_loss


def _masked_count(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return scalar sample count for a boolean mask.

    Shape:
    - mask: [Q]
    """
    return mask.sum().to(dtype)


def _containment_from_model(
    pred_sdf: torch.Tensor,
    containment_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Containment term evaluated only on the explicit containment group.

    Shapes:
    - pred_sdf: [Q]
    - containment_mask: [Q]
    - pred_sdf[containment_mask]: [Qc]
    """
    if not torch.any(containment_mask):
        return pred_sdf.new_zeros(())
    return containment_loss(pred_sdf[containment_mask], margin=margin)


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

        # Explicit query-group scopes for each objective.
        is_global = query_group == QUERY_GROUP_GLOBAL  # [Q]
        is_containment = query_group == QUERY_GROUP_CONTAINMENT  # [Q]
        is_surface_band = query_group == QUERY_GROUP_SURFACE_BAND  # [Q]
        is_eikonal = is_global | is_surface_band  # [Q]
        # Rationale: keep eikonal on broad-space + near-surface samples. This preserves
        # stable gradient regularization without tying it to containment anchors whose main
        # role is inside/outside supervision. TODO: revisit whether full-query eikonal is
        # preferable once the non-toy objective is introduced.

        losses = {
            "area": area_loss(pred_sdf, query_points, mask=is_surface_band, eps=delta_eps),
            "volume": volume_loss(
                pred_sdf,
                mask=is_global,
                target_volume_fraction=target_volume_fraction,
                eps=heaviside_eps,
            ),
            "weak_prior": weak_prior_loss(coords, radii, query_points, pred_sdf, mask=is_surface_band),
            "eikonal": eikonal_loss(pred_sdf, query_points, mask=is_eikonal),
            "containment": _containment_from_model(pred_sdf, is_containment, margin=containment_margin),
        }
        losses["target_sdf"] = atomic_union_field(coords, radii, query_points).detach().mean()
        losses["global_count"] = _masked_count(is_global, pred_sdf.dtype)
        losses["containment_count"] = _masked_count(is_containment, pred_sdf.dtype)
        losses["surface_band_count"] = _masked_count(is_surface_band, pred_sdf.dtype)
        losses["area_count"] = _masked_count(is_surface_band, pred_sdf.dtype)
        losses["weak_prior_count"] = _masked_count(is_surface_band, pred_sdf.dtype)
        losses["volume_count"] = _masked_count(is_global, pred_sdf.dtype)
        losses["eikonal_count"] = _masked_count(is_eikonal, pred_sdf.dtype)
        losses["total"] = (
            lambda_area * losses["area"]
            + lambda_volume * losses["volume"]
            + lambda_containment * losses["containment"]
            + lambda_prior * losses["weak_prior"]
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
