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
from biomol_surface_unsup.utils.config import normalize_loss_config


QUERY_GROUP_IDS = {
    "global": QUERY_GROUP_GLOBAL,
    "containment": QUERY_GROUP_CONTAINMENT,
    "surface_band": QUERY_GROUP_SURFACE_BAND,
}

SUPPORTED_LOSSES = ("containment", "weak_prior", "area", "volume", "eikonal")


def _masked_count(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return scalar sample count for a boolean mask.

    Shape:
    - mask: [Q]
    """
    return mask.sum().to(dtype)


def _group_mask(query_group: torch.Tensor, group_names: list[str]) -> torch.Tensor:
    """Build the union mask for one loss from configured query-group names.

    Shapes:
    - query_group: [Q]
    - return: [Q]

    Notes:
    - Empty `group_names` is allowed and returns an all-false mask so the loss is
      stable for ablations that disable a scope.
    - Multiple group names are combined with boolean OR to form a set union.
    """
    mask = torch.zeros_like(query_group, dtype=torch.bool)
    for group_name in group_names:
        if group_name not in QUERY_GROUP_IDS:
            supported = ", ".join(sorted(QUERY_GROUP_IDS))
            raise ValueError(f"unsupported query group '{group_name}', expected one of: {supported}")
        mask = mask | (query_group == QUERY_GROUP_IDS[group_name])
    return mask


def _containment_from_model(
    pred_sdf: torch.Tensor,
    containment_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Containment term evaluated on the configured containment mask.

    Shapes:
    - pred_sdf: [Q]
    - containment_mask: [Q]
    - pred_sdf[containment_mask]: [Qc]
    """
    if not torch.any(containment_mask):
        return pred_sdf.new_zeros(())
    return containment_loss(pred_sdf[containment_mask], margin=margin)


def build_loss_fn(cfg: dict[str, object]):
    loss_cfg = normalize_loss_config(dict(cfg.get("loss", {})))
    configured_losses = loss_cfg["losses"]
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

        if pred_sdf.ndim != 1:
            pred_sdf = pred_sdf.reshape(-1)
        if not query_points.requires_grad:
            query_points = query_points.requires_grad_(True)

        # Base query-group counts are logged independently of the objective so debug
        # output still shows how many samples came from each sampler bucket.
        base_masks = {
            "global": _group_mask(query_group, ["global"]),
            "containment": _group_mask(query_group, ["containment"]),
            "surface_band": _group_mask(query_group, ["surface_band"]),
        }
        loss_masks = {
            loss_name: _group_mask(query_group, configured_losses[loss_name]["groups"])
            for loss_name in SUPPORTED_LOSSES
        }

        losses = {
            "area": area_loss(pred_sdf, query_points, mask=loss_masks["area"], eps=delta_eps),
            "volume": volume_loss(
                pred_sdf,
                mask=loss_masks["volume"],
                target_volume_fraction=target_volume_fraction,
                eps=heaviside_eps,
            ),
            "weak_prior": weak_prior_loss(
                coords,
                radii,
                query_points,
                pred_sdf,
                mask=loss_masks["weak_prior"],
            ),
            "eikonal": eikonal_loss(pred_sdf, query_points, mask=loss_masks["eikonal"]),
            "containment": _containment_from_model(
                pred_sdf,
                loss_masks["containment"],
                margin=containment_margin,
            ),
        }
        losses["target_sdf"] = atomic_union_field(coords, radii, query_points).detach().mean()
        losses["global_count"] = _masked_count(base_masks["global"], pred_sdf.dtype)
        losses["containment_count"] = _masked_count(base_masks["containment"], pred_sdf.dtype)
        losses["surface_band_count"] = _masked_count(base_masks["surface_band"], pred_sdf.dtype)
        for loss_name in SUPPORTED_LOSSES:
            losses[f"{loss_name}_count"] = _masked_count(loss_masks[loss_name], pred_sdf.dtype)

        total = pred_sdf.new_zeros(())
        for loss_name in SUPPORTED_LOSSES:
            total = total + float(configured_losses[loss_name]["weight"]) * losses[loss_name]
        losses["total"] = total
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
