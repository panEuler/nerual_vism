from __future__ import annotations

import torch

from biomol_surface_unsup.datasets.sampling import (
    QUERY_GROUP_CONTAINMENT,
    QUERY_GROUP_GLOBAL,
    QUERY_GROUP_SURFACE_BAND,
)
from biomol_surface_unsup.losses.area import area_loss
from biomol_surface_unsup.losses.containment import containment_loss
from biomol_surface_unsup.losses.eikonal import eikonal_loss
from biomol_surface_unsup.losses.pressure_volume import pressure_volume_loss
from biomol_surface_unsup.losses.volume import volume_loss
from biomol_surface_unsup.losses.weak_prior import weak_prior_loss
from biomol_surface_unsup.utils.config import normalize_loss_config


QUERY_GROUP_IDS = {
    "global": QUERY_GROUP_GLOBAL,
    "containment": QUERY_GROUP_CONTAINMENT,
    "surface_band": QUERY_GROUP_SURFACE_BAND,
}

SUPPORTED_LOSSES = ("containment", "weak_prior", "area", "pressure_volume", "volume", "eikonal")


def _batched_atomic_union_field(coords: torch.Tensor, radii: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    pairwise = torch.cdist(query_points, coords) - radii.unsqueeze(-2)
    return -torch.logsumexp(-10.0 * pairwise, dim=-1) / 10.0


def _masked_count(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return mask.sum().to(dtype)


def _group_mask(query_group: torch.Tensor, query_mask: torch.Tensor, group_names: list[str]) -> torch.Tensor:
    mask = torch.zeros_like(query_group, dtype=torch.bool)
    for group_name in group_names:
        if group_name not in QUERY_GROUP_IDS:
            supported = ", ".join(sorted(QUERY_GROUP_IDS))
            raise ValueError(f"unsupported query group '{group_name}', expected one of: {supported}")
        mask = mask | (query_group == QUERY_GROUP_IDS[group_name])
    return mask & query_mask


def build_loss_fn(cfg: dict[str, object]):
    loss_cfg = normalize_loss_config(dict(cfg.get("loss", {})))
    configured_losses = loss_cfg["losses"]
    target_volume_fraction = float(loss_cfg.get("target_volume_fraction", 0.5))
    delta_eps = float(loss_cfg.get("delta_eps", 0.1))
    heaviside_eps = float(loss_cfg.get("heaviside_eps", 0.1))
    containment_margin = float(loss_cfg.get("containment_margin", 0.5))
    pressure = float(loss_cfg.get("pressure", 0.01))

    def loss_fn(batch: dict[str, torch.Tensor], model_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]  # [B, N, 3]
        radii = batch["radii"]  # [B, N]
        atom_mask = batch["atom_mask"]  # [B, N]
        query_points = batch["query_points"]  # [B, Q, 3]
        query_group = batch["query_group"]  # [B, Q]
        query_mask = batch["query_mask"]  # [B, Q]
        pred_sdf = model_out["sdf"]  # [B, Q]

        if pred_sdf.ndim == 1:
            pred_sdf = pred_sdf.unsqueeze(0)
        if query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)
            query_group = query_group.unsqueeze(0)
            query_mask = query_mask.unsqueeze(0)
            coords = coords.unsqueeze(0)
            radii = radii.unsqueeze(0)
            atom_mask = atom_mask.unsqueeze(0)
        if not query_points.requires_grad:
            query_points = query_points.requires_grad_(True)

        base_masks = {
            "global": _group_mask(query_group, query_mask, ["global"]),
            "containment": _group_mask(query_group, query_mask, ["containment"]),
            "surface_band": _group_mask(query_group, query_mask, ["surface_band"]),
        }
        loss_masks = {
            loss_name: _group_mask(query_group, query_mask, configured_losses[loss_name]["groups"])
            for loss_name in SUPPORTED_LOSSES
        }

        losses = {
            "area": area_loss(pred_sdf, query_points, mask=loss_masks["area"], eps=delta_eps),
            "pressure_volume": pressure_volume_loss(
                pred_sdf,
                mask=loss_masks["pressure_volume"],
                pressure=pressure,
                eps=heaviside_eps,
            ),
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
                atom_mask=atom_mask,
            ),
            "eikonal": eikonal_loss(pred_sdf, query_points, mask=loss_masks["eikonal"]),
            "containment": containment_loss(
                pred_sdf,
                margin=containment_margin,
                mask=loss_masks["containment"],
            ),
        }
        safe_coords = coords.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
        safe_radii = radii.masked_fill(~atom_mask, 0.0)
        target_sdf = _batched_atomic_union_field(safe_coords, safe_radii, query_points).detach()
        losses["target_sdf"] = target_sdf[query_mask].mean() if torch.any(query_mask) else pred_sdf.new_zeros(())
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
