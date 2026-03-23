from .area import area_loss
from .volume import volume_loss
from .weak_prior import weak_prior_loss
from .eikonal import eikonal_loss


def vism_lite_loss(
    coords,
    radii,
    query_points,
    pred_sdf,
    lambda_area=1.0,
    lambda_volume=0.5,
    lambda_prior=0.5,
    lambda_eikonal=0.1,
):
    """Legacy toy VISM-lite wrapper kept for compatibility.

    Notes:
    - `query_points` should require grad if area/eikonal are enabled.
    - The prior/volume terms are intentionally simple stable surrogates in the toy path.
    """
    losses = {}
    losses["area"] = area_loss(pred_sdf, query_points)
    losses["volume"] = volume_loss(pred_sdf)
    losses["prior"] = weak_prior_loss(coords, radii, query_points, pred_sdf)
    losses["eikonal"] = eikonal_loss(pred_sdf, query_points)
    total = (
        lambda_area * losses["area"]
        + lambda_volume * losses["volume"]
        + lambda_prior * losses["prior"]
        + lambda_eikonal * losses["eikonal"]
    )
    losses["total"] = total
    return losses
