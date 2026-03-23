from __future__ import annotations

import torch


def containment_loss(pred_sdf: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Penalize points that are not sufficiently inside the predicted surface.

    Shapes:
    - pred_sdf: [C]

    We use the convention that inside points should satisfy pred_sdf <= -margin.
    The penalty is relu(pred_sdf + margin)^2, which discourages trivial boundary touching.
    """
    return torch.relu(pred_sdf + margin).pow(2).mean()
