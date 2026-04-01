import argparse

import yaml


DEFAULT_LOSS_GROUPS = {
    "containment": ["containment"],
    "weak_prior": ["surface_band"],
    "area": ["surface_band"],
    "pressure_volume": ["global"],
    "volume": ["global"],
    "eikonal": ["global", "surface_band"],
}

DEFAULT_LOSS_WEIGHTS = {
    "containment": 0.0,
    "weak_prior": 0.5,
    "area": 1.0,
    "pressure_volume": 0.0,
    "volume": 0.0,
    "eikonal": 0.5,
}

LEGACY_LOSS_WEIGHT_KEYS = {
    "containment": "lambda_containment",
    "weak_prior": "lambda_prior",
    "area": "lambda_area",
    "pressure_volume": "lambda_volume",
    "volume": "lambda_volume",
    "eikonal": "lambda_eikonal",
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_loss_config(loss_cfg):
    """Normalize loss config to a single config-driven schema.

    Output format:
    {
        "losses": {
            <loss_name>: {
                "weight": float,
                "groups": [<group_name>, ...],
            }
        },
        ... shared scalar hyperparameters ...
    }

    The legacy lambda_* keys are still accepted so old experiment configs keep
    working, but the runtime always reads the normalized `losses` structure.
    """
    normalized = dict(loss_cfg or {})
    configured_losses = normalized.get("losses", {}) or {}
    losses = {}
    for loss_name, default_groups in DEFAULT_LOSS_GROUPS.items():
        legacy_weight_key = LEGACY_LOSS_WEIGHT_KEYS[loss_name]
        raw_entry = configured_losses.get(loss_name, {}) or {}
        groups = raw_entry.get("groups", default_groups)
        if isinstance(groups, str):
            groups = [groups]
        weight = raw_entry.get("weight", normalized.get(legacy_weight_key, DEFAULT_LOSS_WEIGHTS[loss_name]))
        losses[loss_name] = {
            "weight": float(weight),
            "groups": list(groups),
        }
    normalized["losses"] = losses
    return normalized


def load_experiment_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    exp = load_yaml(args.config)
    return {
        "experiment": exp,
        "data": load_yaml(exp["data"]["config"]),
        "model": load_yaml(exp["model"]["config"]),
        "loss": normalize_loss_config(load_yaml(exp["loss"]["config"])),
        "train": load_yaml(exp["train"]["config"]),
    }


def load_eval_config():
    return load_experiment_config()
