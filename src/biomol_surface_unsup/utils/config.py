import argparse

import yaml


DEFAULT_LOSS_GROUPS = {
    "containment": ["containment"],
    "weak_prior": ["surface_band"],
    "area": ["surface_band"],
    "pressure_volume": ["global"],
    "lj_body": ["global"],
    "volume": ["global"],
    "eikonal": ["global", "surface_band"],
}

DEFAULT_LOSS_WEIGHTS = {
    "containment": 0.0,
    "weak_prior": 0.5,
    "area": 1.0,
    "pressure_volume": 0.0,
    "lj_body": 0.0,
    "volume": 0.0,
    "eikonal": 0.5,
}

LEGACY_LOSS_WEIGHT_KEYS = {
    "containment": "lambda_containment",
    "weak_prior": "lambda_prior",
    "area": "lambda_area",
    "pressure_volume": "lambda_volume",
    "lj_body": "lambda_lj",
    "volume": "lambda_volume",
    "eikonal": "lambda_eikonal",
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_loss_config(loss_cfg):
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


def load_infer_config():
    parser = argparse.ArgumentParser(description="Neural-VISM inference / mesh extraction")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment YAML used during training",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to run inference on (default: test)",
    )
    parser.add_argument(
        "--spacing_angstrom",
        type=float,
        default=0.1,
        help="Target physical sampling spacing in Angstrom (default: 0.1)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Legacy fallback grid resolution when spacing-based sampling is disabled",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/meshes",
        help="Directory for mesh + slice outputs (default: outputs/meshes)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Query points per forward pass (default: 8192)",
    )
    parser.add_argument(
        "--block_voxel_size",
        type=int,
        default=64,
        help="Number of voxels per axis in each inference block (default: 64)",
    )
    parser.add_argument(
        "--narrow_band_width",
        type=float,
        default=2.0,
        help="Absolute SDF threshold in Angstrom used to crop a narrow band for mesh extraction (default: 2.0)",
    )
    parser.add_argument(
        "--isosurface_level",
        type=float,
        default=0.0,
        help="Isosurface level used by marching cubes (default: 0.0)",
    )
    parser.add_argument(
        "--min_component_faces",
        type=int,
        default=0,
        help="Discard mesh connected components with fewer faces than this threshold before keeping the largest shell (default: 0)",
    )
    parser.add_argument(
        "--no_mesh",
        action="store_true",
        help="Skip mesh extraction (only predicts SDF grid)",
    )
    parser.add_argument(
        "--no_slices",
        action="store_true",
        help="Skip SDF slice visualization",
    )
    parser.add_argument(
        "--no_narrow_band_crop",
        action="store_true",
        help="Disable narrow-band cropping before marching cubes",
    )
    parser.add_argument(
        "--no_native_ops",
        action="store_true",
        help="Disable optional C++/CUDA native ops and use pure Python fallbacks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device ('cpu' or 'cuda'). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Max number of molecules to process (default: all in split)",
    )
    parser.add_argument(
        "--processed_sample_dir",
        type=str,
        default=None,
        help="Path to one processed sample directory for single-protein inference",
    )
    parser.add_argument(
        "--pdb_file",
        type=str,
        default=None,
        help="Path to one raw PDB file for single-protein inference",
    )
    parser.add_argument(
        "--chain_id",
        type=str,
        default=None,
        help="Chain ID used together with --pdb_file for single-protein inference",
    )
    parser.add_argument(
        "--preprocess_dir",
        type=str,
        default="outputs/infer_processed",
        help="Directory used to store temporary processed files for single-protein inference",
    )

    args = parser.parse_args()
    exp = load_yaml(args.config)
    return {
        "experiment": exp,
        "data": load_yaml(exp["data"]["config"]),
        "model": load_yaml(exp["model"]["config"]),
        "loss": normalize_loss_config(load_yaml(exp["loss"]["config"])),
        "train": load_yaml(exp["train"]["config"]),
        "infer": {
            "ckpt": args.ckpt,
            "split": args.split,
            "spacing_angstrom": args.spacing_angstrom,
            "resolution": args.resolution,
            "output_dir": args.output_dir,
            "batch_size": args.batch_size,
            "block_voxel_size": args.block_voxel_size,
            "narrow_band_width": args.narrow_band_width,
            "isosurface_level": args.isosurface_level,
            "min_component_faces": args.min_component_faces,
            "extract_mesh": not args.no_mesh,
            "plot_slices": not args.no_slices,
            "narrow_band_crop": not args.no_narrow_band_crop,
            "use_native_ops": not args.no_native_ops,
            "device": args.device,
            "num_samples": args.num_samples,
            "processed_sample_dir": args.processed_sample_dir,
            "pdb_file": args.pdb_file,
            "chain_id": args.chain_id,
            "preprocess_dir": args.preprocess_dir,
        },
    }
