import argparse
import yaml
DEFAULT_LOSS_WEIGHTS = {
    "area": 1.0,
    "volume": 0.0,
    "pressure_volume": 1.0,
    "containment": 2.0,
    "weak_prior": 0.5,
    "eikonal": 0.5,
    "lj": 0.0,
}
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def _apply_loss_defaults(loss_cfg):
    merged = dict(loss_cfg or {})
    merged.setdefault("lambda_area", DEFAULT_LOSS_WEIGHTS["area"])
    merged.setdefault("lambda_volume", DEFAULT_LOSS_WEIGHTS["volume"])
    merged.setdefault("lambda_pressure_volume", DEFAULT_LOSS_WEIGHTS["pressure_volume"])
    merged.setdefault("lambda_containment", DEFAULT_LOSS_WEIGHTS["containment"])
    merged.setdefault("lambda_prior", DEFAULT_LOSS_WEIGHTS["weak_prior"])
    merged.setdefault("lambda_eikonal", DEFAULT_LOSS_WEIGHTS["eikonal"])
    merged.setdefault("lambda_lj", DEFAULT_LOSS_WEIGHTS["lj"])
    return merged
def load_experiment_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    exp = load_yaml(args.config)
    loss_cfg = _apply_loss_defaults(load_yaml(exp["loss"]["config"]))
    return {
        "experiment": exp,
        "data": load_yaml(exp["data"]["config"]),
        "model": load_yaml(exp["model"]["config"]),
        "loss": loss_cfg,
        "train": load_yaml(exp["train"]["config"]),
    }
def load_eval_config():
    return load_experiment_config()
def load_infer_config():
    """Parse CLI arguments for the inference script.
    Accepted arguments
    ------------------
    --ckpt   : path to a saved checkpoint (.pt file). Required.
    --config : path to the experiment YAML that was used for training. Required.
    --split  : dataset split to run inference on (default: "test").
    --spacing_angstrom : target physical grid spacing in Å (default: 0.1).
    --resolution : legacy fallback grid resolution per axis.
    --output_dir : directory for mesh / slice outputs (default: outputs/meshes).
    --batch_size : number of query points per forward pass (default: 8192).
    --no_mesh : skip mesh extraction (only predict SDF).
    --no_slices : skip SDF slice visualization.
    --device : force device ("cpu" or "cuda"). Auto-detects if omitted.
    --num_samples : max number of molecules to process (default: all).
    Returns
    -------
    dict with keys: "experiment", "data", "model", "loss", "train", "infer"
    where "infer" contains the inference-specific settings.
    """
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
    args = parser.parse_args()
    exp = load_yaml(args.config)
    loss_cfg = _apply_loss_defaults(load_yaml(exp["loss"]["config"]))
    cfg = {
        "experiment": exp,
        "data": load_yaml(exp["data"]["config"]),
        "model": load_yaml(exp["model"]["config"]),
        "loss": loss_cfg,
        "train": load_yaml(exp["train"]["config"]),
        "infer": {
            "ckpt": args.ckpt,
            "split": args.split,
            "spacing_angstrom": args.spacing_angstrom,
            "resolution": args.resolution,
            "output_dir": args.output_dir,
            "batch_size": args.batch_size,
            "extract_mesh": not args.no_mesh,
            "plot_slices": not args.no_slices,
            "device": args.device,
            "num_samples": args.num_samples,
            "sample_id": args.sample_id,
        },
    }
    return cfg
