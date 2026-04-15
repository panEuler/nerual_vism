from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for a folder of processed protein samples")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing processed sample subdirectories")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Experiment YAML used during training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/batch_infer",
        help="Directory for all inference outputs (default: outputs/batch_infer)",
    )
    parser.add_argument(
        "--spacing_angstrom",
        type=float,
        default=0.5,
        help="Physical sampling spacing in Angstrom (default: 0.5)",
    )
    parser.add_argument(
        "--block_voxel_size",
        type=int,
        default=64,
        help="Number of voxels per axis in each inference block (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device ('cpu' or 'cuda'). Auto-detects if omitted.",
    )
    parser.add_argument("--no_mesh", action="store_true", help="Skip mesh extraction")
    parser.add_argument("--no_slices", action="store_true", help="Skip slice plotting")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples whose <id>_sdf.npy already exists in output_dir",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately if one sample inference fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir is not a directory: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"no processed sample subdirectories found under {input_dir}")

    infer_mesh_path = Path(__file__).with_name("infer_mesh.py")
    total = len(sample_dirs)
    failures: list[str] = []

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        sample_id = sample_dir.name
        sdf_path = output_dir / f"{sample_id}_sdf.npy"
        if args.skip_existing and sdf_path.exists():
            print(f"[infer_folder] [{idx}/{total}] skipping {sample_id} (existing: {sdf_path.name})")
            continue

        command = [
            sys.executable,
            str(infer_mesh_path),
            "--ckpt",
            args.ckpt,
            "--config",
            args.config,
            "--processed_sample_dir",
            str(sample_dir),
            "--spacing_angstrom",
            str(args.spacing_angstrom),
            "--block_voxel_size",
            str(args.block_voxel_size),
            "--output_dir",
            str(output_dir),
        ]
        if args.device is not None:
            command.extend(["--device", args.device])
        if args.no_mesh:
            command.append("--no_mesh")
        if args.no_slices:
            command.append("--no_slices")

        print(f"[infer_folder] [{idx}/{total}] running {sample_id}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures.append(sample_id)
            print(f"[infer_folder] [{idx}/{total}] failed: {sample_id} (exit={result.returncode})")
            if args.fail_fast:
                raise SystemExit(result.returncode)

    if failures:
        print(f"[infer_folder] finished with failures: {failures}")
        raise SystemExit(1)

    print(f"[infer_folder] done. Processed {total} samples from {input_dir}")


if __name__ == "__main__":
    main()
