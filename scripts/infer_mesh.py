"""Inference script: load a trained checkpoint and extract molecular surfaces.
Usage (from project root)
--------------------------
python scripts/infer_mesh.py \\
    --ckpt  outputs/checkpoints/latest.pt \\
    --config configs/experiment/real_schnet_debug.yaml \\
    --processed_sample_dir /path/to/processed/sample \\
    --spacing_angstrom 0.1 \\
    --output_dir outputs/meshes
Optional flags
--------------
--no_mesh       Skip marching-cubes extraction (only dump SDF grid as .npy)
--no_slices     Skip matplotlib SDF slice plots
--batch_size N  Number of query points per forward pass (default 8192)
--device cpu    Force CPU even when CUDA is available
--num_samples N Limit to first N molecules in the split
"""
from __future__ import annotations
import sys
from pathlib import Path
# ── allow running from the repo root without installing the package ──────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from biomol_surface_unsup.datasets.molecule_dataset import ATOM_TYPE_TO_ID, MoleculeDataset
from biomol_surface_unsup.inference import load_processed_molecule
from biomol_surface_unsup.models.surface_model import SurfaceModel
from biomol_surface_unsup.utils.config import load_infer_config
from biomol_surface_unsup.visualization.export_mesh import export_mesh
from biomol_surface_unsup.visualization.plot_slices import plot_slices
# ────────────────────────────────────────────────────────────────────────────
# Helper: build a uniform 3-D query grid around a molecule
# ────────────────────────────────────────────────────────────────────────────
def _make_query_grid(
    coords: torch.Tensor,
    radii: torch.Tensor,
    spacing_angstrom: float,
    padding: float = 4.0,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Return a flat tensor of grid points and the bounding-box metadata.
    Args:
        coords: Atom positions, shape (N, 3).
        radii: Van-der-Waals radii, shape (N,).
        spacing_angstrom: Target physical spacing between adjacent samples.
        padding: Extra Å margin beyond the vdW surface.
    Returns:
        query_flat: Tensor of shape (R³, 3).
        min_xyz:    np.ndarray of shape (3,), world-space lower corner.
        spacing:    np.ndarray of shape (3,), actual Å step per axis.
        grid_shape: Tuple[int, int, int] used to reshape the predicted SDF grid.
    """
    spacing_angstrom = float(spacing_angstrom)
    if spacing_angstrom <= 0.0:
        raise ValueError(f"spacing_angstrom must be positive, got {spacing_angstrom}")

    margin = float(radii.max().item()) + padding
    lo = coords.min(dim=0).values.cpu().numpy() - margin
    hi = coords.max(dim=0).values.cpu().numpy() + margin

    axis_counts = [
        max(2, int(np.ceil((hi[i] - lo[i]) / spacing_angstrom)) + 1)
        for i in range(3)
    ]
    axes = [np.linspace(lo[i], hi[i], axis_counts[i], dtype=np.float32) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid_xyz = np.stack([gx, gy, gz], axis=-1)
    query_flat = torch.tensor(
        grid_xyz.reshape(-1, 3), dtype=torch.float32
    )
    spacing = np.asarray(
        [
            (axes[i][1] - axes[i][0]) if axis_counts[i] > 1 else spacing_angstrom
            for i in range(3)
        ],
        dtype=np.float32,
    )
    return query_flat, lo.astype(np.float32), spacing, tuple(axis_counts)
# ────────────────────────────────────────────────────────────────────────────
# Helper: run the model over a large set of query points in mini-batches
# ────────────────────────────────────────────────────────────────────────────
def _predict_sdf(
    model: SurfaceModel,
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    radii: torch.Tensor,
    query_points: torch.Tensor,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """Run SDF prediction in mini-batches to avoid OOM.
    Args:
        model: Trained SurfaceModel in eval mode.
        coords: (N, 3) atom coordinates on ``device``.
        atom_types: (N,) integer atom-type indices on ``device``.
        radii: (N,) vdW radii on ``device``.
        query_points: (Q, 3) grid points on CPU (moved to device in batches).
        device: Target device.
        batch_size: Max query points per forward pass.
    Returns:
        sdf_values: np.ndarray of shape (Q,).
    """
    results: list[np.ndarray] = []
    total = query_points.shape[0]
    for start in range(0, total, batch_size):
        batch_q = query_points[start : start + batch_size].to(device)
        with torch.no_grad():
            out = model(coords, atom_types, radii, batch_q)
        results.append(out["sdf"].squeeze(-1).cpu().numpy())
    return np.concatenate(results, axis=0)


def _resolve_single_sample_dir(infer_cfg: dict[str, object]) -> Path | None:
    processed_sample_dir = infer_cfg.get("processed_sample_dir")
    if processed_sample_dir:
        return Path(str(processed_sample_dir))

    pdb_file = infer_cfg.get("pdb_file")
    chain_id = infer_cfg.get("chain_id")
    if not pdb_file:
        return None
    if not chain_id:
        raise ValueError("provide --chain_id together with --pdb_file for single-protein inference")

    from preprocess import process_one_pdb

    preprocess_dir = Path(str(infer_cfg.get("preprocess_dir", "outputs/infer_processed")))
    process_one_pdb(str(pdb_file), str(chain_id), str(preprocess_dir))
    return preprocess_dir / Path(str(pdb_file)).stem
# ────────────────────────────────────────────────────────────────────────────
# Helper: marching cubes with graceful fallback
# ────────────────────────────────────────────────────────────────────────────
def _marching_cubes(
    sdf_grid: np.ndarray,
    lo: np.ndarray,
    spacing: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Extract the zero-isosurface from ``sdf_grid``.
    Returns:
        dict with 'verts' (V, 3) and 'faces' (F, 3), or None if no surface.
    """
    try:
        from skimage.measure import marching_cubes as skimage_mc
    except ImportError:
        print(
            "[infer_mesh] WARNING: scikit-image not installed — skipping mesh extraction.\n"
            "             Install with: pip install scikit-image"
        )
        return None
    try:
        verts_vox, faces, _normals, _vals = skimage_mc(sdf_grid, level=0.0)
    except ValueError as exc:
        print(f"[infer_mesh] marching_cubes could not find a surface: {exc}")
        return None
    # Convert voxel indices → real-space Å coordinates
    verts_world = verts_vox * spacing + lo
    return {"verts": verts_world.astype(np.float32), "faces": faces.astype(np.int32)}
# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_infer_config()
    infer_cfg = cfg["infer"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    # ── device ──────────────────────────────────────────────────────────────
    if infer_cfg["device"] is not None:
        device = torch.device(infer_cfg["device"])
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[infer_mesh] using device: {device}")
    # ── input selection ─────────────────────────────────────────────────────
    split = str(infer_cfg["split"])
    single_sample_dir = _resolve_single_sample_dir(infer_cfg)
    dataset = None
    samples: list[dict[str, object]] = []
    if single_sample_dir is not None:
        molecule = load_processed_molecule(single_sample_dir)
        samples = [
            {
                "id": single_sample_dir.name,
                "coords": molecule["coords"],
                "atom_types": molecule["atom_types"],
                "radii": molecule["radii"],
            }
        ]
        print(f"[infer_mesh] single sample mode: {single_sample_dir}")
    else:
        split = infer_cfg["split"]
        num_samples = infer_cfg["num_samples"]  # None means all
        dataset = MoleculeDataset(
            root=data_cfg.get("root", "data/processed"),
            split=split,
            num_samples=num_samples,
            num_atoms=int(data_cfg.get("num_atoms", 4)),
            num_query_points=int(data_cfg.get("num_query_points", 512)),
            bbox_padding=float(data_cfg.get("bbox_padding", 4.0)),
            containment_jitter=float(data_cfg.get("containment_jitter", 0.15)),
            surface_band_width=float(data_cfg.get("surface_bandwidth", 0.5)),
        )
        samples = [dataset[idx] for idx in range(len(dataset))]
        print(f"[infer_mesh] split='{split}', found {len(samples)} samples")
    # ── model ───────────────────────────────────────────────────────────────
    num_atom_types = len(ATOM_TYPE_TO_ID) if dataset is None else dataset.num_atom_types
    model = SurfaceModel.from_config(model_cfg, num_atom_types=num_atom_types)
    model.to(device)
    # ── load checkpoint ─────────────────────────────────────────────────────
    ckpt_path = Path(infer_cfg["ckpt"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Support multiple checkpoint formats:
    #   1) {"model": ..., "epoch": ..., ...} saved by training/checkpoint.py
    #   2) {"model_state_dict": ..., "epoch": ...}
    #   3) raw state_dict
    if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        state_dict = raw_ckpt["model"]
        saved_epoch = raw_ckpt.get("epoch", "?")
        print(f"[infer_mesh] loaded training checkpoint from epoch {saved_epoch}: {ckpt_path}")
    elif isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["model_state_dict"]
        saved_epoch = raw_ckpt.get("epoch", "?")
        print(f"[infer_mesh] loaded checkpoint from epoch {saved_epoch}: {ckpt_path}")
    else:
        state_dict = raw_ckpt
        print(f"[infer_mesh] loaded raw state dict from: {ckpt_path}")
    model.load_state_dict(state_dict)
    model.eval()
    # ── output directory ────────────────────────────────────────────────────
    out_dir = Path(infer_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing_angstrom = float(infer_cfg.get("spacing_angstrom", 0.1))
    batch_size = infer_cfg["batch_size"]
    padding = float(data_cfg.get("bbox_padding", 4.0))
    # ── per-molecule inference loop ─────────────────────────────────────────
    for idx, sample in enumerate(samples):
        mol_id = str(sample.get("id", f"sample_{idx}"))
        print(f"\n[infer_mesh] [{idx+1}/{len(samples)}] molecule: {mol_id}")
        coords = sample["coords"].to(device)       # (N, 3)
        atom_types = sample["atom_types"].to(device)  # (N,)
        radii = sample["radii"].to(device)          # (N,)
        # Build 3-D query grid
        query_flat, lo, spacing, grid_shape = _make_query_grid(
            coords,
            radii,
            spacing_angstrom,
            padding,
        )
        # Predict SDF
        sdf_values = _predict_sdf(
            model, coords, atom_types, radii, query_flat, device, batch_size
        )
        sdf_grid = sdf_values.reshape(grid_shape)
        print(
            f"[infer_mesh]   SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}], "
            f"grid shape: {sdf_grid.shape}, spacing(Å): {tuple(float(v) for v in spacing)}"
        )
        # Save raw SDF grid (.npy)
        npy_path = out_dir / f"{mol_id}_sdf.npy"
        np.save(npy_path, sdf_grid)
        print(f"[infer_mesh]   SDF grid saved → {npy_path}")
        # ── mesh extraction ─────────────────────────────────────────────────
        if infer_cfg["extract_mesh"]:
            mesh = _marching_cubes(sdf_grid, lo, spacing)
            if mesh is not None:
                mesh_path = out_dir / f"{mol_id}_surface.obj"
                export_mesh(mesh, mesh_path)
                print(
                    f"[infer_mesh]   mesh saved → {mesh_path}  "
                    f"({len(mesh['verts'])} verts, {len(mesh['faces'])} faces)"
                )
            else:
                print("[infer_mesh]   (no surface extracted — SDF may not cross zero)")
        # ── SDF slice plots ─────────────────────────────────────────────────
        if infer_cfg["plot_slices"]:
            slice_path = out_dir / f"{mol_id}_slices.png"
            try:
                plot_slices(
                    sdf_grid,
                    output_path=slice_path,
                    axis=2,
                    num_slices=4,
                    molecule_id=mol_id,
                )
                print(f"[infer_mesh]   slices saved → {slice_path}")
            except ImportError as exc:
                print(f"[infer_mesh]   WARNING: {exc}")
    print(f"\n[infer_mesh] Done. All outputs in: {out_dir.resolve()}")
if __name__ == "__main__":
    main()
