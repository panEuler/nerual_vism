# biomol_surface_unsup
Unsupervised neural implicit biomolecular surface learning with VISM-lite objectives.

## Goals
- Input: molecule atoms
- Output: scalar implicit field
- Training: unsupervised / weakly anchored variational objective

## Main commands
- `python scripts/preprocess.py --config configs/data/toy.yaml`
- `python scripts/train.py --config configs/experiment/debug.yaml`
- `python scripts/train.py --config configs/experiment/my_train.yaml`
- `python scripts/train.py --config configs/experiment/my_train.yaml --data_num_samples 100 --train_epochs 30 --train_batch_size 1 --train_output_dir outputs/checkpoints_fast`
- `python scripts/train.py --config configs/experiment/my_train.yaml --train_resume_from outputs/checkpoints_fast/best_model.pt --data_num_samples 5000 --train_output_dir outputs/checkpoints_stage2`

## Inference / mesh extraction
After training, run `infer_mesh.py` to predict the SDF on a blockwise 3-D grid
and optionally extract the molecular surface via marching cubes:

```bash
# Full pipeline (SDF grid + OBJ mesh + PNG slice plots) for one processed sample
python scripts/infer_mesh.py \
    --ckpt outputs/checkpoints/latest.pt \
    --config configs/experiment/my_train.yaml \
    --processed_sample_dir data/test/1A5Z_ACBD \
    --spacing_angstrom 0.5 \
    --block_voxel_size 64 \
    --output_dir outputs/meshes

# Higher-resolution sampling (slower)
python scripts/infer_mesh.py \
    --ckpt outputs/checkpoints/latest.pt \
    --config configs/experiment/my_train.yaml \
    --processed_sample_dir data/test/1A5Z_ACBD \
    --spacing_angstrom 0.25 \
    --block_voxel_size 64 \
    --output_dir outputs/meshes_hd

# Only predict SDF, skip mesh and slice plots
python scripts/infer_mesh.py \
    --ckpt outputs/checkpoints/latest.pt \
    --config configs/experiment/my_train.yaml \
    --processed_sample_dir data/test/1A5Z_ACBD \
    --no_mesh --no_slices

# Process the first N molecules from a split
python scripts/infer_mesh.py \
    --ckpt outputs/checkpoints/latest.pt \
    --config configs/experiment/my_train.yaml \
    --split test \
    --num_samples 1
```

Outputs per molecule (saved to `--output_dir`, default `outputs/meshes`):

| File | Description |
|------|-------------|
| `<id>_sdf.npy` | Raw SDF volume stored as float32 |
| `<id>_sdf_meta.json` | Grid origin, spacing, and shape metadata |
| `<id>_surface.obj` | Extracted triangle mesh (Wavefront OBJ) |
| `<id>_slices.png` | Four cross-sectional SDF heatmaps with zero-contour |

Optional dependencies:
- `pip install scikit-image` — required for marching-cubes mesh extraction
- `pip install matplotlib`  — required for SDF slice plots
