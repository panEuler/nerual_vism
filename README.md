# biomol_surface_unsup
Unsupervised neural implicit biomolecular surface learning with VISM-lite objectives.
## Goals
- Input: molecule atoms
- Output: scalar implicit field
- Training: unsupervised / weakly anchored variational objective
## Main commands
- `python scripts/preprocess.py --config configs/data/toy.yaml`
- `python scripts/train.py --config configs/experiment/debug.yaml`
- `python scripts/train.py --config configs/experiment/real_debug.yaml`
- `python scripts/train.py --config configs/experiment/real_schnet_debug.yaml`
- `python scripts/train.py --config configs/experiment/real_schnet_adaptive_debug.yaml`
- `python scripts/evaluate.py --ckpt outputs/checkpoints/latest.pt`
## Inference / mesh extraction
After training, run `infer_mesh.py` to predict the SDF on a dense 3-D grid and
extract the molecular surface via marching cubes:
```bash
# Full pipeline (SDF grid + OBJ mesh + PNG slice plots)
python scripts/infer_mesh.py \
    --ckpt   outputs/checkpoints/latest.pt \
    --config configs/experiment/real_schnet_debug.yaml \
    --split  test \
    --resolution 64 \
    --output_dir outputs/meshes
# Higher resolution mesh (slower, more RAM)
python scripts/infer_mesh.py \
    --ckpt outputs/checkpoints/latest.pt \
    --config configs/experiment/real_schnet_debug.yaml \
    --resolution 128 --output_dir outputs/meshes_hd
# Only predict SDF, skip mesh and slice plots
python scripts/infer_mesh.py \
    --ckpt outputs/checkpoints/latest.pt \
    --config configs/experiment/real_schnet_debug.yaml \
    --no_mesh --no_slices
# Process a single molecule (first one in the test split)
python scripts/infer_mesh.py \
    --ckpt outputs/checkpoints/latest.pt \
    --config configs/experiment/real_schnet_debug.yaml \
    --num_samples 1
```
Outputs per molecule (saved to `--output_dir`, default `outputs/meshes`):
| File | Description |
|------|-------------|
| `<id>_sdf.npy` | Raw SDF volume, shape `(R, R, R)`, float32 |
| `<id>_surface.obj` | Extracted triangle mesh (Wavefront OBJ) |
| `<id>_slices.png` | Four cross-sectional SDF heatmaps with zero-contour |
Optional dependencies:
- `pip install scikit-image` — required for marching-cubes mesh extraction
- `pip install matplotlib`  — required for SDF slice plots