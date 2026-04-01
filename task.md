# Neural-VISM MVP Task Tracker

## Step 1: Task 0.1 — Fix GlobalEncoder (translation invariance)
- [ ] Modify `global_features.py`: use center-of-mass-relative coords
- [ ] Add translation invariance test in `test_model_forward.py`
- [ ] Run `pytest -q`

## Step 2: Task 0.2 — Eikonal weight ↑ + L1
- [ ] Modify `eikonal.py`: `.pow(2)` → `.abs()`
- [ ] Modify `configs/loss/vism_lite.yaml`: weight 0.1 → 0.5
- [ ] Modify `utils/config.py`: default weight 0.1 → 0.5
- [ ] Update `test_losses.py` for L1 expectation
- [ ] Run `pytest -q`

## Step 3: Task 0.3 — Replace volume loss with pressure-volume
- [ ] Create `losses/pressure_volume.py`
- [ ] Update `loss_builder.py` to support both volume types
- [ ] Update `config.py` defaults
- [ ] Update `configs/loss/vism_lite.yaml`
- [ ] Add test for pressure_volume_loss
- [ ] Run `pytest -q`

## Step 4: Task 0.4 — Interstitial containment sampling
- [ ] Modify `sampling.py`: add bond midpoint + convex hull containment
- [ ] Update `test_sampling.py`
- [ ] Run `pytest -q`

## MVP Checkpoint (after Step 4)
- [ ] Run smoke test
- [ ] Run 2-epoch toy training

## Step 5: Task 1.2 — Fourier positional encoding
- [ ] Create `models/positional_encoding.py`
- [ ] Modify `surface_model.py` to include Fourier PE
- [ ] Update `SDFDecoder` in_dim
- [ ] Update tests
- [ ] Run `pytest -q`

## Step 6: Task 1.4 — Loss weight annealing
- [ ] Create `training/loss_scheduler.py`
- [ ] Refactor `loss_builder.py` to accept dynamic weights
- [ ] Update `trainer.py` to use scheduler
- [ ] Update config schema
- [ ] Add tests
- [ ] Run `pytest -q`

## Step 7: Task 1.1 — LJ body integral
- [ ] Create `losses/lj_body.py`
- [ ] Add LJ params to toy dataset + collate
- [ ] Wire into `loss_builder.py`
- [ ] Update config
- [ ] Add tests
- [ ] Run `pytest -q`

## VISM-lite Checkpoint (after Step 7)
- [ ] Full smoke test with LJ
- [ ] Two-phase training validation
