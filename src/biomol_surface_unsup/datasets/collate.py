from __future__ import annotations

import torch


def _pad_tensor_1d(values: list[torch.Tensor], target_len: int, pad_value: int | float) -> torch.Tensor:
    padded = values[0].new_full((len(values), target_len), pad_value)
    for idx, value in enumerate(values):
        padded[idx, : value.shape[0]] = value
    return padded


def _pad_tensor_2d(values: list[torch.Tensor], target_len: int) -> torch.Tensor:
    feature_dim = values[0].shape[-1]
    padded = values[0].new_zeros((len(values), target_len, feature_dim))
    for idx, value in enumerate(values):
        padded[idx, : value.shape[0]] = value
    return padded


def collate_fn(batch):
    """Pad the toy samples into explicit batched tensors.

    Shapes:
    - coords: [B, N_max, 3]
    - atom_types: [B, N_max]
    - radii: [B, N_max]
    - atom_mask: [B, N_max]
    - query_points: [B, Q_max, 3]
    - query_group: [B, Q_max]
    - query_mask: [B, Q_max]
    - containment_points: [B, C_max, 3]
    - containment_mask: [B, C_max]
    """
    if not batch:
        raise ValueError("batch must not be empty")

    atom_counts = [sample["coords"].shape[0] for sample in batch]
    query_counts = [sample["query_points"].shape[0] for sample in batch]
    containment_counts = [sample["containment_points"].shape[0] for sample in batch]
    max_atoms = max(atom_counts)
    max_queries = max(query_counts)
    max_containment = max(containment_counts)

    coords = _pad_tensor_2d([sample["coords"] for sample in batch], max_atoms)
    atom_types = _pad_tensor_1d([sample["atom_types"] for sample in batch], max_atoms, pad_value=0)
    radii = _pad_tensor_1d([sample["radii"] for sample in batch], max_atoms, pad_value=0.0)
    charges = _pad_tensor_1d([sample["charges"] for sample in batch], max_atoms, pad_value=0.0)
    epsilon = _pad_tensor_1d([sample["epsilon"] for sample in batch], max_atoms, pad_value=0.0)
    sigma = _pad_tensor_1d([sample["sigma"] for sample in batch], max_atoms, pad_value=0.0)
    res_ids = _pad_tensor_1d([sample["res_ids"] for sample in batch], max_atoms, pad_value=0)
    atom_mask = torch.zeros((len(batch), max_atoms), dtype=torch.bool)

    query_points = _pad_tensor_2d([sample["query_points"] for sample in batch], max_queries)
    query_group = _pad_tensor_1d([sample["query_group"] for sample in batch], max_queries, pad_value=0)
    query_mask = torch.zeros((len(batch), max_queries), dtype=torch.bool)

    containment_points = _pad_tensor_2d([sample["containment_points"] for sample in batch], max_containment)
    containment_mask = torch.zeros((len(batch), max_containment), dtype=torch.bool)

    for idx, (num_atoms, num_queries, num_containment) in enumerate(zip(atom_counts, query_counts, containment_counts)):
        atom_mask[idx, :num_atoms] = True
        query_mask[idx, :num_queries] = True
        containment_mask[idx, :num_containment] = True

    sampling_totals = {"global": 0, "containment": 0, "surface_band": 0}
    for sample in batch:
        for key, value in sample.get("sampling_counts", {}).items():
            sampling_totals[key] = sampling_totals.get(key, 0) + int(value)

    return {
        "id": [sample["id"] for sample in batch],
        "coords": coords,
        "atom_types": atom_types,
        "radii": radii,
        "charges": charges,
        "epsilon": epsilon,
        "sigma": sigma,
        "res_ids": res_ids,
        "atom_mask": atom_mask,
        "query_points": query_points,
        "query_group": query_group,
        "query_mask": query_mask,
        "containment_points": containment_points,
        "containment_mask": containment_mask,
        "sampling_counts": sampling_totals,
    }
