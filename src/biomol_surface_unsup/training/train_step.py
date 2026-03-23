def train_step(model, batch, loss_fn, optimizer, device):
    model.train()
    coords = batch["coords"].to(device)
    atom_types = batch["atom_types"].to(device)
    radii = batch["radii"].to(device)
    query_points = batch["query_points"].to(device).requires_grad_(True)  # [Q, 3]
    query_group = batch["query_group"].to(device)  # [Q]
    containment_points = batch["containment_points"].to(device)  # [C, 3]

    out = model(coords, atom_types, radii, query_points)
    if "sdf" in out and out["sdf"].ndim != 1:
        out["sdf"] = out["sdf"].reshape(-1)
    losses = loss_fn(
        {
            "coords": coords,
            "atom_types": atom_types,
            "radii": radii,
            "query_points": query_points,
            "query_group": query_group,
            "containment_points": containment_points,
        },
        out,
    )
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    # Keep train_step focused on logging/aggregation. Loss scoping and sample selection stay
    # inside the loss builder so the objective/query-group mapping remains explicit in one place.
    metrics = {k: float(v.detach().cpu()) for k, v in losses.items()}
    metrics.update({f"sampling_{k}": float(v) for k, v in batch.get("sampling_counts", {}).items()})
    return metrics
