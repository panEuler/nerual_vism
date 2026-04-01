def train_step(model, batch, loss_fn, optimizer, device, loss_weights=None):
    model.train()
    coords = batch["coords"].to(device)
    atom_types = batch["atom_types"].to(device)
    radii = batch["radii"].to(device)
    charges = batch["charges"].to(device) if "charges" in batch else None
    epsilon = batch["epsilon"].to(device) if "epsilon" in batch else None
    sigma = batch["sigma"].to(device) if "sigma" in batch else None
    res_ids = batch["res_ids"].to(device) if "res_ids" in batch else None
    atom_mask = batch["atom_mask"].to(device)
    query_points = batch["query_points"].to(device).requires_grad_(True)  # [B, Q, 3]
    query_group = batch["query_group"].to(device)  # [B, Q]
    query_mask = batch["query_mask"].to(device)  # [B, Q]
    containment_points = batch["containment_points"].to(device)  # [B, C, 3]
    containment_mask = batch["containment_mask"].to(device)  # [B, C]

    out = model(
        coords,
        atom_types,
        radii,
        query_points,
        atom_mask=atom_mask,
        query_mask=query_mask,
    )
    losses = loss_fn(
        {
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
        },
        out,
        loss_weights=loss_weights,
    )
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    metrics = {k: float(v.detach().cpu()) for k, v in losses.items()}
    metrics.update({f"sampling_{k}": float(v) for k, v in batch.get("sampling_counts", {}).items()})
    return metrics
