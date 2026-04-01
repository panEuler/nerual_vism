from __future__ import annotations

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - environment fallback
    torch = None
    DataLoader = None

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset
from biomol_surface_unsup.losses.loss_builder import build_loss_fn
from biomol_surface_unsup.models.surface_model import SurfaceModel
from biomol_surface_unsup.training.optimizer import build_optimizer
from biomol_surface_unsup.training.train_step import train_step
from biomol_surface_unsup.utils.config import normalize_loss_config


class Trainer:
    def __init__(self, cfg):
        if torch is None or DataLoader is None:
            raise RuntimeError("torch is required to run Trainer in this environment")

        self.cfg = cfg
        requested_device = str(cfg["train"].get("device", "cpu"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device

        data_cfg = cfg["data"]
        train_cfg = cfg["train"]
        batch_size = int(train_cfg.get("batch_size", 1))
        dataset_num_samples = int(data_cfg.get("num_samples", max(batch_size, 2)))
        self.train_dataset = MoleculeDataset(
            root=data_cfg.get("root", "data/processed"),
            split=data_cfg.get("train_split", "train"),
            num_samples=dataset_num_samples,
            num_atoms=int(data_cfg.get("num_atoms", 4)),
            num_query_points=int(data_cfg.get("num_query_points", 32)),
            bbox_padding=float(data_cfg.get("bbox_padding", 2.0)),
            containment_jitter=float(data_cfg.get("containment_jitter", 0.15)),
            surface_band_width=float(
                data_cfg.get("surface_band_width", data_cfg.get("surface_bandwidth", 0.25))
            ),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=collate_fn,
        )

        self.model = SurfaceModel(num_atom_types=self.train_dataset.num_atom_types).to(self.device)
        loss_runtime_cfg = dict(cfg)
        loss_runtime_cfg["loss"] = normalize_loss_config(cfg.get("loss", {}))
        self.loss_fn = build_loss_fn(loss_runtime_cfg)
        self.optimizer = build_optimizer(
            self.model,
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        )

    def train(self):
        num_epochs = int(self.cfg["train"].get("epochs", 1))
        for epoch in range(num_epochs):
            for step, batch in enumerate(self.train_loader):
                metrics = train_step(self.model, batch, self.loss_fn, self.optimizer, self.device)
                print(f"epoch={epoch} step={step} metrics={metrics}")

    def evaluate(self):
        print("TODO")
