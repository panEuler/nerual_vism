from __future__ import annotations


class LossWeightScheduler:
    def __init__(self, initial_weights: dict[str, float], final_weights: dict[str, float], warmup_epochs: int) -> None:
        self.initial_weights = {key: float(value) for key, value in initial_weights.items()}
        self.final_weights = {key: float(value) for key, value in final_weights.items()}
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.loss_names = sorted(set(self.initial_weights) | set(self.final_weights))

    def get_weights(self, epoch: int) -> dict[str, float]:
        if self.warmup_epochs == 0:
            return {name: self.final_weights.get(name, self.initial_weights.get(name, 0.0)) for name in self.loss_names}
        alpha = min(max(epoch, 0), self.warmup_epochs) / float(self.warmup_epochs)
        return {
            name: (1.0 - alpha) * self.initial_weights.get(name, 0.0)
            + alpha * self.final_weights.get(name, self.initial_weights.get(name, 0.0))
            for name in self.loss_names
        }
