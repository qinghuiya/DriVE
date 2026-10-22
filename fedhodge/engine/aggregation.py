"""Sample-weighted streaming aggregation of model states."""
import torch
from torch import nn

class StreamingFedAvgAccumulator:
    """FedAvg accumulation without materializing one full endpoint per client."""

    def __init__(self, global_model: nn.Module):
        self.global_state = dict(global_model.state_dict())
        self.parameter_names = {name for name, _ in global_model.named_parameters()}
        self.float_names = [k for k, v in self.global_state.items() if torch.is_floating_point(v)]
        self.nonfloat_names = [k for k, v in self.global_state.items() if not torch.is_floating_point(v)]
        self.sums = {k: torch.zeros_like(self.global_state[k]) for k in self.float_names}
        self.best_nonfloat = {k: torch.empty_like(self.global_state[k]) for k in self.nonfloat_names}
        self.total_weight = 0.0
        self.best_weight = -1.0

    @torch.no_grad()
    def reset(self) -> None:
        for value in self.sums.values():
            value.zero_()
        self.total_weight = 0.0
        self.best_weight = -1.0

    @torch.no_grad()
    def add_state(self, state: dict[str, torch.Tensor], weight: float) -> None:
        w = float(weight)
        for name in self.float_names:
            self.sums[name].add_(state[name], alpha=w)
        if w > self.best_weight:
            for name in self.nonfloat_names:
                self.best_nonfloat[name].copy_(state[name])
            self.best_weight = w
        self.total_weight += w

    @torch.no_grad()
    def finalize_into_global(self, *, need_flat_update: bool) -> torch.Tensor | None:
        if self.total_weight <= 0:
            raise RuntimeError('Cannot finalize an empty FedAvg accumulator.')
        inv_total = 1.0 / self.total_weight
        flat_parts: list[torch.Tensor] = []
        for name in self.float_names:
            avg = self.sums[name]
            avg.mul_(inv_total)
            target = self.global_state[name]
            if need_flat_update and name in self.parameter_names:
                flat_parts.append((avg - target).reshape(-1))
            target.copy_(avg)
        for name in self.nonfloat_names:
            self.global_state[name].copy_(self.best_nonfloat[name])
        if not need_flat_update:
            return None
        return torch.cat(flat_parts, dim=0).detach()
