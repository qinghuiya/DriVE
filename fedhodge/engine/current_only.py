"""Scale the current real parameter displacement without retaining history."""
from dataclasses import dataclass
import math

import torch


@dataclass
class CurrentOnlyResult:
    update: torch.Tensor | None
    metrics: dict


class CurrentOnlyController:
    def __init__(self, *, scale: float = 2.):
        self.scale = float(scale)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('current_only.scale must be finite and positive')
        self.rounds = 0

    @torch.no_grad()
    def observe(self, raw_update: torch.Tensor) -> CurrentOnlyResult:
        if raw_update.ndim != 1 or not raw_update.is_floating_point() or not bool(torch.isfinite(raw_update).all()):
            raise ValueError('Expected finite floating 1D real FL displacement')
        self.rounds += 1
        # The real FedAvg endpoint is already committed. Add only its extra part.
        correction = raw_update * (self.scale-1.) if self.scale != 1. else None
        if correction is not None and not bool(torch.isfinite(correction).all()):
            raise FloatingPointError('Nonfinite current-displacement correction')
        return CurrentOnlyResult(correction, {
            'current_only_round': self.rounds,
            'current_only_scale': self.scale,
            'current_only_applied': int(correction is not None),
        })
