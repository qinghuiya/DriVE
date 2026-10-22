from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class DriVEStepResult:
    update: torch.Tensor | None
    metrics: dict[str, float | int | str]


class DriVEController:
    """Native update-only DriVE p0/1/2 on the shared optimized real FL engine.

    Fits a degree-p polynomial to real displacements and integrates it. For p0:

      1. Run exactly one *real* wrapped FL round (handled by FederatedEngine).
      2. Record only that real displacement ``Delta_t`` in a FIFO of length W.
      3. As soon as the FIFO contains W displacements, including the current
         round, use those displacements to form

             v_t = (lambda_v / W) * sum_j Delta_j

         and apply ``v_t`` to the model without selecting/training any client.

    Virtual corrections never enter the history.  History stays on the same
    device as the raw FedAvg displacement, avoiding repeated host/device transfers.
    """

    implementation = 'drive_poly_integral_v1'

    def __init__(self, *, window_size: int = 30, lambda_v: float = 5.0, poly_order: int = 0) -> None:
        if isinstance(poly_order, bool) or int(poly_order) != poly_order or poly_order not in (0, 1, 2):
            raise ValueError('DriVE poly_order must be 0, 1 or 2')
        if isinstance(window_size, bool) or int(window_size) != window_size or window_size < poly_order + 1:
            raise ValueError('DriVE window_size must be an integer >= poly_order+1')
        self.poly_order = int(poly_order)
        self.window_size = int(window_size)
        self.lambda_v = float(lambda_v)
        if self.window_size <= 0:
            raise ValueError("DriVE window_size must be positive.")
        if not math.isfinite(self.lambda_v) or self.lambda_v < 0.0:
            raise ValueError("DriVE lambda_v must be finite and non-negative.")

        # Construct the small system once on CPU, in the source's float64 dtype.
        u = -torch.arange(self.window_size, dtype=torch.float64)
        phi = torch.stack([u.pow(d) for d in range(self.poly_order + 1)], dim=1)
        integral = torch.tensor([self.lambda_v ** (d+1) / (d+1)
                                 for d in range(self.poly_order + 1)], dtype=torch.float64)
        weights = torch.linalg.pinv(phi).T.matmul(integral)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError('DriVE coefficients are non-finite')
        self._weights = tuple(float(w) for w in weights.tolist())
        singular = torch.linalg.svdvals(phi)
        self._condition = float(singular.max() / singular.min())

        self.real_history: list[torch.Tensor] = []
        self.completed_real_rounds = 0
        self.virtual_events = 0

    @staticmethod
    def _norm(x: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(x).item())

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
        denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        if float(denom.item()) <= eps:
            return 0.0
        return float((torch.dot(a, b) / denom).item())

    @torch.no_grad()
    def observe(self, raw_update: torch.Tensor) -> DriVEStepResult:
        if raw_update.ndim != 1 or not raw_update.is_floating_point():
            raise ValueError("DriVE expects a flat floating-point real displacement.")
        if not bool(torch.isfinite(raw_update).all()):
            raise FloatingPointError('Non-finite real DriVE displacement')
        if self.real_history and (raw_update.shape != self.real_history[-1].shape
                                 or raw_update.device != self.real_history[-1].device
                                 or raw_update.dtype != self.real_history[-1].dtype):
            raise ValueError('DriVE displacement shape/device/dtype changed mid-run')

        # IMPORTANT: this is the completed real FedAvg displacement only.  The
        # virtual correction returned below is never fed back into this FIFO.
        self.real_history.append(raw_update.detach().clone())
        if len(self.real_history) > self.window_size:
            self.real_history = self.real_history[-self.window_size :]
        self.completed_real_rounds += 1

        applied = (
            self.lambda_v > 0.0
            and self.completed_real_rounds >= self.window_size
            and len(self.real_history) == self.window_size
        )

        metrics: dict[str, float | int | str] = {
            "drive_enabled": 1,
            "drive_method": f"drive_p{self.poly_order}",
            "drive_implementation": self.implementation,
            "drive_real_round": self.completed_real_rounds,
            "drive_time_index": self.completed_real_rounds - 1,
            "drive_first_virtual_round": self.window_size,
            "drive_warmup_rounds": self.window_size - 1,
            "drive_history_size": len(self.real_history),
            "drive_window_size": self.window_size,
            "drive_lambda_v": self.lambda_v,
            "drive_polynomial_order": self.poly_order,
            "drive_virtual_applied": int(applied),
            "drive_virtual_events": self.virtual_events,
            "drive_history_on_cpu": int(raw_update.device.type == 'cpu'),
            "drive_base_state_frozen_during_virtual": 1,
            "drive_update_poly_weight_norm": math.sqrt(sum(w*w for w in self._weights)),
            "drive_update_poly_condition": self._condition,
        }

        if not applied:
            metrics.update(
                {
                    "drive_virtual_norm": 0.0,
                    "drive_virtual_raw_norm_ratio": 0.0,
                    "drive_virtual_current_cosine": 0.0,
                    "drive_cumulative_virtual_horizon": self.virtual_events * self.lambda_v,
                }
            )
            return DriVEStepResult(None, metrics)

        correction = self._correction(raw_update)
        if not bool(torch.isfinite(correction).all()):
            raise FloatingPointError('Non-finite DriVE virtual displacement')
        # Fractional source treats an exactly zero correction as not applied.
        if self._norm(correction) == 0:
            metrics.update(drive_virtual_applied=0, drive_virtual_norm=0.0,
                           drive_virtual_raw_norm_ratio=0.0, drive_virtual_current_cosine=0.0,
                           drive_cumulative_virtual_horizon=self.virtual_events*self.lambda_v)
            return DriVEStepResult(None, metrics)

        self.virtual_events += 1
        raw_norm = self._norm(raw_update)
        virtual_norm = self._norm(correction)
        metrics.update(
            {
                "drive_virtual_events": self.virtual_events,
                "drive_virtual_norm": virtual_norm,
                "drive_virtual_raw_norm_ratio": virtual_norm / max(raw_norm, 1e-12),
                "drive_virtual_current_cosine": self._cosine(correction, raw_update),
                "drive_cumulative_virtual_horizon": self.virtual_events * self.lambda_v,
            }
        )
        return DriVEStepResult(correction.detach(), metrics)

    def update_only_weights(self):
        return self._weights

    def _correction(self, raw_update):
        correction = torch.zeros_like(raw_update)
        for weight, displacement in zip(self._weights, reversed(self.real_history)):
            correction.add_(displacement, alpha=weight)
        return correction

    def state_dict(self):
        return {'implementation': self.implementation, 'window_size': self.window_size,
                'poly_order': self.poly_order, 'lambda_v': self.lambda_v,
                'completed_real_rounds': self.completed_real_rounds, 'virtual_events': self.virtual_events,
                'real_history': [x.detach().clone() for x in self.real_history]}

    def load_state_dict(self, state, *, device=None):
        if any(state[key] != getattr(self, key) for key in ('implementation', 'window_size', 'poly_order', 'lambda_v')):
            raise ValueError('DriVE checkpoint configuration does not match')
        history = state['real_history']
        rounds, events = int(state['completed_real_rounds']), int(state['virtual_events'])
        if rounds < 0 or len(history) != min(rounds, self.window_size) or not 0 <= events <= max(0, rounds-self.window_size+1):
            raise ValueError('Invalid DriVE checkpoint counters/history')
        restored = [x.detach().to(device=device if device is not None else x.device).clone() for x in history]
        if any(x.ndim != 1 or not x.is_floating_point() or not bool(torch.isfinite(x).all()) for x in restored):
            raise ValueError('Invalid DriVE checkpoint tensor')
        if restored and any(x.shape != restored[0].shape or x.dtype != restored[0].dtype
                            or x.device != restored[0].device for x in restored):
            raise ValueError('Inconsistent DriVE checkpoint tensors')
        self.real_history = restored
        self.completed_real_rounds, self.virtual_events = rounds, events
