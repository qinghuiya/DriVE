"""Shared sequential FL engine: one global model and one reusable client model."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .aggregation import StreamingFedAvgAccumulator
from .drive import DriVEController
from .current_only import CurrentOnlyController

METHODS = (
    'fedavg', 'drive', 'fedprox', 'fedavgm', 'fedadam', 'fedgmt',
    'fedexprox_constant', 'fedexprox_grads',
    'current_only',
)


@dataclass
class RoundResult:
    selected_clients: list[int]
    client_samples: int
    metrics: dict


class FederatedEngine:
    def __init__(self, model, client_loaders, config):
        self.config = config
        self.method = config['algorithm']['name']
        if self.method not in METHODS:
            raise ValueError(f'Unknown method: {self.method}')
        self.device = torch.device(config['experiment']['device'])
        self.global_model = model.to(self.device)
        self.worker_model = copy.deepcopy(self.global_model)
        self.client_loaders = client_loaders
        self.seed = config['experiment']['seed']
        self.training = config['training']
        self.local_steps = self.training['local_steps']
        self.local_epochs = self.training['local_epochs']
        self.clients_per_round = self.training['clients_per_round']
        if self.training.get('participation') is not None:
            self.clients_per_round = max(1,round(self.training['participation']*len(client_loaders)))
        if not 1 <= self.clients_per_round <= len(client_loaders):
            raise ValueError('clients_per_round must be between 1 and num_clients')
        self.options = config['baseline']
        self.optimizer = torch.optim.SGD(
            self.worker_model.parameters(), **config['algorithm']['client_optimizer'])
        self._global_state = dict(self.global_model.state_dict())
        self._worker_state = dict(self.worker_model.state_dict())
        self.accumulator = StreamingFedAvgAccumulator(self.global_model)
        self.drive = DriVEController(**config['drive']) if self.method == 'drive' else None
        self.current_only = (
            CurrentOnlyController(**config['current_only']) if self.method == 'current_only' else None)
        self._momentum = None
        self._variance = None
        self._dual_bank = {}
        self._dual_sum = None
        self.teacher = None
        if self.method == 'fedgmt':
            self.teacher = copy.deepcopy(self.global_model).eval().requires_grad_(False)
            self._dual_sum = torch.zeros_like(self._flat_parameters())

    def _flat_parameters(self, model=None):
        model = self.global_model if model is None else model
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    @torch.no_grad()
    def _add_delta(self, delta):
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError('Nonfinite server displacement')
        offset = 0
        for p in self.global_model.parameters():
            n = p.numel()
            p.add_(delta[offset:offset+n].view_as(p))
            offset += n
        if offset != delta.numel():
            raise ValueError('Server displacement dimension mismatch')

    @torch.no_grad()
    def _reset_client(self):
        # Restore parameters AND non-parameter buffers (including BatchNorm).
        for name, tensor in self._worker_state.items():
            tensor.copy_(self._global_state[name])
        # Storage is reused but SGD momentum never crosses client boundaries.
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.zero_()

    @staticmethod
    def _next_batch(loader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            try:
                return next(iterator), iterator
            except StopIteration as exc:
                raise RuntimeError('A selected client has an empty loader') from exc

    def _train_client(self, loader, client_id):
        self._reset_client()
        self.worker_model.train()
        proximal = self.method in ('fedprox', 'fedexprox_constant', 'fedexprox_grads')
        global_parameters = list(self.global_model.parameters())
        dual = None
        if self.method == 'fedgmt':
            host = self._dual_bank.get(client_id)
            dual = torch.zeros_like(self._dual_sum) if host is None else host.to(self.device)
            dual_parts = dual.split([p.numel() for p in self.worker_model.parameters()])
        samples = 0

        def train_batch(x, y):
            nonlocal samples
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.worker_model(x)
            loss = F.cross_entropy(logits, y)
            if self.method == 'fedgmt':
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                tau = self.options['gmt_tau']
                kl = F.kl_div(F.log_softmax(logits/tau, dim=1),
                              F.softmax(teacher_logits/tau, dim=1), reduction='batchmean')
                loss = loss + self.options['gmt_gamma'] * tau**2 * kl
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite local loss')
            loss.backward()
            with torch.no_grad():
                if proximal and self.options['proximal_mu']:
                    for wp, gp in zip(self.worker_model.parameters(), global_parameters):
                        if wp.grad is not None:
                            wp.grad.add_(wp.detach()-gp.detach(), alpha=self.options['proximal_mu'])
                if dual is not None:
                    for p, part in zip(self.worker_model.parameters(), dual_parts):
                        if p.grad is not None:
                            p.grad.add_(part.view_as(p), alpha=1/self.options['gmt_beta'])
            self.optimizer.step()
            samples += int(y.numel())

        if self.local_epochs is not None:
            for _ in range(self.local_epochs):
                for x, y in loader:
                    train_batch(x, y)
        else:
            iterator = iter(loader)
            # Repeat short loaders until exactly K minibatch updates are made.
            for _ in range(self.local_steps):
                (x, y), iterator = self._next_batch(loader, iterator)
                train_batch(x, y)
        if samples == 0:
            raise RuntimeError('A selected client produced no samples')
        squared_norm = 0.
        if self.method == 'fedgmt':
            delta = torch.cat([(p.detach()-g.detach()).flatten().to(torch.float32)
                               for p, g in zip(self.worker_model.parameters(), global_parameters)])
            # Lazy CPU dual bank: unvisited clients have an implicit zero dual.
            host_delta = delta.cpu()
            if client_id not in self._dual_bank:
                self._dual_bank[client_id] = torch.zeros_like(host_delta)
            self._dual_bank[client_id].add_(host_delta)
            self._dual_sum.add_(delta)
        elif self.method == 'fedexprox_grads':
            sq = torch.zeros((), device=self.device)
            for p, g in zip(self.worker_model.parameters(), global_parameters):
                sq.add_((p.detach()-g.detach()).to(torch.float32).square().sum())
            squared_norm = float(sq)
        return samples, squared_norm

    def train_one_round(self, round_index):
        metrics = {}
        selected = random.Random(self.seed+round_index).sample(
            range(len(self.client_loaders)), self.clients_per_round)
        self.accumulator.reset()
        total_samples, sum_sq = 0, 0.
        for cid in selected:
            samples, sq = self._train_client(self.client_loaders[cid], cid)
            self.accumulator.add_state(self._worker_state, samples)
            total_samples += samples
            sum_sq += samples * sq
        raw = self.accumulator.finalize_into_global(need_flat_update=self.method != 'fedavg')
        # Floating buffers are sample-averaged; integer buffers come from the
        # client with the largest processed-sample count (first on a tie).
        if self.drive is not None:
            result = self.drive.observe(raw)
            if result.update is not None:
                self._add_delta(result.update)
            metrics.update(result.metrics)
        elif self.current_only is not None:
            result = self.current_only.observe(raw)
            if result.update is not None:
                self._add_delta(result.update)
            metrics.update(result.metrics)
        elif self.method == 'fedgmt':
            self._add_delta(self._dual_sum/len(self.client_loaders))
            with torch.no_grad():
                alpha = self.options['gmt_ema_alpha']
                for tp, gp in zip(self.teacher.parameters(), self.global_model.parameters()):
                    tp.mul_(alpha).add_(gp, alpha=1-alpha)
                buffers = dict(self.global_model.named_buffers())
                for name, buffer in self.teacher.named_buffers():
                    buffer.copy_(buffers[name])
        elif self.method in ('fedavgm', 'fedadam', 'fedexprox_constant', 'fedexprox_grads'):
            opts = self.options
            if self.method == 'fedavgm':
                if self._momentum is None:
                    self._momentum = torch.zeros_like(raw)
                self._momentum.mul_(opts['momentum_beta']).add_(raw)
                applied = opts['server_lr'] * self._momentum
            elif self.method == 'fedadam':
                if self._momentum is None:
                    self._momentum, self._variance = torch.zeros_like(raw), torch.zeros_like(raw)
                self._momentum.mul_(opts['adam_beta1']).add_(raw, alpha=1-opts['adam_beta1'])
                # This framework's experimental variant: second moment of m_t,
                # no bias correction. See README; this is not canonical FedAdam.
                self._variance.mul_(opts['adam_beta2']).addcmul_(
                    self._momentum, self._momentum, value=1-opts['adam_beta2'])
                applied = opts['adam_lr']*self._momentum/(self._variance.sqrt()+opts['adam_tau'])
            elif self.method == 'fedexprox_constant':
                applied = opts['extrap_eta'] * raw
                metrics['extrap_eta'] = opts['extrap_eta']
            else:
                ratio = (sum_sq/max(total_samples, 1))/max(float(torch.dot(raw, raw)), 1e-24)
                if not torch.isfinite(torch.tensor(ratio, dtype=torch.float64)):
                    raise FloatingPointError('Nonfinite extrapolation coefficient')
                eta = min(opts['grads_eta_cap'], max(1., ratio))
                applied = eta * raw
                metrics.update(extrap_eta_raw=ratio, extrap_eta=eta)
            # FedAvg has already committed raw; replace only the parameter step.
            self._add_delta(applied-raw)
        if not all(bool(torch.isfinite(v).all()) for v in self._global_state.values()):
            raise FloatingPointError('Nonfinite global model state')
        return RoundResult(selected, total_samples, metrics)

    @torch.inference_mode()
    def evaluate(self, loader):
        self.global_model.eval()
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        correct = torch.zeros((), device=self.device, dtype=torch.int64)
        total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.global_model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum.add_(loss.to(torch.float64), alpha=len(y))
            correct.add_((logits.argmax(dim=1) == y).sum())
            total += len(y)
        if total == 0:
            raise ValueError('Empty evaluation loader')
        return {'loss': float(loss_sum/total), 'accuracy': float(correct/total)}
