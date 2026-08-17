from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, List

import torch
from mmcv.runner.optimizer.builder import OPTIMIZERS


def zeropower_via_newtonschulz5(
        grad: torch.Tensor,
        steps: int,
        eps: float = 1e-7) -> torch.Tensor:
    """Approximate the nearest orthogonal update with Newton-Schulz iterations.

    The update is computed in BF16 on CUDA for speed and then cast back to the
    gradient dtype before being applied to the FP32 parameters.
    """
    if grad.ndim != 2:
        raise ValueError(f'Expected a 2D tensor, but got shape {tuple(grad.shape)}')

    a, b, c = (3.4445, -4.7750, 2.0315)
    if grad.is_cuda and torch.cuda.is_bf16_supported():
        x = grad.to(dtype=torch.bfloat16)
    else:
        x = grad.to(dtype=torch.float32)

    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.transpose(0, 1)

    x.div_(x.norm().clamp(min=eps))
    for _ in range(steps):
        gram = x @ x.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, gram_update, x, beta=a)

    if transposed:
        x = x.T
    return x.to(dtype=grad.dtype)


def zeropower_via_newtonschulz5_batched(
        grads: torch.Tensor,
        steps: int,
        eps: float = 1e-7) -> torch.Tensor:
    """Batched Newton-Schulz orthogonalization for tensors of shape [B, M, N]."""
    if grads.ndim != 3:
        raise ValueError(
            f'Expected a 3D tensor of shape [B, M, N], but got {tuple(grads.shape)}')

    a, b, c = (3.4445, -4.7750, 2.0315)
    if grads.is_cuda and torch.cuda.is_bf16_supported():
        x = grads.to(dtype=torch.bfloat16)
    else:
        x = grads.to(dtype=torch.float32)

    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)

    x.div_(torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp(min=eps))
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, gram_update, x, beta=a)

    if transposed:
        x = x.transpose(-2, -1)
    return x.to(dtype=grads.dtype)


def _should_use_muon(param: torch.Tensor, min_muon_ndim: int,
                     min_muon_numel: int) -> bool:
    return (param.requires_grad and param.ndim >= min_muon_ndim
            and param.numel() >= min_muon_numel)


def _as_single_param_groups(params: Iterable,
                            min_muon_ndim: int,
                            min_muon_numel: int) -> List[dict]:
    """Expand the incoming params to one parameter per group.

    MMCV's optimizer constructor already does this for paramwise_cfg, but we
    keep the logic here so the optimizer also works when instantiated directly.
    """
    param_groups = []
    for item in params:
        if isinstance(item, dict):
            group = dict(item)
            raw_params = group.pop('params')
            if isinstance(raw_params, torch.Tensor):
                raw_params = [raw_params]
            else:
                raw_params = list(raw_params)
            for param in raw_params:
                param_group = dict(group)
                param_group['params'] = [param]
                if 'use_muon' not in param_group:
                    use_adamw = bool(param_group.pop('use_adamw', False))
                    param_group['use_muon'] = (
                        _should_use_muon(param, min_muon_ndim, min_muon_numel)
                        and not use_adamw)
                param_groups.append(param_group)
        else:
            param_groups.append(
                dict(
                    params=[item],
                    use_muon=_should_use_muon(item, min_muon_ndim,
                                              min_muon_numel)))
    return param_groups


@OPTIMIZERS.register_module(force=True)
class Muon(torch.optim.Optimizer):
    """Muon optimizer with AdamW fallback for small tensors.

    By default, tensors with ``ndim >= min_muon_ndim`` use Muon and tensors
    below that threshold use AdamW. A specific param group can override this
    with ``use_muon=True/False`` or ``use_adamw=True``.
    """

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 weight_decay: float = 0.01,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 ns_steps: int = 5,
                 ns_eps: float = 1e-7,
                 muon_lr_scale: float = 0.2,
                 min_muon_ndim: int = 2,
                 min_muon_numel: int = 0,
                 adamw_betas=(0.9, 0.95),
                 adamw_eps: float = 1e-8):
        if lr <= 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if weight_decay < 0:
            raise ValueError(f'Invalid weight decay: {weight_decay}')
        if not 0 <= momentum < 1:
            raise ValueError(f'Invalid momentum: {momentum}')
        if ns_steps <= 0:
            raise ValueError(f'Invalid ns_steps: {ns_steps}')
        if ns_eps <= 0:
            raise ValueError(f'Invalid ns_eps: {ns_eps}')
        if muon_lr_scale <= 0:
            raise ValueError(f'Invalid muon_lr_scale: {muon_lr_scale}')
        if min_muon_ndim < 2:
            raise ValueError(f'Invalid min_muon_ndim: {min_muon_ndim}')
        if min_muon_numel < 0:
            raise ValueError(f'Invalid min_muon_numel: {min_muon_numel}')

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            muon_lr_scale=muon_lr_scale,
            min_muon_ndim=min_muon_ndim,
            min_muon_numel=min_muon_numel,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        param_groups = _as_single_param_groups(params, min_muon_ndim,
                                               min_muon_numel)
        super().__init__(param_groups, defaults)

        for group in self.param_groups:
            use_muon = bool(group.get('use_muon', False))
            for param in group['params']:
                self.state[param]['use_muon'] = use_muon and param.requires_grad

    @staticmethod
    def _adjust_muon_lr(lr: float,
                        param_shape,
                        muon_lr_scale: float) -> float:
        rows, cols = param_shape[:2]
        return lr * muon_lr_scale * math.sqrt(max(rows, cols))

    @staticmethod
    def _reshape_grad(grad: torch.Tensor) -> torch.Tensor:
        if grad.ndim > 2:
            return grad.reshape(grad.size(0), -1)
        return grad

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_buckets = defaultdict(list)
        for group in self.param_groups:
            use_muon = bool(group.get('use_muon', False))

            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')

                if use_muon and self.state[param].get('use_muon', False):
                    grad_matrix = self._reshape_grad(grad)
                    bucket_key = (
                        grad_matrix.device,
                        grad_matrix.dtype,
                        tuple(grad_matrix.shape),
                        group['lr'],
                        group.get('weight_decay', 0.0),
                        group['momentum'],
                        group['nesterov'],
                        group['ns_steps'],
                        group['ns_eps'],
                        group['muon_lr_scale'],
                    )
                    muon_buckets[bucket_key].append((param, grad_matrix))
                    continue

                lr = group['lr']
                weight_decay = group.get('weight_decay', 0.0)
                state = self.state[param]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['adamw_betas']
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.square(), 1 - beta2)

                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                denom = exp_avg_sq.sqrt().add_(group['adamw_eps'])
                param.addcdiv_(exp_avg, denom, value=-step_size)

        for bucket_key, bucket_items in muon_buckets.items():
            (_, _, grad_shape, lr, weight_decay, momentum, nesterov, ns_steps,
             ns_eps, muon_lr_scale) = bucket_key
            params = []
            grad_mats = []
            momentum_buffers = []

            for param, grad_matrix in bucket_items:
                state = self.state[param]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad_matrix)
                params.append(param)
                grad_mats.append(grad_matrix)
                momentum_buffers.append(state['momentum_buffer'])

            if len(bucket_items) == 1:
                momentum_buffer = momentum_buffers[0]
                grad_matrix = grad_mats[0]
                momentum_buffer.mul_(momentum).add_(grad_matrix)
                if nesterov:
                    update_input = grad_matrix.add(momentum_buffer,
                                                   alpha=momentum)
                else:
                    update_input = momentum_buffer
                update = zeropower_via_newtonschulz5(
                    update_input, steps=ns_steps, eps=ns_eps)
                adjusted_lr = self._adjust_muon_lr(lr, grad_shape,
                                                   muon_lr_scale)
                if weight_decay != 0:
                    params[0].mul_(1 - lr * weight_decay)
                params[0].add_(update.reshape_as(params[0]), alpha=-adjusted_lr)
                continue

            torch._foreach_mul_(momentum_buffers, momentum)
            torch._foreach_add_(momentum_buffers, grad_mats)
            if nesterov:
                stacked_updates = torch.stack(grad_mats, dim=0).add(
                    torch.stack(momentum_buffers, dim=0), alpha=momentum)
            else:
                stacked_updates = torch.stack(momentum_buffers, dim=0)
            batched_update = zeropower_via_newtonschulz5_batched(
                stacked_updates, steps=ns_steps, eps=ns_eps)
            adjusted_lr = self._adjust_muon_lr(lr, grad_shape, muon_lr_scale)

            for idx, param in enumerate(params):
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                param.add_(batched_update[idx].reshape_as(param),
                           alpha=-adjusted_lr)

        return loss
