# Copyright (c) OpenMMLab. All rights reserved.
# BF16 Optimizer Hook with mixed-precision and per-module BF16 support

from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmcv.runner import OptimizerHook
from mmcv.runner.hooks import HOOKS
from torch.cuda.amp import GradScaler, autocast

# Norm layer types that should remain FP32 for numerical stability
_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm,
               nn.SyncBatchNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the real model when wrapped by MMDataParallel/DDP."""
    return getattr(model, 'module', model)


def _get_named_module(model: nn.Module, name: str) -> Optional[nn.Module]:
    model = _unwrap_model(model)
    if hasattr(model, 'get_submodule'):
        try:
            return model.get_submodule(name)
        except AttributeError:
            pass

    module = model
    for attr in name.split('.'):
        module = getattr(module, attr, None)
        if module is None:
            return None
    return module


def _set_module_attr(module: nn.Module, attr: str, value) -> None:
    for submodule in module.modules():
        setattr(submodule, attr, value)


def convert_modules_to_bf16(model: nn.Module,
                            module_names: List[str],
                            logger=None,
                            keep_norm_fp32: bool = True,
                            force_bf16_output: bool = False) -> None:
    """Convert specified sub-modules to BF16.

    Args:
        model: The top-level model.
        module_names: List of attribute names to convert (e.g. ['img_backbone']).
        logger: Optional logger for info output.
        keep_norm_fp32: Keep norm layers in FP32 for stability when True.
        force_bf16_output: Ask custom force_fp32 wrappers to cast outputs back
            to BF16 inside converted modules.
    """
    converted = []
    for name in module_names:
        module = _get_named_module(model, name)
        if module is None:
            if logger:
                logger.warning(f'model has no attribute "{name}", skipping')
            continue
        # Convert to BF16
        module.bfloat16()
        _set_module_attr(module, 'bf16_enabled', True)
        _set_module_attr(module, 'force_bf16_output', force_bf16_output)
        if keep_norm_fp32:
            # Restore norm layers to FP32
            for m in module.modules():
                if isinstance(m, _NORM_TYPES):
                    m.float()
        converted.append(name)
    if logger and converted:
        norm_msg = 'norm layers kept FP32' if keep_norm_fp32 else (
            'all parameters including norm layers converted')
        output_msg = ', force_fp32 outputs cast back to BF16' if force_bf16_output else ''
        logger.info(f'BF16 modules: {converted} ({norm_msg}{output_msg})')


@HOOKS.register_module()
class BF16OptimizerHook(OptimizerHook):
    """BF16 Optimizer Hook using PyTorch's native AMP.

    Two modes of operation:

    1. **Mixed precision** (default): Model parameters stay FP32, forward pass
       uses BF16 via ``autocast``. GradScaler is used for loss scaling.

    2. **Per-module BF16** (``bf16_modules`` specified): Selected modules
       (e.g. backbone) are converted to BF16 parameters. Gradients and
       optimizer states for those modules are BF16, halving their memory and
       communication cost. Norm layers inside those modules are kept FP32.
       GradScaler is skipped (BF16 has the same exponent range as FP32).

    Args:
        bf16_modules (list[str], optional): Module names to convert to BF16
            (e.g. ``['img_backbone']``). When set, GradScaler is disabled.
        keep_norm_fp32 (bool): Keep norm layers inside ``bf16_modules`` in
            FP32. Set to False for full BF16 module parameters.
        force_bf16_output (bool): Cast outputs of project custom
            ``force_fp32`` wrappers back to BF16 inside converted modules.
        loss_scale (float | str | dict, optional): Loss scaling configuration.
            Ignored when ``bf16_modules`` is set. Defaults to 'dynamic'.
        grad_clip (dict, optional): Gradient clipping config.
        detect_anomalous_params (bool): See ``OptimizerHook``.
        detect_anomaly (bool): Enable PyTorch anomaly detection.
        distributed (bool): Whether in distributed training.
    """

    def __init__(self,
                 bf16_modules: Optional[List[str]] = None,
                 keep_norm_fp32: bool = True,
                 force_bf16_output: bool = False,
                 loss_scale: Union[float, str, dict] = 'dynamic',
                 grad_clip: Optional[dict] = None,
                 detect_anomalous_params: bool = False,
                 detect_anomaly: bool = False,
                 distributed: bool = True):
        super().__init__(grad_clip=grad_clip,
                         detect_anomalous_params=detect_anomalous_params)
        self.detect_anomaly = detect_anomaly
        self.distributed = distributed
        self.bf16_modules = bf16_modules or []
        self.keep_norm_fp32 = keep_norm_fp32
        self.force_bf16_output = force_bf16_output
        self.use_module_bf16 = len(self.bf16_modules) > 0

        # GradScaler is only needed for mixed-precision (FP32 params) mode.
        # BF16 has the same exponent range as FP32, so loss scaling is not
        # needed to prevent underflow when params are already BF16.
        if self.use_module_bf16:
            self.scaler = None
        else:
            if loss_scale == 'dynamic':
                self.scaler = GradScaler()
            elif isinstance(loss_scale, float):
                self.scaler = GradScaler(init_scale=loss_scale)
            elif isinstance(loss_scale, dict):
                self.scaler = GradScaler(**loss_scale)
            else:
                raise ValueError(
                    f"loss_scale must be 'dynamic', float, or dict, "
                    f"but got {loss_scale}")

    def before_run(self, runner) -> None:
        """Prepare for BF16 training."""
        for m in runner.model.modules():
            if hasattr(m, 'bf16_enabled'):
                m.bf16_enabled = True

        if self.use_module_bf16:
            convert_modules_to_bf16(
                runner.model,
                self.bf16_modules,
                runner.logger,
                keep_norm_fp32=self.keep_norm_fp32,
                force_bf16_output=self.force_bf16_output)
        else:
            runner.logger.info(
                'BF16 mixed precision: params FP32, forward BF16 via autocast')

        # Restore scaler state (mixed-precision mode only)
        if self.scaler is not None:
            if 'bf16' in runner.meta and 'scaler' in runner.meta['bf16']:
                self.scaler.load_state_dict(runner.meta['bf16']['scaler'])

        if self.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            runner.logger.info('Enabled autograd anomaly detection.')

    def before_train_iter(self, runner) -> None:
        """Enable autocast context manager for BF16."""
        self.autocast_context = autocast(dtype=torch.bfloat16)
        self.autocast_context.__enter__()

    def after_train_iter(self, runner) -> None:
        """Perform backward pass, gradient clipping, and optimization step."""
        self.autocast_context.__exit__(None, None, None)

        optimizer = runner.optimizer
        loss = runner.outputs['loss']

        optimizer.zero_grad(set_to_none=True)

        if self.scaler is not None:
            # Mixed-precision mode: FP32 params, GradScaler
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    runner.log_buffer.update(
                        {'grad_norm': float(grad_norm)},
                        runner.outputs['num_samples'])

            self.scaler.step(optimizer)
            self.scaler.update()
            runner.meta.setdefault('bf16', {})['scaler'] = (
                self.scaler.state_dict())
        else:
            # Per-module BF16 mode: no GradScaler needed
            loss.backward()

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    runner.log_buffer.update(
                        {'grad_norm': float(grad_norm)},
                        runner.outputs['num_samples'])

            optimizer.step()


@HOOKS.register_module()
class GradientCumulativeBF16OptimizerHook(BF16OptimizerHook):
    """BF16 optimizer hook with gradient accumulation support.

    This hook combines BF16 training with multi-iter gradient accumulation.
    It inherits the autocast and GradScaler logic from BF16OptimizerHook.
    """
    def __init__(self,
                 cumulative_iters: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.cumulative_iters = cumulative_iters
        self.divisible_iters = 0
        self.remainder_iters = 0
        self.initialized = False

    def _init(self, runner):
        if runner.iter % self.cumulative_iters != 0:
            runner.logger.warning(
                'Resume iteration is not divisible by cumulative_iters. '
                'Some gradients may be lost.')
        self.divisible_iters = (
            runner.max_iters // self.cumulative_iters * self.cumulative_iters)
        self.remainder_iters = runner.max_iters - self.divisible_iters
        self.initialized = True

    def _get_loss_factor(self, runner):
        """Get divisor for loss."""
        if runner.iter < self.divisible_iters:
            return self.cumulative_iters
        else:
            return self.remainder_iters if self.remainder_iters > 0 else 1

    def after_train_iter(self, runner) -> None:
        """Accumulate gradients and step optimizer every `cumulative_iters`."""
        if not self.initialized:
            self._init(runner)

        loss = runner.outputs['loss'] / self._get_loss_factor(runner)

        if self.scaler is not None:
            # Mixed-precision mode: GradScaler
            self.scaler.scale(loss).backward()
        else:
            # Per-module BF16 mode: direct backward
            loss.backward()

        # Exit autocast context
        self.autocast_context.__exit__(None, None, None)

        # Every cumulative_iters or last iteration: optimizer step
        if (self.every_n_iters(runner, self.cumulative_iters) or
                self.is_last_iter(runner)):

            if self.scaler is not None:
                self.scaler.unscale_(runner.optimizer)

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])

            if self.scaler is not None:
                self.scaler.step(runner.optimizer)
                self.scaler.update()
                runner.meta.setdefault('bf16', {})['scaler'] = (
                    self.scaler.state_dict())
            else:
                runner.optimizer.step()

            # Clear gradients
            runner.optimizer.zero_grad(set_to_none=True)
