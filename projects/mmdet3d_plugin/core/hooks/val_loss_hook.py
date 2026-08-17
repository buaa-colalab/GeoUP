from collections import OrderedDict
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook, get_dist_info


@HOOKS.register_module()
class ValLossHook(Hook):
    """Periodically run validation loss on a held-out dataloader."""

    def __init__(self,
                 dataloader,
                 interval=1,
                 by_epoch=False,
                 epoch_length=None,
                 prefix='val'):
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.epoch_length = epoch_length
        self.prefix = prefix.rstrip('/')

    def after_train_iter(self, runner):
        if self.by_epoch:
            if self.epoch_length is None:
                raise ValueError('epoch_length must be set when by_epoch=True '
                                 'under IterBasedRunner.')
            if (runner.iter + 1) % (self.interval * self.epoch_length) != 0:
                return
            self._run_val_loss(runner)
            return

        if self.every_n_iters(runner, self.interval):
            self._run_val_loss(runner)

    def after_train_epoch(self, runner):
        if self.by_epoch and self.every_n_epochs(runner, self.interval):
            self._run_val_loss(runner)

    def _run_val_loss(self, runner):
        model = runner.model
        was_training = model.training
        model.eval()

        log_sums = OrderedDict()
        num_batches = 0
        rank, world_size = get_dist_info()
        prog_bar = None
        if rank == 0:
            runner.logger.info('Running validation loss...')
            prog_bar = mmcv.ProgressBar(len(self.dataloader.dataset))
        time.sleep(2)

        with torch.no_grad():
            for data in self.dataloader:
                outputs = model.train_step(data, optimizer=None)
                for key, value in outputs['log_vars'].items():
                    if torch.is_tensor(value):
                        value = value.item()
                    else:
                        value = float(value)
                    log_sums[key] = log_sums.get(key, 0.0) + value
                num_batches += 1
                if prog_bar is not None:
                    batch_size = int(outputs.get('num_samples', 1))
                    for _ in range(batch_size * world_size):
                        prog_bar.update()

        averaged_logs = self._reduce_logs(log_sums, num_batches)
        if rank == 0:
            if self.by_epoch:
                if self.epoch_length is not None:
                    progress = (runner.iter + 1) // self.epoch_length
                else:
                    progress = runner.epoch + 1
            else:
                progress = runner.iter + 1
            unit = 'epoch' if self.by_epoch else 'iter'
            summary = ', '.join(
                f'{self.prefix}_{key}: {value:.4f}'
                for key, value in averaged_logs.items())
            runner.logger.info(f'Validation loss at {unit} {progress}: {summary}')
            for key, value in averaged_logs.items():
                runner.log_buffer.output[f'{self.prefix}_{key}'] = value
            runner.log_buffer.ready = True

        if was_training:
            model.train()

    def _reduce_logs(self, log_sums, num_batches):
        if num_batches == 0:
            return OrderedDict()

        device = torch.device('cuda', torch.cuda.current_device()) \
            if torch.cuda.is_available() else torch.device('cpu')
        averaged_logs = OrderedDict()
        world_dist = dist.is_available() and dist.is_initialized()

        for key, value in log_sums.items():
            stats = torch.tensor([value, float(num_batches)],
                                 dtype=torch.float64,
                                 device=device)
            if world_dist:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            averaged_logs[key] = (stats[0] / stats[1].clamp(min=1.0)).item()

        return averaged_logs
