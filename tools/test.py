# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import glob
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
from datetime import timedelta
import torch.distributed as dist


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_model_param_info(model):
    """Return total/trainable parameter numbers."""
    model = _unwrap_model(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_param_info(model, rank=0):
    if rank != 0:
        return
    total_params, trainable_params = get_model_param_info(model)
    print('\n================ Model Parameters ================')
    print(f'Total params:     {total_params:,} ({total_params / 1e6:.2f} M)')
    print(f'Trainable params: {trainable_params:,} ({trainable_params / 1e6:.2f} M)')
    print('==================================================\n')


def benchmark_fps(model,
                  data_loader,
                  samples_per_gpu=1,
                  warmup=20,
                  iters=200,
                  distributed=False,
                  rank=0):
    """Benchmark model forward FPS.

    The reported FPS only includes model forward time. It does not include
    post-evaluation, result formatting, result dumping, or dataset evaluation.
    In distributed mode, total FPS is computed as:
        total processed samples across all ranks / max elapsed time across ranks.
    """
    model.eval()
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))

    local_samples = 0
    measured_iters = 0
    start_time = None

    if rank == 0:
        print('\n================ FPS Benchmark ================')
        print(f'Warmup iters: {warmup}, measured iters: {iters}')
        print('FPS measures model forward only.')

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= warmup + iters:
                break

            if i == warmup:
                torch.cuda.synchronize()
                start_time = time.perf_counter()

            _ = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()

            if i >= warmup:
                measured_iters += 1
                local_samples += samples_per_gpu

    if start_time is None or measured_iters == 0:
        if rank == 0:
            print('No valid iterations were measured. Please reduce --fps-warmup/--fps-iters.')
        return

    elapsed = time.perf_counter() - start_time

    if distributed and dist.is_available() and dist.is_initialized():
        samples_tensor = torch.tensor(float(local_samples), device='cuda')
        elapsed_tensor = torch.tensor(float(elapsed), device='cuda')
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        total_samples = samples_tensor.item()
        total_elapsed = elapsed_tensor.item()
        world_size = dist.get_world_size()
    else:
        total_samples = float(local_samples)
        total_elapsed = float(elapsed)
        world_size = 1

    total_fps = total_samples / max(total_elapsed, 1e-12)
    per_gpu_fps = total_fps / max(world_size, 1)

    if rank == 0:
        print(f'World size:       {world_size}')
        print(f'Measured samples: {int(total_samples)}')
        print(f'Elapsed time:     {total_elapsed:.4f} s')
        print(f'Total FPS:        {total_fps:.2f} samples/s')
        print(f'Per-GPU FPS:      {per_gpu_fps:.2f} samples/s/GPU')
        print('================================================\n')


def _extract_results_payload(payload):
    if isinstance(payload, dict) and 'meta' in payload and 'results' in payload:
        return payload['results'], payload.get('meta') or {}
    return payload, {}


def _expand_pkl_part_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    paths = list(dict.fromkeys(paths))
    missing = [path for path in paths if not osp.exists(path)]
    if missing:
        raise FileNotFoundError('Missing pkl part(s): ' + ', '.join(missing))
    if not paths:
        raise FileNotFoundError('No pkl parts were provided.')
    return paths


def _part_sort_key(item):
    path, meta = item
    node_rank = meta.get('node_rank', None)
    if node_rank is not None:
        return (0, int(node_rank), path)
    return (1, path)


def _merge_pkl_parts(patterns, dataset_size=None):
    parts = []
    for path in _expand_pkl_part_paths(patterns):
        results, meta = _extract_results_payload(mmcv.load(path))
        parts.append((path, meta, results))

    parts = sorted(parts, key=lambda item: _part_sort_key((item[0], item[1])))
    merged = []
    for _, _, results in parts:
        if isinstance(results, dict) and 'bbox_results' in results:
            raise TypeError('Merging dict-style mask results is not implemented for node parts.')
        merged.extend(list(results))
    if dataset_size is not None:
        merged = merged[:dataset_size]
    return merged, [path for path, _, _ in parts]


def _get_node_info(rank, world_size, local_world_size=None):
    if local_world_size is None:
        local_world_size = os.environ.get('LOCAL_WORLD_SIZE', None)
    if local_world_size is None:
        local_world_size = torch.cuda.device_count() if torch.cuda.is_available() else world_size
    local_world_size = min(int(local_world_size), world_size)
    if local_world_size <= 0:
        raise ValueError('local_world_size must be positive.')
    node_rank = rank // local_world_size
    local_rank = rank - node_rank * local_world_size
    num_nodes = (world_size + local_world_size - 1) // local_world_size
    return local_rank, local_world_size, node_rank, num_nodes


def _node_out_path(out, node_rank, num_nodes):
    root, ext = osp.splitext(out)
    return f'{root}_node{node_rank:04d}-of{num_nodes:04d}{ext}'


def _handle_outputs(args, cfg, dataset, outputs, rank):
    if rank != 0:
        return
    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)
    kwargs = {} if args.eval_options is None else args.eval_options
    kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
        '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
    if args.format_only:
        dataset.format_results(outputs, **kwargs)

    if args.vis_depth or args.save_depth:
        if args.eval is None:
            args.eval = ['depth']
        elif 'depth' not in args.eval and 'all' not in args.eval:
            args.eval.append('depth')

    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)

        if args.vis_depth or args.save_depth:
            if 'depth_eval' not in eval_kwargs:
                eval_kwargs['depth_eval'] = {}

            if args.vis_depth:
                eval_kwargs['depth_eval']['vis_depth'] = True
                eval_kwargs['depth_eval']['vis_dir'] = args.vis_depth_dir

            if args.save_depth:
                eval_kwargs['depth_eval']['save_depth'] = True
                eval_kwargs['depth_eval']['save_depth_dir'] = args.save_depth_dir

        if args.eval != 'all':
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

        print(dataset.evaluate(outputs, **eval_kwargs))

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config',help='test config file path')
    parser.add_argument('checkpoint', nargs='?', default='None', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--out-per-node',
        action='store_true',
        help='in distributed mode, each node writes its local result pkl instead of collecting all nodes')
    parser.add_argument(
        '--local-world-size',
        type=int,
        default=None,
        help='number of test processes on each node; defaults to LOCAL_WORLD_SIZE or visible CUDA device count')
    parser.add_argument(
        '--load-pkl',
        help='load an existing merged result pkl and run format/eval without model inference')
    parser.add_argument(
        '--merge-pkl-parts',
        nargs='+',
        help='merge node result pkls or glob patterns; can be combined with --out and --eval')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')

    parser.add_argument(
        '--vis-depth',
        action='store_true',
        help='Whether to visualize and save depth predictions vs ground truth')
    parser.add_argument(
        '--vis-depth-dir',
        type=str,
        default='depth_vis_results',
        help='Directory where depth visualizations will be saved')
    parser.add_argument(
        '--save-depth',
        action='store_true',
        help='Whether to save depth predictions in specific dir format')
    parser.add_argument(
        '--save-depth-dir',
        type=str,
        default='saved_depths',
        help='Directory where depth maps will be saved')

    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument('--return_loss', action='store_true', help='test val loss')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--params',
        action='store_true',
        help='print total and trainable parameter numbers')
    parser.add_argument(
        '--fps',
        action='store_true',
        help='benchmark model forward FPS')
    parser.add_argument(
        '--fps-iters',
        type=int,
        default=200,
        help='number of measured iterations for FPS benchmark')
    parser.add_argument(
        '--fps-warmup',
        type=int,
        default=20,
        help='number of warmup iterations before FPS benchmark')
    parser.add_argument(
        '--fps-only',
        action='store_true',
        help='only run params/FPS benchmark and skip normal test/eval')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    if args.load_pkl and args.merge_pkl_parts:
        raise ValueError('--load-pkl and --merge-pkl-parts cannot both be specified')
    if args.out_per_node and not args.out:
        raise ValueError('--out-per-node requires --out')
    return args


def main():
    args = parse_args()

    normal_test_op = (args.out or args.eval or args.format_only or args.show
                      or args.show_dir or args.return_loss
                      or args.vis_depth or args.save_depth
                      or args.load_pkl or args.merge_pkl_parts)
    benchmark_op = args.params or args.fps
    benchmark_only = benchmark_op and not normal_test_op

    assert normal_test_op or benchmark_op, \
        ('Please specify at least one operation with the argument "--out", '
         '"--eval", "--format-only", "--show", "--show-dir", '
         '"--return_loss", "--vis-depth", "--save-depth", '
         '"--params", "--fps", "--load-pkl" or "--merge-pkl-parts"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    skip_external_checkpoint_load = False
    if args.checkpoint != 'None' and args.checkpoint.endswith('.safetensors'):
        if 'checkpoint_path' not in cfg.model:
            raise ValueError(
                'Received a .safetensors checkpoint, but cfg.model has no '
                'checkpoint_path field for model-internal loading. Use a '
                '.pth/.pt checkpoint or update the model wrapper.')
        cfg.model.checkpoint_path = args.checkpoint
        skip_external_checkpoint_load = True

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    if args.return_loss:
        cfg.data.test.test_mode = False
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # ---- make a copy; convert seconds -> timedelta without mutating cfg ----
        _dp = dict(cfg.dist_params)  # copy! do NOT mutate cfg.dist_params
        _to = _dp.get('timeout', None)
        if isinstance(_to, (int, float)):
            _dp['timeout'] = timedelta(seconds=int(_to))
        elif _to is None:
            _dp.pop('timeout', None)  # keep default if not provided
        init_dist(args.launcher, **_dp)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    rank, world_size = get_dist_info() if distributed else (0, 1)

    if args.load_pkl or args.merge_pkl_parts:
        if distributed:
            raise ValueError('--load-pkl/--merge-pkl-parts should be run without distributed launcher')
        if args.merge_pkl_parts:
            outputs, part_paths = _merge_pkl_parts(args.merge_pkl_parts, len(dataset))
            print(f'Merged {len(part_paths)} pkl part(s) into {len(outputs)} results.')
        else:
            outputs, _ = _extract_results_payload(mmcv.load(args.load_pkl))
            if hasattr(outputs, '__len__') and len(outputs) != len(dataset):
                warnings.warn(
                    f'Loaded {len(outputs)} results, but dataset has {len(dataset)} samples.')
        _handle_outputs(args, cfg, dataset, outputs, rank)
        return

    # build the model and load checkpoint
    if not args.return_loss:
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    else:
        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)

    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if not args.checkpoint == 'None' and not skip_external_checkpoint_load:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE
    else:
        if skip_external_checkpoint_load:
            print(f'Skipping MMCV load_checkpoint for safetensors: {args.checkpoint}')
            model.CLASSES = dataset.CLASSES
        model.init_weights()
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    rank, world_size = get_dist_info() if distributed else (0, 1)

    # Parameter counting should be done before DDP wrapping to avoid duplicated printing.
    if args.params:
        print_model_param_info(model, rank=rank)
        if benchmark_only and not args.fps:
            if distributed and dist.is_available() and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            return

    if not distributed:
        model = MMDataParallel(model.cuda(), device_ids=[0])
        if args.fps:
            benchmark_fps(
                model,
                data_loader,
                samples_per_gpu=samples_per_gpu,
                warmup=args.fps_warmup,
                iters=args.fps_iters,
                distributed=False,
                rank=rank)
            if args.fps_only or benchmark_only:
                return
        if args.return_loss:
            model.train()
            outputs = single_gpu_test(model, data_loader)
            return
        else:
            outputs = single_gpu_test(model, data_loader)
    else:
        model = model.cuda()
        if any(param.requires_grad for param in model.parameters()):
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
        else:
            model = MMDataParallel(
                model,
                device_ids=[torch.cuda.current_device()])
        if args.fps:
            benchmark_fps(
                model,
                data_loader,
                samples_per_gpu=samples_per_gpu,
                warmup=args.fps_warmup,
                iters=args.fps_iters,
                distributed=True,
                rank=rank)
            if args.fps_only or benchmark_only:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
                return
        if args.return_loss:
            model.train()
            outputs = custom_multi_gpu_test(
                model, data_loader, args.tmpdir, args.gpu_collect, True,
                collect_by_node=args.out_per_node,
                local_world_size=args.local_world_size)
            return
        else:
            outputs = custom_multi_gpu_test(
                model, data_loader, args.tmpdir, args.gpu_collect, False,
                collect_by_node=args.out_per_node,
                local_world_size=args.local_world_size)

    try:
        if args.out_per_node:
            if not distributed:
                raise ValueError('--out-per-node requires distributed mode')
            local_rank, local_world_size, node_rank, num_nodes = _get_node_info(
                rank, world_size, args.local_world_size)
            if local_rank == 0:
                out_path = _node_out_path(args.out, node_rank, num_nodes)
                meta = dict(
                    type='streampetr_test_part',
                    scope='node',
                    node_rank=node_rank,
                    num_nodes=num_nodes,
                    local_world_size=local_world_size,
                    world_size=world_size,
                    dataset_size=len(dataset),
                    config=args.config,
                    checkpoint=args.checkpoint)
                print(f'\nwriting node results to {out_path}')
                mmcv.dump(dict(meta=meta, results=outputs), out_path)
        else:
            _handle_outputs(args, cfg, dataset, outputs, rank)

        if distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()
    finally:
        # 关键：所有 rank 显式销毁 PG，避免析构阶段乱序 abort
        if distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()