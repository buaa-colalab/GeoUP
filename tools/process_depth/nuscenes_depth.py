"""Generate sparse nuScenes camera depth maps from LiDAR points."""

import argparse
import multiprocessing as mp
import os
import os.path as osp

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm


CAMERA_NAMES = (
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_FRONT_LEFT',
)

_NUSC = None
_OUTPUT_DIR = None
_NUM_SWEEPS = 1
_MIN_DEPTH = 1.0
_OVERWRITE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Project nuScenes LiDAR points into camera depth maps.')
    parser.add_argument(
        '--data-root', default='data/nuscenes',
        help='nuScenes dataset root')
    parser.add_argument(
        '--output-dir', default='data/nuscenes_depth',
        help='output root mirroring the nuScenes image hierarchy')
    parser.add_argument(
        '--version', '--split', dest='version', default='v1.0-trainval',
        choices=('v1.0-trainval', 'v1.0-test', 'v1.0-mini'),
        help='nuScenes version to process')
    parser.add_argument(
        '--num-sweeps', type=int, default=1,
        help='number of LiDAR sweeps projected into each camera')
    parser.add_argument(
        '--min-depth', type=float, default=1.0,
        help='discard points at or below this camera-frame depth in metres')
    parser.add_argument(
        '--num-workers', type=int, default=min(mp.cpu_count(), 16),
        help='worker processes; use 1 to disable multiprocessing')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='regenerate depth maps that already exist')
    return parser.parse_args()


def _output_path(camera_data):
    relative_path = osp.splitext(camera_data['filename'])[0] + '.png'
    return osp.join(_OUTPUT_DIR, relative_path)


def _save_float16_png(depth_map, output_path):
    depth_bits = np.ascontiguousarray(depth_map, dtype=np.float16).view(np.uint16)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    Image.fromarray(depth_bits).save(output_path)


def process_sample_camera(task):
    sample_idx, camera_name = task
    sample = _NUSC.sample[sample_idx]
    camera_token = sample['data'][camera_name]
    camera_data = _NUSC.get('sample_data', camera_token)
    output_path = _output_path(camera_data)
    if osp.exists(output_path) and not _OVERWRITE:
        return output_path

    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = _NUSC.get('sample_data', lidar_token)
    point_cloud, _ = LidarPointCloud.from_file_multisweep(
        _NUSC, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=_NUM_SWEEPS)

    lidar_calib = _NUSC.get(
        'calibrated_sensor', lidar_data['calibrated_sensor_token'])
    point_cloud.rotate(Quaternion(lidar_calib['rotation']).rotation_matrix)
    point_cloud.translate(np.asarray(lidar_calib['translation']))

    lidar_pose = _NUSC.get('ego_pose', lidar_data['ego_pose_token'])
    point_cloud.rotate(Quaternion(lidar_pose['rotation']).rotation_matrix)
    point_cloud.translate(np.asarray(lidar_pose['translation']))

    camera_pose = _NUSC.get('ego_pose', camera_data['ego_pose_token'])
    point_cloud.translate(-np.asarray(camera_pose['translation']))
    point_cloud.rotate(Quaternion(camera_pose['rotation']).rotation_matrix.T)

    camera_calib = _NUSC.get(
        'calibrated_sensor', camera_data['calibrated_sensor_token'])
    point_cloud.translate(-np.asarray(camera_calib['translation']))
    point_cloud.rotate(Quaternion(camera_calib['rotation']).rotation_matrix.T)

    camera_path = osp.join(_NUSC.dataroot, camera_data['filename'])
    with Image.open(camera_path) as image:
        width, height = image.size

    depths = point_cloud.points[2]
    pixels = view_points(
        point_cloud.points[:3],
        np.asarray(camera_calib['camera_intrinsic']),
        normalize=True)
    visible = (
        np.isfinite(pixels[:2]).all(axis=0)
        & (depths > _MIN_DEPTH)
        & (pixels[0] > 1)
        & (pixels[0] < width - 1)
        & (pixels[1] > 1)
        & (pixels[1] < height - 1)
    )

    x = np.rint(pixels[0, visible]).astype(np.int64)
    y = np.rint(pixels[1, visible]).astype(np.int64)
    visible_depths = depths[visible].astype(np.float32, copy=False)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth_map, (y, x), visible_depths)
    depth_map[~np.isfinite(depth_map)] = 0

    _save_float16_png(depth_map, output_path)
    return output_path


def main():
    args = parse_args()
    if args.num_sweeps < 1:
        raise ValueError('--num-sweeps must be at least 1')
    if args.num_workers < 1:
        raise ValueError('--num-workers must be at least 1')

    global _NUSC, _OUTPUT_DIR, _NUM_SWEEPS, _MIN_DEPTH, _OVERWRITE
    _OUTPUT_DIR = osp.abspath(args.output_dir)
    _NUM_SWEEPS = args.num_sweeps
    _MIN_DEPTH = args.min_depth
    _OVERWRITE = args.overwrite
    _NUSC = NuScenes(
        version=args.version,
        dataroot=osp.abspath(args.data_root),
        verbose=True)

    tasks = [
        (sample_idx, camera_name)
        for sample_idx in range(len(_NUSC.sample))
        for camera_name in CAMERA_NAMES
    ]
    if args.num_workers == 1:
        for task in tqdm(tasks, desc='Generating camera depth'):
            process_sample_camera(task)
        return

    if 'fork' not in mp.get_all_start_methods():
        raise RuntimeError(
            'multiprocessing requires the fork start method; use --num-workers 1')
    context = mp.get_context('fork')
    chunk_size = max(1, len(tasks) // (args.num_workers * 20))
    with context.Pool(args.num_workers) as pool:
        iterator = pool.imap_unordered(
            process_sample_camera, tasks, chunksize=chunk_size)
        for _ in tqdm(iterator, total=len(tasks), desc='Generating camera depth'):
            pass


if __name__ == '__main__':
    main()
