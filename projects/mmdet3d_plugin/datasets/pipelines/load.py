import os
import mmcv
import numpy as np
import torch
from PIL import Image

from mmdet3d.datasets.builder import PIPELINES
from torchvision.transforms.functional import rotate, InterpolationMode

@PIPELINES.register_module()
class LoadMultiViewImageFromFilesV1(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = [mmcv.imread(name, self.color_type) for name in filename]
        # img_shapes = np.stack([i.shape for i in img], axis=0)
        # img_shape_max = np.max(img_shapes, axis=0)
        # img_shape_min = np.min(img_shapes, axis=0)
        # if not np.all(img_shape_max == img_shape_min):
        #     pad_shape = img_shape_max[:2]
        # else:
        #     pad_shape = None
        # if pad_shape is not None:
        #     img = [
        #         mmcv.impad(i, shape=pad_shape, pad_val=0) for i in img
        #     ]
        # img = np.stack(img, axis=-1)
        # img = np.stack(
        #     [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        # print(f"loading image, scene: {results['scene_token']}, time: {results['timestamp']}")
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = img
        results['img_shape'] = img[0].shape
        results['ori_shape'] = img[0].shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img[0].shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewDepthFromFiles(object):
    """Load multi channel depth map from a list of separate channel files.

    Expects results['depth_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the depth to float32.
            Defaults to False.
    """

    def __init__(self, to_float32=False, min_dist=1e-8, max_dist=70, use_all_depth=False, no_file=False):
        self.to_float32 = to_float32
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.use_all_depth = use_all_depth
        self.no_file = no_file

    def __call__(self, results):
        """Call function to load multi-view depth map from files.

        Args:
            results (dict): Result dict containing multi-view depth map filenames.

        Returns:
            dict: The result dict containing the multi-view depth map data.
                Added keys and values are described below.

                - filename (str): Multi-view depth map filenames.
                - gt_depth (np.ndarray): Multi-view depth map arrays.
                - point_mask (np.ndarray): Multi-view depth map point mask.

                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
        """
        filename = results['depth_filename']
        # img is of shape (h, w, num_views)
        if self.no_file:
            depth = np.zeros((results['img_shape'][0], results['img_shape'][1], len(filename)), dtype=np.float32)
        else:
            depth = np.stack(
                [load_16big_png_depth(name) for name in filename], axis=-1)
        if self.to_float32:
            depth = depth.astype(np.float32)
        # which will transpose each image separately and then stack into array
        if self.use_all_depth:
            point_mask = (depth > self.min_dist)
        else:
            point_mask = (depth > self.min_dist) & (depth < self.max_dist)
        results['gt_depth'] = [depth[..., i] / self.max_dist for i in range(depth.shape[-1])]
        # def norm_cam_extrinsics(mat, dist):
        #     mat[:3, 3] = mat[:3, 3] / dist
        #     return mat
        # results['cam_extrinsics_global'] = [norm_cam_extrinsics(results['cam_extrinsics_global'][i], self.max_dist) for i in range(depth.shape[-1])]
        results['point_mask'] = [point_mask[..., i] for i in range(point_mask.shape[-1])]
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    """
    Loads a 16-bit PNG as a half-float depth map (H, W), returning a float16 NumPy array.

    Implementation detail:
      - PIL loads 16-bit data as 32-bit "I" mode.

    Args:
        depth_png (str):
            File path to the 16-bit PNG.

    Returns:
        np.ndarray:
            A float16 depth array of shape (H, W).
    """
    with Image.open(depth_png) as depth_pil:
        depth = np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16).reshape((depth_pil.size[1], depth_pil.size[0]))
    return depth


@PIPELINES.register_module()
class LoadMultiViewDepthFromNpyFiles(object):
    """从一系列 .npy 文件中加载多视图深度图。

    这个类能够处理不同视图的深度图尺寸不一致的情况。
    它期望 results['depth_filename'] 是一个包含文件路径的列表。

    Args:
        to_float32 (bool, optional): 是否将深度图转换为 float32 类型。
            默认为 True。
        min_dist (float, optional): 有效深度的最小距离。小于此值的点将被遮蔽。
            默认为 1e-8。
        max_dist (float, optional): 有效深度的最大距离。此值也用于深度图的归一化。
            默认为 70。
    """

    def __init__(self, to_float32=True, min_dist=1e-8, max_dist=70, use_all_depth=False):
        self.to_float32 = to_float32
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.use_all_depth = use_all_depth

    def __call__(self, results):
        """加载多视图深度图文件的调用函数。

        Args:
            results (dict): 包含多视图深度图文件名的结果字典。

        Returns:
            dict: 包含多视图深度图数据的结果字典。
                添加的键和值如下：
                - gt_depth (list[np.ndarray]): 多视图深度图数组的列表，每个都被 max_dist 归一化。
                - point_mask (list[np.ndarray]): 对应每个视图的有效深度点的布尔掩码列表。
        """
        filenames = results['depth_filename']

        # 从.npy文件加载深度图到列表中，这天然地支持了不同尺寸
        depth_list = [np.load(name) for name in filenames]

        if self.to_float32:
            depth_list = [depth.astype(np.float32) for depth in depth_list]

        # 为每个深度图创建一个布尔掩码列表
        if self.use_all_depth:
            point_mask_list = [depth > self.min_dist for depth in depth_list]
        else:
            point_mask_list = [(depth > self.min_dist) & (depth < self.max_dist) for depth in depth_list]

        # 将归一化后的深度图列表存储到 results 中
        results['gt_depth'] = [depth / self.max_dist for depth in depth_list]

        # 将掩码列表存储到 results 中
        results['point_mask'] = point_mask_list

        # 归一化相机外参的平移分量
        def norm_cam_extrinsics(mat, dist):
            mat[:3, 3] = mat[:3, 3] / dist
            return mat

        # breakpoint()
        # num_views = len(depth_list)
        # results['cam_extrinsics_global'] = [
        #     norm_cam_extrinsics(results['cam_extrinsics_global'][i], self.max_dist)
        #     for i in range(num_views)
        # ]
        # print(f"loading depth, scene: {results['scene_token']}, time: {results['timestamp']}")

        return results

    def __repr__(self):
        """str: 返回描述该模块的字符串。"""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f'min_dist={self.min_dist}, '
        repr_str += f'max_dist={self.max_dist})'
        return repr_str


@PIPELINES.register_module()
class LoadOccAnnotations(object):
    """Load OPUSV2 OCC supervision."""

    def __init__(self, occ_path='occ_gts'):
        self.occ_path = occ_path

    def __call__(self, results):
        """Load OCC annotations from file."""
        if 'occ_path' not in results:
            # Construct OCC path from scene_token
            scene_name = results.get('scene_name', '')
            token = results.get('sample_idx', '')
            if scene_name and token:
                occ_file = os.path.join(self.occ_path, scene_name, token + '/labels.npz')
                if os.path.exists(occ_file):
                    results['occ_path'] = occ_file
                else:
                    raise FileNotFoundError(f"OCC file not found: {occ_file}")
            else:
                raise ValueError("Both scene_name and token are required to construct OCC path")

        try:
            occ_labels = np.load(results['occ_path'])
        except Exception as e:
            raise RuntimeError(f"Failed to load OCC file {results['occ_path']}: {str(e)}")

        semantics = occ_labels['semantics'].copy()

        results['voxel_semantics'] = semantics

        if 'mask_camera' in occ_labels:
            results['mask_camera'] = occ_labels['mask_camera'].astype(np.float32).copy()
        else:
            results['mask_camera'] = np.ones_like(semantics, dtype=np.float32)

        if results.get('rotate_bda', False):
            semantics_t = torch.from_numpy(semantics).permute(2, 0, 1)  # [16, 200, 200]
            semantics_t = rotate(semantics_t, results['rotate_bda'], interpolation=InterpolationMode.NEAREST, fill=17).permute(1, 2, 0)  # [200, 200, 16]
            results['voxel_semantics'] = semantics_t.contiguous().numpy().copy()

            mask_camera_t = torch.from_numpy(results['mask_camera']).permute(2, 0, 1)
            mask_camera_t = rotate(mask_camera_t, results['rotate_bda'], interpolation=InterpolationMode.NEAREST, fill=0).permute(1, 2, 0)
            results['mask_camera'] = mask_camera_t.contiguous().numpy().copy()

        if results.get('flip_dx', False):
            results['voxel_semantics'] = results['voxel_semantics'][::-1, ...].copy()
            results['mask_camera'] = results['mask_camera'][::-1, ...].copy()

        if results.get('flip_dy', False):
            results['voxel_semantics'] = results['voxel_semantics'][:, ::-1, ...].copy()
            results['mask_camera'] = results['mask_camera'][:, ::-1, ...].copy()

        # Add ego2occ transformation (Identity for NuScenes Occ)
        results['ego2occ'] = np.eye(4, dtype=np.float32)

        return results
