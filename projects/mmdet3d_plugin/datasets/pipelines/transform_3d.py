# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------

import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
import torch
from PIL import Image
import cv2
import hashlib


@PIPELINES.register_module()
class PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0, use_all_depth=False):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.use_all_depth = use_all_depth
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img,
                                shape = self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(img,
                                self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fix_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        eps = 1e-8
        self._pad_img(results)

        target_shapes = results['pad_shape']
        if 'gt_depth' in results:
            padded_depths = []
            for i, depth_map in enumerate(results['gt_depth']):
                target_2d_shape = target_shapes[i][:2]
                padded_d = mmcv.impad(depth_map, shape=target_2d_shape, pad_val=0)
                padded_depths.append(padded_d)
            results['gt_depth'] = padded_depths

        if 'point_mask' in results:
            padded_masks = []
            for i in range(len(results['point_mask'])):
                if self.use_all_depth:
                    padded_m = results['gt_depth'][i] > eps
                else:
                    padded_m = (results['gt_depth'][i] > eps) & (results['gt_depth'][i] < 1)
                padded_masks.append(padded_m)
            results['point_mask'] = padded_masks

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class AV2ResizeCropFlipRotImageV2():
    def __init__(self, data_aug_conf=None, multi_stamps=False, training=False):
        self.data_aug_conf = data_aug_conf
        self.min_size = 2.0
        self.multi_stamps = multi_stamps
        self.training = training

    def __call__(self, results):
        imgs = results['img']
        N = len(imgs) // 2 if self.multi_stamps else len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        ida_mats = []

        with_depthmap = ('gt_depth' in results.keys())
        if with_depthmap:
            new_depthmaps = []

        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        for i in range(N):
            H, W = imgs[i].shape[:2]
            if H > W:
                resize, resize_dims, crop = self._sample_augmentation_f(imgs[i])
                img = Image.fromarray(np.uint8(imgs[i]))
                if with_depthmap:
                    depthmap_im = Image.fromarray(results['gt_depth'][i])
                else:
                    depthmap_im = None
                img, ida_mat_f, depthmap = self._img_transform(img, resize=resize, resize_dims=resize_dims, crop=crop,
                                                               depthmap=depthmap_im)
                img_f = np.array(img).astype(np.float32)
                if 'gt_bboxes' in results.keys() and len(results['gt_bboxes']) > 0:
                    gt_bboxes = results['gt_bboxes'][i]
                    centers2d = results['centers2d'][i]
                    gt_labels = results['gt_labels'][i]
                    depths = results['depths'][i]
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                            img_f, gt_bboxes, centers2d, gt_labels, depths, resize=resize, crop=crop,
                        )
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation(img_f)
                img = Image.fromarray(np.uint8(img_f))
                img, ida_mat, depthmap = self._img_transform(img, resize=resize, resize_dims=resize_dims, crop=crop,
                                                             flip=flip, rotate=rotate, depthmap=depthmap)
                img = np.array(img).astype(np.float32)
                if 'gt_bboxes' in results.keys() and len(results['gt_bboxes']) > 0:
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(img, gt_bboxes, centers2d,
                                                                                         gt_labels, depths,
                                                                                         resize=resize, crop=crop,
                                                                                         flip=flip, )
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(img, gt_bboxes, centers2d,
                                                                                         gt_labels, depths)

                    new_gt_bboxes.append(gt_bboxes)
                    new_centers2d.append(centers2d)
                    new_gt_labels.append(gt_labels)
                    new_depths.append(depths)

                new_imgs.append(img)
                results['intrinsics'][i][:3, :3] = ida_mat @ ida_mat_f @ results['intrinsics'][i][:3, :3]
                ida_mats.append(np.array(ida_mat @ ida_mat_f))
                if with_depthmap:
                    new_depthmaps.append(np.array(depthmap))

            else:
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation(imgs[i])
                img = Image.fromarray(np.uint8(imgs[i]))
                if with_depthmap:
                    depthmap_im = Image.fromarray(results['gt_depth'][i])
                else:
                    depthmap_im = None
                img, ida_mat, depthmap = self._img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                    depthmap=depthmap_im,
                )
                img = np.array(img).astype(np.float32)
                if 'gt_bboxes' in results.keys() and len(results['gt_bboxes']) > 0:
                    gt_bboxes = results['gt_bboxes'][i]
                    centers2d = results['centers2d'][i]
                    gt_labels = results['gt_labels'][i]
                    depths = results['depths'][i]
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                            img,
                            gt_bboxes,
                            centers2d,
                            gt_labels,
                            depths,
                            resize=resize,
                            crop=crop,
                            flip=flip,
                        )
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(
                            img,
                            gt_bboxes,
                            centers2d,
                            gt_labels,
                            depths
                        )

                    new_gt_bboxes.append(gt_bboxes)
                    new_centers2d.append(centers2d)
                    new_gt_labels.append(gt_labels)
                    new_depths.append(depths)

                new_imgs.append(img)
                results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
                ida_mats.append(np.array(ida_mat))
                if with_depthmap:
                    new_depthmaps.append(np.array(depthmap))

        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['cam2img'] = results['intrinsics']
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in
                                range(len(results['extrinsics']))]

        results['img_shape'] = [img.shape for img in new_imgs]
        results['pad_shape'] = [img.shape for img in new_imgs]

        results['ida_mat'] = ida_mats  # shape N * (3, 3)

        if with_depthmap:
            results['gt_depth'] = new_depthmaps

        return results

    def _bboxes_transform(self, img, bboxes, centers2d, gt_labels, depths, resize, crop, flip=False):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)

        fH, fW = img.shape[:2]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        # normalize
        bboxes = bboxes[keep]

        centers2d = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH)
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]
        # normalize

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths

    def offline_2d_transform(self, img, bboxes, resize, crop, flip=False):
        fH, fW = img.shape[:2]
        bboxes = np.array(bboxes).reshape(-1, 6)
        bboxes[..., :4] = bboxes[..., :4] * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        return bboxes

    def _filter_invisible(self, img, bboxes, centers2d, gt_labels, depths):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)

        fH, fW = img.shape[:2]
        indices_maps = np.zeros((fH, fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip=False, rotate=0, depthmap=None):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # adjust depthmap (int32)
        if depthmap is not None:
            depthmap = depthmap.resize(resize_dims, resample=Image.NEAREST)
            depthmap = depthmap.crop(crop)
            if flip:
                depthmap = depthmap.transpose(method=Image.FLIP_LEFT_RIGHT)
            depthmap = depthmap.rotate(rotate, resample=Image.NEAREST)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat, depthmap

    def _sample_augmentation(self, img):
        # H, W = img.shape[:2]
        # fH, fW = self.data_aug_conf["final_dim"]
        # resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
        # resize_dims = (int(W * resize), int(H * resize))
        # newW, newH = resize_dims
        # crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
        # crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        # crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        # flip = False
        # if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
        #     flip = True
        # rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])

        H, W = img.shape[:2]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:

            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _sample_augmentation_f(self, img):
        H, W = img.shape[:2]
        fH, fW = W, H

        resize = np.round(((H + 50) / W), 2)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((newH - fH) / 2)
        crop_w = int((newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize, resize_dims, crop

@PIPELINES.register_module()
class ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True,
                 training=True, seq_shared_aug=False, test_resize_ratio=None,
                 test_flip=False):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible
        self.seq_shared_aug = seq_shared_aug
        self.test_resize_ratio = test_resize_ratio
        self.test_flip = test_flip

    def __call__(self, results):

        imgs = results['img']
        if 'gt_depth' in results:
            gt_depths = results['gt_depth']
        else:
            gt_depths = None
        point_masks = results.get('point_mask', None)
        eps = 1e-8
        N = len(imgs)
        new_imgs = []
        new_gt_depths = []
        new_point_mask = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        ida_mats = []
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation(results)


        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if gt_depths is not None and len(gt_depths) > 0:
                gt_depth = gt_depths[i]
                gt_depth = cv2.resize(gt_depth, resize_dims, interpolation=cv2.INTER_NEAREST)
                gt_depth = gt_depth[crop[1]:crop[3], crop[0]:crop[2]]
                # img = img.crop(crop)
                if flip:
                    gt_depth = cv2.flip(gt_depth, 1)
                    # img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
                # 没有开旋转
                # img = img.rotate(rotate)
                if point_masks is not None and len(point_masks) > 0:
                    point_mask = point_masks[i].astype(np.uint8)
                    point_mask = cv2.resize(
                        point_mask, resize_dims,
                        interpolation=cv2.INTER_NEAREST)
                    point_mask = point_mask[crop[1]:crop[3], crop[0]:crop[2]]
                    if flip:
                        point_mask = cv2.flip(point_mask, 1)
                    point_mask = point_mask.astype(bool)
                else:
                    point_mask = (gt_depth > eps) & (gt_depth < 1)
            if self.training and self.with_2d and 'gt_bboxes' in results: # sync_2d bbox labels
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers2d'][i]
                gt_labels = results['gt_labels'][i]
                depths = results['depths'][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes,
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths =  self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths)

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            if gt_depths is not None and len(gt_depths) > 0:
                new_gt_depths.append(gt_depth)
                new_point_mask.append(point_mask)
            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
            ida_mats.append(np.array(ida_mat))
        if gt_depths is not None and len(gt_depths) > 0:
            results['gt_depth'] = new_gt_depths
            results['point_mask'] = new_point_mask
        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]
        if 'ego2lidar' in results:
            results['ego2img'] = [results['lidar2img'][i] @ results['ego2lidar'][i] for i in range(len(results['ego2lidar']))]
        results['ida_mat'] = ida_mats  # shape N * (3, 3)
        results['flip'] = flip

        return results

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths,resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH)
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths


    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths



    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _get_seq_rng(self, results):
        if not self.seq_shared_aug:
            return None
        scene_token = results.get('scene_token', None)
        if scene_token is None:
            return None
        seed_src = f"{self.__class__.__name__}:{scene_token}"
        seed = int(hashlib.md5(seed_src.encode('utf-8')).hexdigest()[:8], 16)
        return np.random.RandomState(seed)

    def _sample_augmentation(self, results=None):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        rng = self._get_seq_rng(results) if results is not None else None
        uniform = rng.uniform if rng is not None else np.random.uniform
        randint = rng.randint if rng is not None else np.random.randint
        if self.training:
            resize = uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and randint(0, 2):
                flip = True
            rotate = uniform(*self.data_aug_conf["rot_lim"])
        else:
            if self.test_resize_ratio is not None:
                resize = self.test_resize_ratio
            else:
                resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.test_flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

@PIPELINES.register_module()
class WaymoResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    def __call__(self, results):

        imgs = results['img']
        if 'gt_depth' in results:
            gt_depths = results['gt_depth']
        else:
            gt_depths = None
        eps = 1e-8
        N = len(imgs)
        new_imgs = []
        new_gt_depths = []
        new_point_mask = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        ida_mats = []
        # 确保不支持旋转，与原代码保持一致
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        for i in range(N):
            img_np = np.uint8(imgs[i])
            # 从图像的实际尺寸获取H和W
            H, W, _ = img_np.shape

            # 传入当前图像的H和W来获取增强参数
            resize, resize_dims, crop, flip, rotate = self._sample_augmentation(H, W)

            img = Image.fromarray(img_np)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if gt_depths is not None and len(gt_depths) > 0:
                gt_depth = gt_depths[i]
                gt_depth = cv2.resize(gt_depth, resize_dims, interpolation=cv2.INTER_NEAREST)
                gt_depth = gt_depth[crop[1]:crop[3], crop[0]:crop[2]]
                if flip:
                    gt_depth = cv2.flip(gt_depth, 1)

            if self.training and self.with_2d: # sync_2d bbox labels
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers2d'][i]
                gt_labels = results['gt_labels'][i]
                depths = results['depths'][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes,
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths)

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            if gt_depths is not None and len(gt_depths) > 0:
                new_gt_depths.append(gt_depth)
                new_point_mask.append((gt_depth > eps) & (gt_depth < 1))
            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
            ida_mats.append(np.array(ida_mat))

        if gt_depths is not None and len(gt_depths) > 0:
            results['gt_depth'] = new_gt_depths
            results['point_mask'] = new_point_mask

        if self.training and self.with_2d:
            results['gt_bboxes'] = new_gt_bboxes
            results['centers2d'] = new_centers2d
            results['gt_labels'] = new_gt_labels
            results['depths'] = new_depths

        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]
        results['ida_mat'] = ida_mats
        # print(f"resize image, scene: {results['scene_token']}, time: {results['timestamp']}")

        return results

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths,resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH)
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths

    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            # 保证v1, v2, u1, u2在界内
            v1, v2 = max(0, v1), min(fH, v2)
            u1, u2 = max(0, u1), min(fW, u2)
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)

        # 确保索引不越界
        valid_indices = indices_res[indices_res < len(bboxes)]

        bboxes = bboxes[valid_indices]
        depths = depths[valid_indices]
        centers2d = centers2d[valid_indices]
        gt_labels = gt_labels[valid_indices]

        return bboxes, centers2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    # 核心改动：接收H和W作为参数
    def _sample_augmentation(self, H, W):
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            # 验证阶段，计算resize比例时同时考虑H和W
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

@PIPELINES.register_module()
class GlobalRotScaleTransImage():
    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
        seq_shared_aug=False,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training
        self.seq_shared_aug = seq_shared_aug

    def _get_seq_rng(self, results):
        if not self.seq_shared_aug:
            return None
        scene_token = results.get('scene_token', None)
        if scene_token is None:
            return None
        seed_src = f"{self.__class__.__name__}:{scene_token}"
        seed = int(hashlib.md5(seed_src.encode('utf-8')).hexdigest()[:8], 16)
        return np.random.RandomState(seed)

    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)
        rng = self._get_seq_rng(results)
        uniform = rng.uniform if rng is not None else np.random.uniform
        normal = rng.normal if rng is not None else np.random.normal

        rot_angle = uniform(*self.rot_range)
        scale_ratio = uniform(*self.scale_ratio_range)
        trans = normal(scale=translation_std, size=3).T

        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_sin = torch.sin(torch.tensor(rot_angle))
        rot_mat = torch.tensor([
            [ rot_cos,  rot_sin, 0., 0.],
            [-rot_sin,  rot_cos, 0., 0.],
            [ 0.,       0.,      1., 0.],
            [ 0.,       0.,      0., 1.],
        ], dtype=torch.float32)
        scale_mat = torch.tensor([
            [scale_ratio, 0.,           0.,           0.],
            [0.,          scale_ratio,  0.,           0.],
            [0.,          0.,           scale_ratio,  0.],
            [0.,          0.,           0.,           1.],
        ], dtype=torch.float32)
        trans_mat = torch.eye(4, dtype=torch.float32)
        trans_mat[:3, 3] = torch.from_numpy(trans.astype(np.float32)).reshape(1, 3)

        bda_mat = trans_mat @ scale_mat @ rot_mat

        results['bda_mat'] = bda_mat.numpy()            # 4x4 前向整体增广
        results['rotate_bda'] = -float(np.degrees(rot_angle))       # 角度
        results['scale_bda']  = float(scale_ratio)
        results['trans_bda']  = trans.astype(np.float32)  # (3,)
        results['flip_dx'] = False
        results['flip_dy'] = False
        # results['bda_mat_inv'] = bda_mat_inv.numpy()

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        if 'gt_bboxes_3d' in results:
            results["gt_bboxes_3d"].rotate(
                np.array(rot_angle)
            )

        # random scale
        self._scale_xyz(results, scale_ratio)
        if 'gt_bboxes_3d' in results:
            results["gt_bboxes_3d"].scale(scale_ratio)

        #random translate
        self._trans_xyz(results, trans)
        if 'gt_bboxes_3d' in results:
            results["gt_bboxes_3d"].translate(trans)

        if 'ego2lidar' in results:
            results['ego2img'] = [
                (results['lidar2img'][i] @ results['ego2lidar'][i]).astype(np.float32)
                for i in range(len(results['lidar2img']))
            ]

        return results

    def _trans_xyz(self, results, trans):
        trans_mat = torch.eye(4, 4)
        trans_mat[:3, -1] = torch.from_numpy(trans).reshape(1, 3)
        trans_mat_inv = torch.inverse(trans_mat)
        num_view = len(results["lidar2img"])
        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ trans_mat_inv).numpy()
        results['ego_pose_inv'] = (trans_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()
        self._update_ego2global(results, trans_mat_inv)

        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ trans_mat_inv).numpy()

        # for i in range(len(results['ego2img'])):
        #     results['ego2img'][i] = (torch.tensor(results['ego2img'][i]).float() @ trans_mat_inv).numpy()


    def _rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, rot_sin, 0, 0], [-rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ rot_mat_inv).numpy()
        results['ego_pose_inv'] = (rot_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()
        self._update_ego2global(results, rot_mat_inv)
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
        # for i in range(len(results['ego2img'])):
        #     results['ego2img'][i] = (torch.tensor(results['ego2img'][i]).float() @ rot_mat_inv).numpy()

    def _scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = torch.inverse(scale_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ scale_mat_inv).numpy()
        results['ego_pose_inv'] = (scale_mat @ torch.tensor(results["ego_pose_inv"]).float()).numpy()
        self._update_ego2global(results, scale_mat_inv)
        if 'gt_depth' in results:
            results['gt_depth'] = [gt_dep * scale_ratio for gt_dep in results['gt_depth']]

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv).numpy()
        # for i in range(len(results['ego2img'])):
        #     results['ego2img'][i] = (torch.tensor(results['ego2img'][i]).float() @ scale_mat_inv).numpy()

    def _update_ego2global(self, results, transform_inv):
        if 'ego2global' not in results:
            return

        ego2global = results['ego2global']
        if isinstance(ego2global, (list, tuple)):
            results['ego2global'] = [
                (torch.tensor(mat).float() @ transform_inv).numpy()
                for mat in ego2global
            ]
            return

        ego2global_tensor = torch.tensor(ego2global).float()
        if ego2global_tensor.ndim == 3:
            transform_inv = transform_inv.unsqueeze(0).expand_as(ego2global_tensor)
        results['ego2global'] = (ego2global_tensor @ transform_inv).numpy()
