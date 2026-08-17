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
from mmdet.datasets import DATASETS
from mmdet3d.datasets import KittiDataset
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
import mmcv
import random
import math
import os
import cv2
import copy
import scipy.io
from datetime import datetime

@DATASETS.register_module()
class CustomKittiDataset(KittiDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_length=30, num_frame_losses=1, queue_length=8, random_length=0, queue_interval=1, target_hz=None, vggt_mode=False, vggt_scale=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.queue_interval = max(1, int(queue_interval))
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        self.vggt_mode = vggt_mode
        self.vggt_scale = vggt_scale
        self.target_hz = target_hz
        self.target_interval = None if target_hz is None else 1.0 / float(target_hz)
        self._timestamp_cache = {}
        if seq_mode:
            self.num_frame_losses = num_frame_losses
            self.queue_length = queue_length
            self.seq_length = seq_length
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

        if self.target_interval is not None:
            self._build_scene_indices()

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        with open(self.data_root + 'mapping/train_rand.txt', 'r') as f:
            rand_info = f.readlines()[0].strip().split(',')

        with open(self.data_root + 'mapping/train_mapping.txt', 'r') as f:
            mapping_lines = f.readlines()
        mapping_dict = {}
        for i, line in enumerate(mapping_lines):
            line = line.strip().split(' ')
            id_info = (rand_info.index(str(i+1)), int(line[2]))
            if mapping_dict.get(line[1]) is None:
                mapping_dict[line[1]] = [id_info]
            else:
                mapping_dict[line[1]].append(id_info)
        data_mapping = {}
        for idx in range(len(self.data_infos)):
            data_mapping[self.data_infos[idx]['image']['image_idx']] = idx

        curr_sequence = -1
        new_data_infos = []
        for scene in mapping_dict:
            frame_ids = mapping_dict[scene]
            new_seq = True
            frame_length = 0
            real_scene = scene
            scene_num = 0
            for frame_id, seq_id in frame_ids:
                if data_mapping.get(frame_id) is not None:
                    frame_length += 1
                    if new_seq:
                        curr_sequence += 1
                        new_seq = False
                    if frame_length > self.seq_length:
                        frame_length = 1
                        curr_sequence += 1
                        scene_num += 1
                        real_scene = scene + '_' + str(scene_num)
                    res.append(curr_sequence)
                    self.data_infos[data_mapping[frame_id]]['scene_token'] = real_scene
                    self.data_infos[data_mapping[frame_id]]['raw_seq_id'] = seq_id
                    new_data_infos.append(self.data_infos[data_mapping[frame_id]])

        self.flag = np.array(res, dtype=np.int64)
        self.data_infos = new_data_infos
        # if self.seq_split_num != 1:
        #     if self.seq_split_num == 'all':
        #         self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
        #     else:
        #         bin_counts = np.bincount(self.flag)
        #         new_flags = []
        #         curr_new_flag = 0
        #         for curr_flag in range(len(bin_counts)):
        #             curr_sequence_length = np.array(
        #                 list(range(0,
        #                         bin_counts[curr_flag],
        #                         math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
        #                 + [bin_counts[curr_flag]])

        #             for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
        #                 for _ in range(sub_seq_idx):
        #                     new_flags.append(curr_new_flag)
        #                 curr_new_flag += 1

        #         assert len(new_flags) == len(self.flag)
        #         assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
        #         self.flag = np.array(new_flags, dtype=np.int64)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return mmcv.load(ann_file, file_format='pkl')

    @staticmethod
    def _parse_kitti_timestamp(timestamp_str):
        dot_index = timestamp_str.find('.')
        if dot_index != -1 and len(timestamp_str) - dot_index - 1 > 6:
            timestamp_str = timestamp_str[:dot_index + 1 + 6]
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f').timestamp()

    @staticmethod
    def _base_scene_token(scene_token):
        if 'sync_' in scene_token:
            return scene_token.split('sync')[0] + 'sync'
        return scene_token

    def _get_timestamp_seconds(self, index):
        if index in self._timestamp_cache:
            return self._timestamp_cache[index]

        info = self.data_infos[index]
        scene_token = info['scene_token']
        real_scene_token = self._base_scene_token(scene_token)
        timestamp_file = os.path.join(
            self.data_root, 'kitti_processed', real_scene_token,
            'image_02_timestamps.txt')
        with open(timestamp_file, 'r') as f:
            timestamp_str = f.readlines()[info['raw_seq_id']].strip()
        timestamp = self._parse_kitti_timestamp(timestamp_str)
        self._timestamp_cache[index] = timestamp
        return timestamp

    def _build_scene_indices(self):
        self.scene_to_indices = {}
        for idx, info in enumerate(self.data_infos):
            scene_token = info['scene_token']
            timestamp = self._get_timestamp_seconds(idx)
            self.scene_to_indices.setdefault(scene_token, []).append((idx, timestamp))

    def _get_target_sequence_indices(self, index):
        if self.target_interval is None:
            return list(range(
                index - (self.queue_length + self.random_length - 1) * self.queue_interval,
                index,
                self.queue_interval))

        info = self.data_infos[index]
        scene_token = info['scene_token']
        scene_indices = self.scene_to_indices[scene_token]
        scene_positions = [idx for idx, _ in scene_indices]
        scene_timestamps = np.asarray([timestamp for _, timestamp in scene_indices], dtype=np.float64)
        curr_pos = scene_positions.index(index)
        curr_time = scene_timestamps[curr_pos]
        indices = []
        last_valid_index = index

        for offset in range(self.queue_length):
            target_time = curr_time - offset * self.target_interval
            pos = int(np.searchsorted(scene_timestamps, target_time, side='right') - 1)
            pos = min(max(pos, 0), curr_pos)
            if pos >= 0:
                last_valid_index = scene_indices[pos][0]
            indices.append(last_valid_index)
        return indices

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(
            index - (self.queue_length + self.random_length - 1) * self.queue_interval,
            index,
            self.queue_interval))
        # print(index_list)
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        index_list = reversed(index_list)
        prev_scene_token = None
        prev_example = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)

            if self.vggt_mode:
                if prev_scene_token == None:
                    prev_scene_token = input_dict['scene_token']
                if input_dict['scene_token'] != prev_scene_token:
                    queue.append(prev_example)
                    continue

            if not self.seq_mode: # for sliding window only
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            prev_example = example.copy()

            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[k] is None or ~(queue[k]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        queue = []
        index_list = self._get_target_sequence_indices(index)
        if self.target_interval is None:
            input_dict = self.get_data_info(index)
            random.shuffle(index_list)
            index_list = sorted(index_list[self.random_length:])
            index_list.append(index)
            index_list = reversed(index_list)
        prev_scene_token = None
        prev_example = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)

            if self.vggt_mode:
                if prev_scene_token == None:
                    prev_scene_token = input_dict['scene_token']
                if input_dict['scene_token'] != prev_scene_token:
                    queue.append(prev_example)
                    continue
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            prev_example = example.copy()

            queue.append(example)
        return self.union2one_test(queue)

    def union2one(self, queue):
        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
                if key == 'gt_bboxes_3d':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)

        queue = queue[-1]
        return queue

    def union2one_test(self, queue):
        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key][0].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key][0].data for each in queue], cpu_only=True)

        queue = queue[-1]
        return queue

    def remove_classes(self, ann_info, classes_to_remove=['DontCare', 'Misc']):
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x not in classes_to_remove
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations



    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                    0, 1, 2 represent xxxxx respectively.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        rect = info['calib']['R0_rect'].astype(np.float64)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float64)

        if 'plane' in info:
            # convert ground plane to velodyne coordinates
            reverse = np.linalg.inv(rect @ Trv2c)

            (plane_norm_cam,
             plane_off_cam) = (info['plane'][:3],
                               -info['plane'][:3] * info['plane'][3])
            plane_norm_lidar = \
                (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
            plane_off_lidar = (
                reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                reverse[:3, 3])
            plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
            plane_lidar[:3] = plane_norm_lidar
            plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
        else:
            plane_lidar = None

        difficulty = info['annos']['difficulty']
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_classes(annos, classes_to_remove=['DontCare', 'Misc', 'Truck', 'Van', 'Person_sitting', 'Tram'])
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float64)
        def roty(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[ c, 0, s],
                            [ 0, 1, 0],
                            [-s, 0, c]])

        def corners_3d_box(h, w, l, ry):
            x_corners = [l/2, l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
            y_corners = [0,   0,    0,    0,   -h,   -h,   -h,   -h]
            z_corners = [w/2,-w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
            corners = np.array([x_corners, y_corners, z_corners], dtype=np.float64)
            return roty(ry) @ corners

        def filter_3d_to_2d(annos, P2, img_shape, margin=5):
            """
            参数
            ----
            annos    : 单帧 KITTI 标注 dict
            P2       : 3×4 相机投影矩阵
            img_shape: (H, W)  图像高宽
            margin   : 允许框超出图像边界多少像素，仍可被保留

            返回
            ----
            keep_boxes   : (K,4)  float64  真正在图内的 2-D 框  [x1,y1,x2,y2]
            keep_centers : (K,2)  float64  物体在图像上的中心像素 [u,v]
            keep_depths  : (K,)   float64  中心到相机的深度（Z_cam）
            keep_idx     : (K,)   int64    对应 annos 里的索引，方便后续对齐
            """
            H, W = img_shape
            loc = annos['location']
            dims = annos['dimensions']
            rots = annos['rotation_y']
            N = len(loc)

            keep_boxes, keep_centers, keep_depths, keep_idx = [], [], [], []
            for i in range(N):
                l, h, w = dims[i]
                ry = rots[i]
                # 1. 3-D 角点
                corners = corners_3d_box(h, w, l, ry) + loc[i].reshape(3, 1)  # 3×8
                # 2. 投影
                corners_homo = np.vstack([corners, np.ones((1, 8))])          # 4×8
                pts_2d_homo = P2 @ corners_homo                               # 3×8
                pts_2d = pts_2d_homo[:2] / (pts_2d_homo[2] + 1e-6)            # 2×8

                # 3. 2-D 框
                x_min, y_min = pts_2d.min(axis=1)
                x_max, y_max = pts_2d.max(axis=1)

                # 4. 中心像素 & 深度（直接用 3-D 中心）
                center_3d = loc[i].reshape(3, 1)                            # 3×1
                center_2d_homo = P2 @ np.vstack([center_3d, 1])             # 3×1
                center_2d = center_2d_homo[:2, 0] / center_2d_homo[2]          # 2
                depth = center_3d[2, 0]                                     # Z_cam

                # 5. 筛选：框必须至少部分在图像内
                if (x_max < -margin) or (x_min > W + margin) or \
                (y_max < -margin) or (y_min > H + margin):
                    continue

                keep_boxes.append([x_min, y_min, x_max, y_max])
                keep_centers.append(center_2d)
                keep_depths.append(depth)
                keep_idx.append(i)

            if len(keep_boxes) == 0:          # 防止空列表
                return (np.empty((0, 4), dtype=np.float32),
                        np.empty((0, 2), dtype=np.float32),
                        np.empty((0,), dtype=np.float32),
                        np.empty((0,), dtype=np.int32))

            return (np.array(keep_boxes, dtype=np.float32),
                    np.array(keep_centers, dtype=np.float32),
                    np.array(keep_depths, dtype=np.float32),
                    np.array(keep_idx, dtype=np.int32))


        # convert gt_bboxes_3d to velodyne coordinates
        nus_lidar_to_kitti_lidar = np.array([[0, 1, 0, 0],
                                            [-1,  0, 0, 0],
                                            [0,  0, 1, 0],
                                            [0,  0, 0, 1]])
        yaw = gt_bboxes_3d[:, 6:7]
        new_yaw = yaw + np.pi / 2
        new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi
        gt_bboxes_3d = np.concatenate((gt_bboxes_3d[:, :6], new_yaw), axis=1)
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c @ nus_lidar_to_kitti_lidar))
        # gt_bboxes = annos['bbox']

        # selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        # gt_bboxes = gt_bboxes[selected].astype('float32')
        # gt_names = gt_names[selected]
        bboxes_cam2, centers_cam2, depths_cam2, idx_cam2 = filter_3d_to_2d(annos, info['calib']['P2'], info['image']['image_shape'])
        bboxes_cam3, centers_cam3, depths_cam3, idx_cam3 = filter_3d_to_2d(annos, info['calib']['P3'], info['image']['image_shape'])
        gt_bboxes = [bboxes_cam2, bboxes_cam3]
        centers_cam = [centers_cam2, centers_cam3]
        depths = [depths_cam2, depths_cam3]

        # KITTI2NU = {
        #     'Car'           : 'car',
        #     # 'Van'           : 'car',
        #     # 'Truck'         : 'truck',
        #     'Pedestrian'    : 'pedestrian',
        #     # 'Person_sitting': 'pedestrian',
        #     'Cyclist'       : 'bicycle',
        #     # 'Tram'          : 'bus',
        #     # Misc/DontCare 直接丢弃
        # }
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_cam_lables = [gt_labels[idx_cam2], gt_labels[idx_cam3]]
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            centers2d=centers_cam,
            depths=depths,
            labels=gt_cam_lables,
            gt_names=gt_names,
            plane=plane_lidar,
            difficulty=difficulty)
        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        input_dict= {}
        input_dict['dataset'] = 'kitti'
        sample_idx = info['image']['image_idx']
        input_dict['sample_idx'] = sample_idx
        input_dict['pts_filename'] = self._get_pts_filename(sample_idx)
        input_dict['sweeps'] = []
        input_dict['prev_idx'] = ''
        input_dict['next_idx'] = ''
        input_dict['frame_idx'] = info['image']['image_idx']

        scene_token = info['scene_token']
        if 'sync_' in scene_token:
            real_scene_token = scene_token.split('sync')[0] + 'sync'
        else:
            real_scene_token = scene_token
        cam2_timestamp_file = os.path.join(self.data_root, 'kitti_processed', real_scene_token, 'image_02_timestamps.txt')
        cam3_timestamp_file = cam2_timestamp_file.replace("image_02", "image_03")
        def convert_kitti_timestamp_to_nuscenes(kitti_timestamp_str):
            dot_index = kitti_timestamp_str.find('.')
            if dot_index != -1 and len(kitti_timestamp_str) - dot_index - 1 > 6:
                kitti_timestamp_str = kitti_timestamp_str[:dot_index + 1 + 6]
            kitti_format = "%Y-%m-%d %H:%M:%S.%f"
            dt_object = datetime.strptime(kitti_timestamp_str, kitti_format)
            unix_seconds = dt_object.timestamp()
            nuscenes_timestamp = int(unix_seconds * 1e6)
            return nuscenes_timestamp

        with open(cam2_timestamp_file, 'r') as f:
            cam2_timestamps = f.readlines()
        with open(cam3_timestamp_file, 'r') as f:
            cam3_timestamps = f.readlines()

        cam2_timestamp = convert_kitti_timestamp_to_nuscenes(cam2_timestamps[info['raw_seq_id']].strip())
        cam3_timestamp = convert_kitti_timestamp_to_nuscenes(cam3_timestamps[info['raw_seq_id']].strip())
        input_dict['img_timestamp'] = [cam2_timestamp / 1e6, cam2_timestamp / 1e6]
        input_dict['timestamp'] = cam2_timestamp / 1e6

        cam2_img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])
        cam3_img_filename = cam2_img_filename.replace("image_2", "image_3")
        input_dict['img_filename'] = [cam2_img_filename, cam3_img_filename]
        input_dict['scene_token'] = info['scene_token']
        input_dict['depth_filename'] = [cam2_img_filename.replace("image_2", "depth_npy/image_2").replace('.png', '.npy'), cam3_img_filename.replace("image_3", "depth_npy/image_3").replace('.png', '.npy')]
        rect = info['calib']['R0_rect'].astype(np.float64)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float64)
        nus_lidar_to_kitti_lidar = np.array([[0, 1, 0, 0],
                                            [-1,  0, 0, 0],
                                            [0,  0, 1, 0],
                                            [0,  0, 0, 1]])
        P2 = info['calib']['P2'].astype(np.float64)
        P3 = info['calib']['P3'].astype(np.float64)
        cam2_lidar2img = P2 @ rect @ Trv2c @ nus_lidar_to_kitti_lidar
        cam3_lidar2img = P3 @ rect @ Trv2c @ nus_lidar_to_kitti_lidar
        input_dict['lidar2img'] = [cam2_lidar2img, cam3_lidar2img]

        pose_file = os.path.join(self.data_root, 'kitti_processed', real_scene_token, 'oxts/pose.mat')
        pose_data = scipy.io.loadmat(pose_file)
        imu2global = pose_data['pose_mat'][info['raw_seq_id']]
        ego_pose = imu2global @ invert_matrix_egopose_numpy(info['calib']['Tr_imu_to_velo']) @ nus_lidar_to_kitti_lidar
        input_dict['ego_pose'] = ego_pose
        input_dict['ego_pose_inv'] = invert_matrix_egopose_numpy(ego_pose)

        def compute_T_cam0_to_camX(P):
            """
            从投影矩阵P中计算相机X相对于相机0的变换矩阵T_cam0_to_camX
            参数:
                P: 3x4的投影矩阵（如P2或P3）
            返回:
                T: 4x4的齐次变换矩阵
            """
            # 步骤1: 从P中提取内参矩阵K（3x3）
            K = P[:3, :3].copy()  # P的前3x3部分为K·R，直接取为K（忽略旋转的微小影响）

            # 步骤2: 计算K的逆矩阵
            K_inv = np.linalg.inv(K)

            # 步骤3: 计算旋转矩阵R（3x3）: R = K_inv · (P的前3x3)
            R = K_inv @ P[:3, :3]  # 矩阵乘法用@运算符

            # 步骤4: 计算平移向量t（3x1）: t = K_inv · (P的第4列)
            t = K_inv @ P[:3, 3:]  # P[:3, 3:] 提取第4列（保持列向量形状）

            # 步骤5: 组合为4x4齐次变换矩阵
            T = np.eye(4)  # 初始化4x4单位矩阵
            T[:3, :3] = R  # 上左3x3为旋转矩阵
            T[:3, 3] = t.flatten()  # 上右3x1为平移向量（展平为1D数组）
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = K

            return T, intrinsics
        cam0_to_cam2, cam2_intrinsics = compute_T_cam0_to_camX(P2)
        cam0_to_cam3, cam3_intrinsics = compute_T_cam0_to_camX(P3)
        input_dict['intrinsics'] = [cam2_intrinsics, cam3_intrinsics]
        cam2_extrinsics = cam0_to_cam2 @ rect @ Trv2c @ nus_lidar_to_kitti_lidar
        cam3_extrinsics = cam0_to_cam3 @ rect @ Trv2c @ nus_lidar_to_kitti_lidar
        input_dict['extrinsics'] = [cam2_extrinsics, cam3_extrinsics]
        input_dict['cam_extrinsics_global'] = [cam2_extrinsics, cam3_extrinsics]
        if not self.test_mode: # for seq_mode
            prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
        else:
            prev_exists = None
        input_dict['prev_exists'] = prev_exists

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        vis = False
        if vis:
            import matplotlib.pyplot as plt
            # Get 3D boxes and labels
            gt_bboxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
            gt_labels_3d = annos['gt_labels_3d']

            # Get 2D boxes and labels
            bboxes2d = annos['bboxes']
            labels2d = annos['labels']

            # Create a color map for classes
            colors = plt.cm.get_cmap('hsv', len(self.CLASSES))

            # --- Visualize and save 2D bounding boxes ---
            for img_idx, img_path in enumerate(input_dict['img_filename']): # Process first two images
                img_2d = cv2.imread(img_path)
                if len(bboxes2d) > img_idx and len(labels2d) > img_idx:
                    for bbox, label in zip(bboxes2d[img_idx], labels2d[img_idx]):
                        color = tuple(c * 255 for c in colors(label)[:3])
                        cv2.rectangle(img_2d, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        class_name = self.CLASSES[label]
                        cv2.putText(img_2d, class_name, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                output_path_2d = f'result_2d_{img_idx}.jpg'
                cv2.imwrite(output_path_2d, img_2d)
                print(f"Saved 2D visualization to {output_path_2d}")


            # --- Visualize and save 3D bounding boxes ---
            for img_idx, (img_path, lidar2img) in enumerate(zip(input_dict['img_filename'], input_dict['lidar2img'])):
                img_3d = cv2.imread(img_path)
                for i in range(len(gt_bboxes_3d)):
                    bbox_3d = gt_bboxes_3d[i]
                    label_3d = gt_labels_3d[i]

                    # ==================== INICIO DE LA CORRECCIÓN ====================
                    # El vector bbox_3d es (x, y, z, w, l, h, yaw)
                    # Si 'z' es el centro de la base, lo ajustamos al centro geométrico
                    # sumando la mitad de la altura (h/2).
                    # La altura 'h' está en el índice 5. La 'z' está en el índice 2.
                    corrected_bbox_3d = bbox_3d.copy()
                    corrected_bbox_3d[2] += corrected_bbox_3d[5] / 2.0
                    # ===================== FIN DE LA CORRECCIÓN ======================

                    # Usamos la caja corregida para crear el objeto y obtener las esquinas
                    box_lidar = LiDARInstance3DBoxes(np.array([corrected_bbox_3d]), box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5))
                    corners_lidar = box_lidar.corners[0].numpy()

                    corners_lidar_homogeneous = np.concatenate([corners_lidar, np.ones((8, 1))], axis=1)
                    corners_img_homogeneous = corners_lidar_homogeneous @ lidar2img.T

                    depth = corners_img_homogeneous[:, 2]
                    if np.any(depth < 0.01): # Small threshold to avoid points at camera origin
                        continue

                    corners_img = corners_img_homogeneous[:, :2] / corners_img_homogeneous[:, 2, None]

                    color = tuple(c * 255 for c in colors(label_3d)[:3])
                    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                             [0, 4], [1, 5], [2, 6], [3, 7]]

                    for edge in edges:
                        p1 = tuple(corners_img[edge[0]].astype(int))
                        p2 = tuple(corners_img[edge[1]].astype(int))
                        cv2.line(img_3d, p1, p2, color, 1)

                    class_name = self.CLASSES[label_3d]
                    cv2.putText(img_3d, class_name, (int(corners_img[0][0]), int(corners_img[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                output_path_3d = f'result_3d_{img_idx}.jpg'
                cv2.imwrite(output_path_3d, img_3d)
                print(f"Saved 3D visualization to {output_path_3d}")
        # breakpoint()

        return input_dict


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            # print(f"kitti, idx: {idx}")
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.
        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag != -1)[0]
        return np.random.choice(pool)

    def transform_result_from_nuscenes(self, results):
        """Transform results from NuScenes to KITTI format.

        Args:
        results (list[dict]): Results in NuScenes format.

        Returns:
        list[dict]: Results in KITTI format.
        """
        new_results = []
        for result in results:
            new_result = copy.deepcopy(result)
            bbox_3d = result['pts_bbox']['boxes_3d'].tensor
            x, y, z = bbox_3d[:, 0:1], bbox_3d[:, 1:2], bbox_3d[:, 2:3]
            yaw = bbox_3d[:, 6:7]
            new_x, new_y, new_z = y, -x, z
            new_yaw = yaw - np.pi / 2
            new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi
            bbox_3d = torch.cat([new_x, new_y, new_z, bbox_3d[:, 3:6], new_yaw, bbox_3d[:, 6:]], dim=1)
            new_result['pts_bbox']['boxes_3d'] = LiDARInstance3DBoxes(bbox_3d, box_dim=bbox_3d.shape[-1])
            new_results.append(new_result)
        return new_results

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 jsonfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 **kwargs):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        from mmcv.utils import print_log
        eval_results = {}

        if not isinstance(metric, list):
            metric = [metric]
        if 'depth' in metric:
            depth_metric_config = kwargs.get('depth_eval', {})
            depth_results = self._evaluate_depth(results, depth_metric_config, logger)
            eval_results.update(depth_results)

        if 'camera' in metric:
            camera_metric_config = kwargs.get('camera_eval', {})
            camera_results = self._evaluate_camera(results, camera_metric_config, logger)
            eval_results.update(camera_results)
        if 'bbox' in metric:
            results = self.transform_result_from_nuscenes(results)
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
            from mmdet3d.core.evaluation import kitti_eval
            from mmcv.utils import print_log
            gt_annos = [info['annos'] for info in self.data_infos]

            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    eval_types = ['bbox', 'bev', '3d']
                    if 'img' in name:
                        eval_types = ['bbox']
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                    print_log(
                        f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                if metric == 'img_bbox':
                    ap_result_str, ap_dict = kitti_eval(
                        gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
                else:
                    ap_result_str, ap_dict = kitti_eval(gt_annos, result_files,
                                                        self.CLASSES)
                print_log('\n' + ap_result_str, logger=logger)

            if tmp_dir is not None:
                tmp_dir.cleanup()
            if show or out_dir:
                self.show(results, out_dir, show=show, pipeline=pipeline)
            eval_results.update(ap_dict)
        return eval_results


    def _evaluate_camera(self, results, camera_config, logger):
        from ..models.utils.pose_enc import pose_encoding_to_extri_intri
        from ..models.utils.geometry import closed_form_inverse_se3
        from ..models.utils.rotation import mat_to_quat

        def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
            """
            Calculate translation angle error between ground truth and predicted translations.

            Args:
                tvec_gt: Ground truth translation vectors
                tvec_pred: Predicted translation vectors
                batch_size: Batch size for reshaping the result
                ambiguity: Whether to handle direction ambiguity

            Returns:
                Translation angle error in degrees
            """
            rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
            rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

            if ambiguity:
                rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

            if batch_size is not None:
                rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

            return rel_tangle_deg



        def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
            """
            Normalize the translation vectors and compute the angle between them.

            Args:
                t_gt: Ground truth translation vectors
                t: Predicted translation vectors
                eps: Small value to avoid division by zero
                default_err: Default error value for invalid cases

            Returns:
                Angular error between translation vectors in radians
            """
            t_norm = torch.norm(t, dim=1, keepdim=True)
            t = t / (t_norm + eps)

            t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
            t_gt = t_gt / (t_gt_norm + eps)

            loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
            err_t = torch.acos(torch.sqrt(1 - loss_t))

            err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
            return err_t

        def build_pair_index(N, B=1):
            """
            Build indices for all possible pairs of frames.

            Args:
                N: Number of frames
                B: Batch size

            Returns:
                i1, i2: Indices for all possible pairs
            """
            i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
            i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
            return i1, i2

        def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
            """
            Calculate rotation angle error between ground truth and predicted rotations.

            Args:
                rot_gt: Ground truth rotation matrices
                rot_pred: Predicted rotation matrices
                batch_size: Batch size for reshaping the result
                eps: Small value to avoid numerical issues

            Returns:
                Rotation angle error in degrees
            """
            q_pred = mat_to_quat(rot_pred)
            q_gt = mat_to_quat(rot_gt)

            loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
            err_q = torch.arccos(1 - 2 * loss_q)

            rel_rangle_deg = err_q * 180 / np.pi

            if batch_size is not None:
                rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

            return rel_rangle_deg

        def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
            """
            Compute rotation and translation errors between predicted and ground truth poses.
            This function assumes the input poses are world-to-camera (w2c) transformations.

            Args:
                pred_se3: Predicted SE(3) transformations (w2c), shape (N, 4, 4)
                gt_se3: Ground truth SE(3) transformations (w2c), shape (N, 4, 4)
                num_frames: Number of frames (N)

            Returns:
                Rotation and translation angle errors in degrees
            """
            pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

            relative_pose_gt = gt_se3[pair_idx_i1].bmm(
                closed_form_inverse_se3(gt_se3[pair_idx_i2])
            )
            relative_pose_pred = pred_se3[pair_idx_i1].bmm(
                closed_form_inverse_se3(pred_se3[pair_idx_i2])
            )

            rel_rangle_deg = rotation_angle(
                relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
            )
            rel_tangle_deg = translation_angle(
                relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
            )

            return rel_rangle_deg, rel_tangle_deg
        camera_T_metrics = []
        camera_R_metrics = []
        for result in results:
            cam_pose_pred = result['cam_pose_pred']
            extri, intri = pose_encoding_to_extri_intri(cam_pose_pred, result['img_hw'])
            cam_extrinsics = result['cam_extrinsics'][:, :, :3, :]
            num_view = extri.shape[0]
            for v in range(num_view):
                pred_extrinsic = extri[v]
                gt_extrinsic = cam_extrinsics[v]
                num_frames = pred_extrinsic.shape[0]
                with torch.cuda.amp.autocast(dtype=torch.float64):
                    add_row = torch.tensor([0, 0, 0, 1], device=pred_extrinsic.device).expand(pred_extrinsic.size(0), 1, 4)

                    pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
                    gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

                    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)
                    Racc_5 = (rel_rangle_deg < 5).float().mean().item()
                    Tacc_5 = (rel_tangle_deg < 5).float().mean().item()
                    # print(f"R_ACC@5: {Racc_5:.4f}")
                    # print(f"T_ACC@5: {Tacc_5:.4f}")
                    camera_R_metrics.append(Racc_5)
                    camera_T_metrics.append(Tacc_5)

        rError = np.array(camera_R_metrics)
        tError = np.array(camera_T_metrics)

        def calculate_auc_np(r_error, t_error, max_threshold=30):
            """
            Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

            Args:
                r_error: numpy array representing R error values (Degree)
                t_error: numpy array representing T error values (Degree)
                max_threshold: Maximum threshold value for binning the histogram

            Returns:
                AUC value and the normalized histogram
            """
            error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
            max_errors = np.max(error_matrix, axis=1)
            bins = np.arange(max_threshold + 1)
            histogram, _ = np.histogram(max_errors, bins=bins)
            num_pairs = float(len(max_errors))
            normalized_histogram = histogram.astype(float) / num_pairs
            return np.mean(np.cumsum(normalized_histogram)), normalized_histogram

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)

        camera_results = {
            # "rError": rError,
            # "tError": tError,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3
        }

        return camera_results


    def _evaluate_depth(self, results, depth_config, logger):
        """
        Evaluate depth prediction results.

        Args:
            results (list): List of depth prediction results
            depth_config (dict): Depth evaluation configuration
            logger (logging.Logger): Logger for output

        Returns:
            dict: Depth evaluation results
        """
        from .depth_metrics import depth_evaluation
        from mmcv.utils import print_log
        import os
        if depth_config is None:
            depth_config = {}

        print_log('Starting depth evaluation', logger=logger)

        vis_depth = depth_config.get('vis_depth', False)

        tmpdepth_dir = depth_config.get('vis_dir', 'tmpdepth')

        # Create directory for saving depth maps
        if vis_depth:
            tmpdepth_dir = os.path.join(os.getcwd(), tmpdepth_dir)
            os.makedirs(tmpdepth_dir, exist_ok=True)
            print_log(f'Saving depth maps to {tmpdepth_dir}', logger=logger)

        depth_metrics = []

        # Process each sample
        for i, (result_dict, info) in enumerate(zip(results, self.data_infos)):
            # Get predicted depth
            if 'depth_pred' in result_dict:
                depth_pred = result_dict['depth_pred'].squeeze(-1)
            else:
                print_log(f'No depth prediction found for sample {i}', logger=logger)
                continue
            depth_gt = None
            depth_mask = None
            if 'depth_map' in result_dict:
                depth_gt = result_dict['depth_map'].detach().cpu().numpy()
                depth_mask = result_dict['depth_map_mask'].detach().cpu().numpy()
            else:
                cam2_img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])
                cam3_img_filename = cam2_img_filename.replace("image_2", "image_3")
                depth_paths = [cam2_img_filename.replace("image_2", "depth_npy/image_2").replace('.png', '.npy'), cam3_img_filename.replace("image_3", "depth_npy/image_3").replace('.png', '.npy')]
                try:
                    import os
                    if os.path.exists(depth_paths[0]):  # Check if first depth file exists
                        # Load multi-view depth maps
                        depth_gt_origin = np.stack([self._load_depth_file(name) for name in depth_paths], axis=0)
                        if 'ida_mat' in result_dict:
                            ida_mat = result_dict['ida_mat']
                        if isinstance(ida_mat, torch.Tensor):
                            ida_mat = ida_mat.detach().cpu().numpy()
                        ida_mat = ida_mat.astype(np.float32)
                        N, out_h, out_w = depth_pred.shape[0], depth_pred.shape[1], depth_pred.shape[2]
                        depth_gt = np.empty((N, out_h, out_w), dtype=np.float32)
                        for i in range(N):
                            M = ida_mat[i]
                            depth_gt[i] = cv2.warpPerspective(
                                depth_gt_origin[i].astype(np.float32),
                                M,
                                (out_w, out_h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.0
                            )

                        if vis_depth:
                            # Save depth maps before evaluation
                            sample_token = info['token']
                            scene_name = self.token2name[info['scene_token']]

                            # Save predicted depth
                            pred_path = os.path.join(tmpdepth_dir, f'{scene_name}_{sample_token}_pred.npz')
                            np.savez(pred_path, depth_pred=depth_pred.detach().cpu().numpy())

                            # Save ground truth depth (warped)
                            gt_path = os.path.join(tmpdepth_dir, f'{scene_name}_{sample_token}_gt.npz')
                            np.savez(gt_path, depth_gt=depth_gt)

                            # Also save visualizations for inspection
                            self._save_depth_visualizations(
                                depth_pred.detach().cpu().numpy(),
                                depth_gt,
                                sample_token,
                                scene_name,
                                tmpdepth_dir
                            )

                            print_log(f'Saved depth maps for sample {i}: {scene_name}_{sample_token}', logger=logger)
                    else:
                        print_log(f'Depth files not found for sample {i}, skipping', logger=logger)
                        continue

                except Exception as e:
                    print_log(f'Error loading depth for sample {i}: {str(e)}', logger=logger)
                    continue

            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                depth_pred.detach().cpu().numpy(),
                depth_gt,
                max_depth=depth_config.get('max_depth', 70),
                use_gpu=True,
                vggt_scale=self.vggt_scale,
                metric_scale=depth_config.get('metric_scale', False),
                pred_depth_scale=depth_config.get('pred_depth_scale', 1.0))
            depth_metrics.append(depth_results)

        # Calculate depth metrics using the imported depth_evaluation function
        try:
            average_metrics = {
                key: np.average(
                    [metrics[key] for metrics in depth_metrics],
                    weights=[metrics["valid_pixels"] for metrics in depth_metrics],
                )
                for key in depth_metrics[0].keys()
                if key != "valid_pixels"
            }
            print_log(f"Average depth evaluation metrics: {average_metrics}", logger=logger)
            return average_metrics
        except Exception as e:
            print_log(f'Error in depth evaluation: {str(e)}', logger=logger)
            return {}

    def _load_depth_file(self, filepath):
        """Overriding to support .npy files generated by DDAD processor."""
        if filepath.endswith('.npy'):
            return np.load(filepath)
        else:
            # Fallback to standard image loading
            import cv2
            depth = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
            return depth

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix
