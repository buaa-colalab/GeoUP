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
import copy
import math
import mmcv
import mmdet3d
import numpy as np
import os
import random
import re
import subprocess
import tempfile
from os import path as osp

import cv2
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                               LiDARInstance3DBoxes, points_cam2img)
from mmdet3d.datasets import WaymoDataset
from nuscenes.eval.common.utils import Quaternion


@DATASETS.register_module()
class CustomWayMoDataset(WaymoDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, vggt_mode=False, load_interval=1, test_seq_interval=None, *args, **kwargs):
        self.load_interval = load_interval
        self.test_seq_interval = test_seq_interval if test_seq_interval is not None else load_interval
        self.test_seq_reordered = False
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        self.vggt_mode = vggt_mode
        if not self.test_mode:
            self.flite_empty()
        if seq_mode:
            self.num_frame_losses = num_frame_losses
            self.queue_length = queue_length
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.
    def flite_empty(self):
        """
        Filter out empty samples
        """
        invalid_indices = set()
        invalid_samples_path = os.path.join(self.data_root, 'waymo_invalid_3d_label_samples_multithread.txt')
        with open(invalid_samples_path, 'r', encoding='utf-8') as f:
            # 跳过表头
            header = f.readline()
            for line in f:
                # 每行格式：样本索引,场景token,时间戳,异常原因
                parts = line.strip().split(',')
                if len(parts) < 1:
                    continue
                try:
                    idx = int(parts[0])  # 提取样本索引
                    invalid_indices.add(idx)
                except ValueError:
                    print(f"跳过无效行：{line}")
        
        # print(f"共加载 {len(invalid_indices)} 个异常样本索引")
        self.data_infos = [
            info for idx, info in enumerate(self.data_infos) if idx not in invalid_indices
        ]
        # print(f"过滤后数据集总样本数：{len(self.data_infos)}")

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        data_infos = mmcv.load(ann_file, file_format='pkl')
        if getattr(self, 'test_mode', False):
            data_infos = self._reorder_test_sequence(data_infos)
            return data_infos
        return data_infos[::self.load_interval]

    def _reorder_test_sequence(self, data_infos):
        interval = max(int(self.test_seq_interval), 1)
        if interval == 1:
            return data_infos

        reordered_infos = []
        scene_infos = []

        def flush_scene():
            if not scene_infos:
                return
            scene_token = scene_infos[0]['scene']
            for offset in range(interval):
                subseq_infos = scene_infos[offset::interval]
                if len(subseq_infos) == 0:
                    continue
                test_scene_token = f'{scene_token}_test_stride{interval}_offset{offset}'
                for info in subseq_infos:
                    info = copy.deepcopy(info)
                    info['_test_scene_token'] = test_scene_token
                    reordered_infos.append(info)

        prev_scene_token = None
        for info in data_infos:
            scene_token = info['scene']
            if prev_scene_token is not None and scene_token != prev_scene_token:
                flush_scene()
                scene_infos = []
            scene_infos.append(info)
            prev_scene_token = scene_token
        flush_scene()

        self.test_seq_reordered = True
        return reordered_infos

    def _get_test_sequence_indices(self, index):
        interval = 1 if self.test_seq_reordered else max(int(self.test_seq_interval), 1)
        curr_scene_token = self.data_infos[index]['scene']
        curr_test_scene_token = self.data_infos[index].get('_test_scene_token', curr_scene_token)
        indices = []
        last_valid_index = index

        for offset in range(self.queue_length):
            curr_index = index - offset * interval
            if (curr_index >= 0 and
                    self.data_infos[curr_index]['scene'] == curr_scene_token and
                    self.data_infos[curr_index].get('_test_scene_token', curr_test_scene_token) == curr_test_scene_token):
                last_valid_index = curr_index
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
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
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
            # print(f"start process data, scene: {input_dict['scene_token']}, time: {input_dict['timestamp']}")

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
            # print(f"finish process data, scene: {input_dict['scene_token']}, time: {input_dict['timestamp']}")
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
        index_list = self._get_test_sequence_indices(index)
        for i in index_list:
            input_dict = self.get_data_info(i)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

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
        def get_test_data(each, key):
            value = each[key]
            if isinstance(value, (list, tuple)):
                value = value[0]
            return value.data if isinstance(value, DC) else value

        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([get_test_data(each, key) for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([get_test_data(each, key) for each in queue], cpu_only=True)

        queue = queue[-1]
        return queue
    
    def remove_classes(self, ann_info, classes_to_remove=['DontCare', 'Sign']):
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
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

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
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

        gt_labels = []
        KITTI2NU = {
            'Car'           : 'car',
            # 'Van'           : 'car',
            # 'Truck'         : 'truck',
            'Pedestrian'    : 'pedestrian',
            # 'Person_sitting': 'pedestrian',
            'Cyclist'       : 'bicycle',
            # 'Tram'          : 'bus',
            # Misc/DontCare 直接丢弃
        }
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        # breakpoint()

        anno_bbox = annos['bbox']
        non_zero_rows_mask = np.any(anno_bbox != 0, axis=1)
        select = np.where(non_zero_rows_mask)[0]
        anno_bbox = anno_bbox[select]
        anno_labels = gt_labels[select]
        gt_labels_3d = copy.deepcopy(anno_labels)
        anno_camera_id = annos['camera_id'][select]
        gt_bboxes_3d = gt_bboxes_3d[select]
        gt_names = gt_names[select]
        gt_velocity = annos['velocity'][select]

        gt_bboxes = []
        gt_labels_2d = []
        for i in range(5):
            bbox_idx = np.where(anno_camera_id == i)[0]
            bboxes = anno_bbox[bbox_idx]
            labels = anno_labels[bbox_idx]
            gt_bboxes.append(bboxes)
            gt_labels_2d.append(labels)
            # img_path = os.path.join(self.data_root, info['image']['image_path']).replace('image_0', f'image_{i}')
            # # visualize_2d_boxes_on_image(img_path, bboxes, annos['name'][bbox_idx], 'result.jpg')

        centers2d = [[] for _ in range(5)]
        depths = [[] for _ in range(5)]
        for i in range(len(gt_bboxes_3d)):
            bboxes_3d = gt_bboxes_3d[i]
            cam_id = int(anno_camera_id[i])
            intrinsics = info['calib'][f'P{cam_id}']
            extrinsics = info['calib'][f'Tr_velo_to_cam{cam_id}']
            ref_extrinsics = info['calib']['Tr_velo_to_cam']
            ref2cam = extrinsics @ invert_matrix_egopose_numpy(ref_extrinsics)
            ref_intrinsics = intrinsics @ ref2cam
            # img_path = os.path.join(self.data_root, info['image']['image_path']).replace('image_0', f'image_{cam_id}')
            # visualize_kitti_gt_bboxes_on_image(img_path, ref_intrinsics, bboxes_3d, info['annos']['name'][i], output_path='result.jpg')
            project_centers2d = points_cam2img(bboxes_3d[:3], ref_intrinsics, with_depth=True)
            center2d, depth = project_centers2d[:2], project_centers2d[2:3]
            centers2d[cam_id].append(center2d)
            depths[cam_id].append(depth)
        for i in range(5):
            if len(centers2d[i]) > 0:
                centers2d[i] = np.concatenate(centers2d[i], axis=0).reshape(-1, 2)
                depths[i] = np.concatenate(depths[i], axis=0).reshape(-1)
            else:
                centers2d[i] = np.zeros((0, 2), dtype=np.float32)
                depths[i] = np.zeros((0,), dtype=np.float32)

        gt_bboxes = [bbox.astype(np.float32) for bbox in gt_bboxes]    
            

        # convert gt_bboxes_3d to velodyne coordinates
        nus_lidar_to_kitti_lidar = np.array([[0, 1, 0, 0],
                                            [-1,  0, 0, 0],
                                            [0,  0, 1, 0],
                                            [0,  0, 0, 1]])

        gt_velocity = gt_velocity @ nus_lidar_to_kitti_lidar[:2, :2]
        yaw = gt_bboxes_3d[:, 6:7]
        new_yaw = yaw + np.pi / 2
        new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi
        gt_bboxes_3d = np.concatenate((gt_bboxes_3d[:, :6], new_yaw, gt_velocity), axis=1)
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d, box_dim=9).convert_to(self.box_mode_3d, np.linalg.inv(rect @ Trv2c @ nus_lidar_to_kitti_lidar))
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels_2d,
            gt_names=gt_names,
            centers2d=centers2d,
            depths=depths,
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
        nus_lidar_to_kitti_lidar = np.array([[0, 1, 0, 0],
                                            [-1,  0, 0, 0],
                                            [0,  0, 1, 0],
                                            [0,  0, 0, 1]])
        input_dict['dataset'] = 'waymo'
        sample_idx = info['image']['image_idx']
        input_dict['sample_idx'] = index
        input_dict['pts_filename'] = self._get_pts_filename(sample_idx)
        input_dict['sweeps'] = info['sweeps']
        input_dict['timestamp'] = info['timestamp'] / 1e6
        input_dict['ego_pose'] = info['pose'] @ nus_lidar_to_kitti_lidar
        input_dict['ego_pose_inv'] = invert_matrix_egopose_numpy(input_dict['ego_pose'])
        input_dict['prev_idx'] = ''
        input_dict['next_idx'] = ''
        input_dict['scene_token'] = info.get('_test_scene_token', info['scene'])
        input_dict['frame_idx'] = sample_idx
        input_dict['img_timestamp'] = [info['timestamp'] / 1e6 for _ in range(5)]
        cam0_img_filename = os.path.join(self.data_root, info['image']['image_path'])
        image_paths = []
        depth_paths = []
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        cam_extrinsics_global = []
        for cam_idx in range(5):
            img_filename = cam0_img_filename.replace('image_0', f'image_{cam_idx}')
            depth_filename = img_filename.replace(f'image_{cam_idx}', f'depth_npy/image_{cam_idx}').replace('.jpg', '.npy')
            P = info['calib'][f'P{cam_idx}']
            rect = info['calib'][f'R0_rect']
            Trv2c = info['calib'][f'Tr_velo_to_cam{cam_idx}']
            nus_lidar_to_cam = rect @ Trv2c @ nus_lidar_to_kitti_lidar
            lidar2img = P @ nus_lidar_to_cam
            global2cam = nus_lidar_to_cam @ input_dict['ego_pose_inv']
            cam_extrinsics_global.append(global2cam)
            image_paths.append(img_filename)
            depth_paths.append(depth_filename)
            lidar2img_rts.append(lidar2img)
            intrinsics.append(P)
            extrinsics.append(nus_lidar_to_cam)
        
        if not self.test_mode: # for seq_mode
            prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
        else:
            prev_exists = None
        input_dict.update(
            dict(
                img_filename=image_paths,
                depth_filename=depth_paths,
                lidar2img=lidar2img_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                cam_extrinsics_global=cam_extrinsics_global,
                prev_exists=prev_exists,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        
        # Visualization code
        vis = False
        if vis:
            import cv2
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

                output_path_2d = f'vis/waymo/result_2d_{img_idx}_waymo.jpg'
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

                output_path_3d = f'vis/waymo/result_3d_{img_idx}_waymo.jpg'
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
            # print(f"waymo, idx: {idx}")
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
            bbox_3d = result['boxes_3d'].tensor
            x, y, z = bbox_3d[:, 0:1], bbox_3d[:, 1:2], bbox_3d[:, 2:3]
            yaw = bbox_3d[:, 6:7]
            new_x, new_y, new_z = y, -x, z
            new_yaw = yaw + np.pi / 2
            new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi
            bbox_3d = torch.cat([new_x, new_y, new_z, bbox_3d[:, 3:6], new_yaw, bbox_3d[:, 7:]], dim=1)
            new_result['boxes_3d'] = LiDARInstance3DBoxes(bbox_3d, box_dim=bbox_3d.shape[-1])
            new_results.append(new_result)
        return new_results

    def _extract_waymo_metric_values(self, metric_text, metric_name):
        pattern = rf'\b{re.escape(metric_name)}\b\s*\[([^\]]+)\]'
        return [float(x) for x in re.findall(pattern, metric_text)]

    def _parse_waymo_detection_metrics(self, metric_text):
        mAP_values = self._extract_waymo_metric_values(metric_text, 'mAP')
        mAPH_values = self._extract_waymo_metric_values(metric_text, 'mAPH')
        metric_order = [
            'Vehicle/L1',
            'Vehicle/L2',
            'Pedestrian/L1',
            'Pedestrian/L2',
            'Sign/L1',
            'Sign/L2',
            'Cyclist/L1',
            'Cyclist/L2',
        ]

        if len(mAP_values) < len(metric_order) or len(mAPH_values) < len(metric_order):
            raise RuntimeError(
                'Failed to parse Waymo detection metrics output. '
                'Please check the output of compute_detection_metrics_main.'
            )

        ap_dict = {}
        for metric_name, values in (('mAP', mAP_values), ('mAPH', mAPH_values)):
            for class_name, value in zip(metric_order, values):
                ap_dict[f'{class_name} {metric_name}'] = value
            if len(values) >= len(metric_order) + 2:
                ap_dict[f'Overall/L1 {metric_name}'] = values[len(metric_order)]
                ap_dict[f'Overall/L2 {metric_name}'] = values[len(metric_order) + 1]

        main_classes = ('Vehicle', 'Pedestrian', 'Cyclist')
        for level in ('L1', 'L2'):
            for metric_name in ('mAP', 'mAPH'):
                ap_dict[f'Overall/{level} {metric_name}'] = float(np.mean([
                    ap_dict[f'{class_name}/{level} {metric_name}']
                    for class_name in main_classes
                ]))
        return ap_dict

    def _parse_waymo_let_metrics(self, metric_text):
        object_metric_pattern = re.compile(
            r'OBJECT_TYPE_TYPE_'
            r'(VEHICLE|PEDESTRIAN|SIGN|CYCLIST)'
            r'_LEVEL_\d+:\s*'
            r'\[LET-mAPL\s+([^\]]+)\]\s*'
            r'\[LET-mAP\s+([^\]]+)\]\s*'
            r'\[LET-mAPH\s+([^\]]+)\]',
            re.MULTILINE)
        object_name_map = {
            'VEHICLE': 'Vehicle',
            'PEDESTRIAN': 'Pedestrian',
            'SIGN': 'Sign',
            'CYCLIST': 'Cyclist',
        }
        object_matches = object_metric_pattern.findall(metric_text)
        if object_matches:
            ap_dict = {}
            for raw_name, mapl_text, map_text, maph_text in object_matches:
                class_name = object_name_map[raw_name]
                metric_triplet = {
                    'mAPL': float(mapl_text),
                    'mAP': float(map_text),
                    'mAPH': float(maph_text),
                }
                for metric_name, value in metric_triplet.items():
                    ap_dict[f'{class_name}/{metric_name}'] = value
                    ap_dict[f'{class_name} {metric_name}'] = value

            main_classes = ('Vehicle', 'Pedestrian', 'Cyclist')
            for metric_name in ('mAPL', 'mAP', 'mAPH'):
                avg_value = float(np.mean([
                    ap_dict[f'{class_name}/{metric_name}']
                    for class_name in main_classes
                ]))
                ap_dict[f'Overall/{metric_name}'] = avg_value
                ap_dict[f'Overall {metric_name}'] = avg_value
            return ap_dict

        metric_values = {
            'mAPL': self._extract_waymo_metric_values(metric_text, 'mAPL'),
            'mAP': self._extract_waymo_metric_values(metric_text, 'mAP'),
            'mAPH': self._extract_waymo_metric_values(metric_text, 'mAPH'),
        }
        value_counts = [len(values) for values in metric_values.values()]
        if min(value_counts) < 3:
            raise RuntimeError(
                'Failed to parse Waymo LET-AP metrics output. '
                'Please check the output of compute_detection_let_metrics_main.'
            )

        num_entries = min(value_counts)
        if num_entries == 3:
            class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
        elif num_entries == 4:
            class_names = ['Vehicle', 'Pedestrian', 'Cyclist', 'Overall']
        elif num_entries == 5:
            class_names = ['Vehicle', 'Pedestrian', 'Cyclist', 'Sign', 'Overall']
        else:
            class_names = [f'Class_{idx}' for idx in range(num_entries)]

        ap_dict = {}
        for metric_name, values in metric_values.items():
            for class_name, value in zip(class_names, values[:num_entries]):
                ap_dict[f'{class_name}/{metric_name}'] = value

        main_classes = ('Vehicle', 'Pedestrian', 'Cyclist')
        for metric_name in metric_values.keys():
            avg_value = float(np.mean([
                ap_dict[f'{class_name}/{metric_name}']
                for class_name in main_classes
            ]))
            # Match the reference implementation: report the final Overall
            # score as the mean of Vehicle/Pedestrian/Cyclist, instead of
            # relying on whether the evaluator emits its own Overall row.
            ap_dict[f'Overall/{metric_name}'] = avg_value
            ap_dict[f'Overall {metric_name}'] = avg_value
        return ap_dict

    def _get_waymo_root(self):
        return osp.join(self.data_root.split('kitti_format')[0], 'waymo_format')

    def _get_waymo_eval_split_info(self):
        ann_file = osp.basename(str(self.ann_file)).lower()
        split = str(self.split).lower()
        if 'test' in ann_file or split in {'testing', 'test'}:
            return 'testing', '2'
        if any(tag in ann_file for tag in ('val', 'train')) or split in {
                'training', 'train', 'val', 'validation', 'trainval'}:
            return 'validation', '1'
        raise ValueError(
            f'Unsupported Waymo split "{self.split}" with ann_file "{self.ann_file}".'
        )

    def _resolve_waymo_metric_binary(self, binary_name):
        binary_dir = osp.join(
            osp.dirname(mmdet3d.__file__), 'core', 'evaluation', 'waymo_utils')
        candidates = [
            osp.join(binary_dir, binary_name),
            osp.join(binary_dir, f'{binary_name}.exe'),
        ]
        for candidate in candidates:
            if osp.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f'Cannot find Waymo metric binary "{binary_name}" under "{binary_dir}". '
            'Please compile the corresponding Waymo evaluator first.'
        )

    def _format_waymo_results(self,
                              outputs,
                              pklfile_prefix=None,
                              submission_prefix=None):
        from mmdet3d.core.evaluation.waymo_utils.prediction_kitti_to_waymo import \
            KITTI2Waymo

        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files, _ = super().format_results(
            outputs,
            pklfile_prefix,
            submission_prefix,
            data_format='kitti')

        waymo_root = self._get_waymo_root()
        split_dir, prefix = self._get_waymo_eval_split_info()
        waymo_tfrecords_dir = osp.join(waymo_root, split_dir)
        waymo_results_final_path = f'{pklfile_prefix}.bin'
        save_tmp_dir = tempfile.TemporaryDirectory()
        try:
            converter_input = result_files['pts_bbox'] \
                if isinstance(result_files, dict) and 'pts_bbox' in result_files \
                else result_files
            converter = KITTI2Waymo(
                converter_input,
                waymo_tfrecords_dir,
                save_tmp_dir.name,
                waymo_results_final_path,
                prefix)
            converter.convert()
        finally:
            save_tmp_dir.cleanup()

        return result_files, tmp_dir, waymo_results_final_path

    def _evaluate_waymo_metric(self,
                               results,
                               logger=None,
                               jsonfile_prefix=None,
                               submission_prefix=None,
                               binary_name='compute_detection_metrics_main',
                               gt_bin_name='gt.bin',
                               parser=None):
        from mmcv.utils import print_log

        _, tmp_dir, pred_bin_path = self._format_waymo_results(
            results, jsonfile_prefix, submission_prefix)
        try:
            waymo_root = self._get_waymo_root()
            gt_path = osp.join(waymo_root, gt_bin_name)
            if not osp.exists(gt_path):
                raise FileNotFoundError(
                    f'Cannot find Waymo ground-truth bin "{gt_path}".')

            binary_path = self._resolve_waymo_metric_binary(binary_name)
            ret_bytes = subprocess.check_output(
                [binary_path, pred_bin_path, gt_path],
                stderr=subprocess.STDOUT)
            ret_text = ret_bytes.decode('utf-8', errors='ignore')
            print_log(ret_text, logger=logger)
            if parser is None:
                return {}
            parsed_metrics = parser(ret_text)
            overview_keys = [
                'Overall mAPL',
                'Overall mAP',
                'Overall mAPH',
                'Overall/L1 mAP',
                'Overall/L1 mAPH',
                'Overall/L2 mAP',
                'Overall/L2 mAPH',
            ]
            overview_lines = []
            for key in overview_keys:
                if key in parsed_metrics:
                    overview_lines.append(f'{key}: {parsed_metrics[key]:.6f}')
            if overview_lines:
                print_log(
                    'Waymo metric overview:\n' + '\n'.join(overview_lines),
                    logger=logger)
            return parsed_metrics
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

    def _evaluate_kitti_bbox_metric(self,
                                    results,
                                    logger=None,
                                    jsonfile_prefix=None,
                                    submission_prefix=None):
        from mmcv.utils import print_log
        from mmdet3d.core.evaluation import kitti_eval

        tmp_dir = None
        try:
            result_files, tmp_dir = self.format_results(
                results,
                jsonfile_prefix,
                submission_prefix,
                data_format='kitti')
            gt_annos = [info['annos'] for info in self.data_infos]
            empty_list = []
            for i, anno in enumerate(gt_annos):
                if len(anno['alpha']) == 0:
                    empty_list.append(i)
            gt_annos = [
                gt_annos[i] for i in range(len(gt_annos))
                if i not in empty_list
            ]
            result_files_pts = [
                result_files[i] for i in range(len(result_files))
                if i not in empty_list
            ]
            result_files = {'pts_bbox': result_files_pts}
            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=['bev', '3d'])
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(f'{ap:.4f}')
                    print_log(
                        f'Results of {name}:\n' + ap_result_str,
                        logger=logger)
            else:
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos,
                    result_files,
                    self.CLASSES,
                    eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)
            return ap_dict
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

    def _evaluate_bbox_metric(self,
                              results,
                              logger=None,
                              jsonfile_prefix=None,
                              submission_prefix=None,
                              bbox_metric_config=None):
        bbox_metric_config = bbox_metric_config or {}
        protocol = str(
            bbox_metric_config.get('protocol', 'waymo_let')).lower()

        if protocol == 'kitti':
            return self._evaluate_kitti_bbox_metric(
                results,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                submission_prefix=submission_prefix)

        if protocol in {'waymo', 'map', 'detection'}:
            return self._evaluate_waymo_metric(
                results,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                submission_prefix=submission_prefix,
                binary_name=bbox_metric_config.get(
                    'binary_name', 'compute_detection_metrics_main'),
                gt_bin_name=bbox_metric_config.get(
                    'gt_bin_name', 'cam_gt.bin'),
                parser=self._parse_waymo_detection_metrics)

        if protocol in {'waymo_let', 'let', 'let_ap', 'camera'}:
            return self._evaluate_waymo_metric(
                results,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                submission_prefix=submission_prefix,
                binary_name=bbox_metric_config.get(
                    'binary_name', 'compute_detection_let_metrics_main'),
                gt_bin_name=bbox_metric_config.get(
                    'gt_bin_name', 'cam_gt.bin'),
                parser=self._parse_waymo_let_metrics)

        raise ValueError(
            f'Unsupported bbox evaluation protocol "{protocol}". '
            'Expected one of {"waymo_let", "waymo", "kitti"}.'
        )
    
    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 jsonfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 bbox_eval=dict(protocol='let'),
                 **kwargs):
        """Evaluation for Waymo-compatible detection protocols.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'waymo'. Supported metrics include 'bbox',
                'waymo', 'waymo_let', 'let' and 'mvfcos3d++'.
                For this dataset, 'bbox' defaults to the Waymo LET metric
                path to match the camera-box evaluation flow. Set
                ``bbox_eval=dict(protocol='kitti')`` in kwargs to recover the
                previous KITTI-style AP evaluation.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of pkl files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        from mmcv.utils import print_log
        eval_results = {}

        if not isinstance(metric, (list, tuple)):
            metric = [metric]
        metric = [m.lower() if isinstance(m, str) else m for m in metric]

        box_metrics = {
            'bbox', 'waymo', 'waymo_let', 'let', 'let_ap',
            'mvfcos3d', 'mvfcos3d++', 'mvfcos3d_plusplus'
        }
        box_results = None
        if any(m in box_metrics for m in metric):
            box_results = self.transform_result_from_nuscenes(results)

        if 'depth' in metric:
            depth_metric_config = kwargs.get('depth_eval', {})
            depth_results = self._evaluate_depth(results, depth_metric_config, logger)
            eval_results.update(depth_results)
        
        if 'camera' in metric:
            camera_metric_config = kwargs.get('camera_eval', {})
            camera_results = self._evaluate_camera(results, camera_metric_config, logger)
            eval_results.update(camera_results)

        if 'bbox' in metric:
            bbox_metric_config = kwargs.get('bbox_eval', {})
            bbox_results = self._evaluate_bbox_metric(
                box_results,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                submission_prefix=submission_prefix,
                bbox_metric_config=bbox_metric_config)
            eval_results.update(bbox_results)

        if 'waymo' in metric:
            waymo_metric_config = kwargs.get('waymo_eval', {})
            waymo_results = self._evaluate_waymo_metric(
                box_results,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                submission_prefix=submission_prefix,
                binary_name=waymo_metric_config.get(
                    'binary_name', 'compute_detection_metrics_main'),
                gt_bin_name=waymo_metric_config.get('gt_bin_name', 'gt.bin'),
                parser=self._parse_waymo_detection_metrics)
            eval_results.update(waymo_results)

        if any(m in metric for m in (
                'waymo_let', 'let', 'let_ap',
                'mvfcos3d', 'mvfcos3d++', 'mvfcos3d_plusplus')):
            let_metric_config = kwargs.get('let_eval', {})
            let_results = self._evaluate_waymo_metric(
                box_results,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                submission_prefix=submission_prefix,
                binary_name=let_metric_config.get(
                    'binary_name', 'compute_detection_let_metrics_main'),
                gt_bin_name=let_metric_config.get('gt_bin_name', 'cam_gt.bin'),
                parser=self._parse_waymo_let_metrics)
            eval_results.update(let_results)

        if box_results is not None and (show or out_dir):
            self.show(box_results, out_dir, show=show, pipeline=pipeline)

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
        if depth_config is None:
            depth_config = {}
        
        print_log('Starting depth evaluation', logger=logger)

        vis_depth = depth_config.get('vis_depth', False)

        tmpdepth_dir = depth_config.get('vis_dir', 'tmpdepth')

        # Create directory for saving depth maps
        if vis_depth:
            import os
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
                import os
                depth_paths = []
                cam0_img_filename = os.path.join(self.data_root, info['image']['image_path'])
                for cam_idx in range(5):
                    img_filename = cam0_img_filename.replace('image_0', f'image_{cam_idx}')
                    depth_filename = img_filename.replace(f'image_{cam_idx}', f'depth_npy/image_{cam_idx}').replace('.jpg', '.npy')
                    depth_paths.append(depth_filename)
                try:
                    if os.path.exists(depth_paths[0]):  # Check if first depth file exists
                        missing_depth_paths = [name for name in depth_paths if not os.path.exists(name)]
                    if missing_depth_paths:
                        print_log(
                            f'Missing depth files for sample {i}, skipping. '
                            f'First missing file: {missing_depth_paths[0]}',
                            logger=logger)
                        continue

                    if 'ida_mat' not in result_dict:
                        print_log(f'ida_mat missing for sample {i}, skipping depth evaluation', logger=logger)
                        continue

                    ida_mat = result_dict['ida_mat']
                    if isinstance(ida_mat, torch.Tensor):
                        ida_mat = ida_mat.detach().cpu().numpy()
                    ida_mat = np.asarray(ida_mat, dtype=np.float32)

                    N, out_h, out_w = depth_pred.shape[0], depth_pred.shape[1], depth_pred.shape[2]
                    num_views = min(N, len(depth_paths), len(ida_mat))
                    if num_views != N or num_views != len(depth_paths) or num_views != len(ida_mat):
                        print_log(
                            f'View count mismatch for sample {i}: '
                            f'depth_pred={N}, depth_files={len(depth_paths)}, ida_mat={len(ida_mat)}. '
                            f'Using first {num_views} views.',
                            logger=logger)

                    # AV2 cameras can have different native resolutions, so load and warp per view
                    # instead of stacking raw depth maps before perspective transform.
                    depth_gt = np.zeros((N, out_h, out_w), dtype=np.float32)
                    for view_idx in range(num_views):
                        depth_gt_origin = self._load_depth_file(depth_paths[view_idx]).astype(np.float32)
                        M = ida_mat[view_idx]
                        depth_gt[view_idx] = cv2.warpPerspective(
                            depth_gt_origin,
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
                    
                except Exception as e:
                    print_log(f'Error loading depth for sample {i}: {str(e)}', logger=logger)
                    continue

            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(depth_pred.detach().cpu().numpy(), depth_gt, max_depth=depth_config.get('max_depth', 70), use_gpu=True)
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
