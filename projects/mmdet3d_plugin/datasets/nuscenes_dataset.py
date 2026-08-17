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
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
import mmcv
import random
import math
import json
from .occ_metrics import Metric_mIoU_Occ3D
from .ray_metrics import main_rayiou
from .depth_metrics import depth_evaluation
from .pipelines.load import load_16big_png_depth
from mmcv.utils import print_log
import cv2

from os import path as osp
import copy
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.common.data_classes import EvalBoxes

# NumPy 2.x removed the legacy np.Inf alias, but older matplotlib
# releases used by this project still access it in layout code paths.
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

def filter_eval_boxes_by_range(nusc, eval_boxes, min_dist, max_dist, verbose=False):
    """
    过滤 EvalBoxes，只保留在 min_dist 到 max_dist 范围内的框（基于 Ego 坐标系）。
    """
    print(f"Filtering boxes for range {min_dist}m - {max_dist}m...")
    filtered_boxes = EvalBoxes()

    for sample_token, boxes in eval_boxes.boxes.items():
        # 获取当前 sample 的 ego pose
        # 注意：NuScenes 的检测通常是基于关键帧的 LIDAR_TOP 时间戳
        sample = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        # 准备 Global 到 Ego 的变换矩阵
        # 旋转 (逆)
        global2ego_rot = Quaternion(pose_record['rotation']).inverse
        # 平移 (负)
        global2ego_trans = -np.array(pose_record['translation'])

        filtered_sample_boxes = []
        for box in boxes:
            # 复制一份，避免修改原始数据
            # EvalBox 只有 translation, size, rotation 等属性
            # 我们只需要计算它的中心点距离

            # 1. 平移
            center = np.array(box.translation) + global2ego_trans
            # 2. 旋转
            center = global2ego_rot.rotate(center)

            # 计算距离 (欧氏距离, 忽略高度 z 轴，通常只看 BEV 平面距离)
            dist = np.sqrt(center[0]**2 + center[1]**2)

            if min_dist <= dist < max_dist:
                filtered_sample_boxes.append(box)

        filtered_boxes.add_boxes(sample_token, filtered_sample_boxes)

    if verbose:
        print(f"Range {min_dist}-{max_dist}m: Kept {len(filtered_boxes.all)} boxes.")

    return filtered_boxes

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, vggt_mode=False, test_scene=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        self.vggt_mode = vggt_mode
        if not test_scene:
            try:
                with open(self.data_root + 'v1.0-trainval/scene.json', 'r') as f:
                    scenes = json.load(f)          # list[dict]
            except:
                with open(self.data_root + 'v1.0-mini/scene.json', 'r') as f:
                    scenes = json.load(f)          # list[dict]
        else:
            with open(self.data_root + 'v1.0-test/scene.json', 'r') as f:
                scenes = json.load(f)          # list[dict]
        self.token2name = {s['token']: s['name'] for s in scenes}
        # Build token-to-index mapping for O(1) lookup
        self.token2idx = {info['token']: idx for idx, info in enumerate(self.data_infos)}
        if seq_mode:
            self.num_frame_losses = num_frame_losses
            self.queue_length = queue_length
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

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
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

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
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
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

        num_augs = len(queue[-1]['img']) if isinstance(queue[-1]['img'], list) else 1

        for key in self.collect_keys:
            if isinstance(queue[-1][key], list):
                if key != 'img_metas':
                    aug_list = []
                    for aug_id in range(num_augs):
                        stacked_tensor = torch.stack([each[key][aug_id].data for each in queue])
                        aug_list.append(DC(stacked_tensor, cpu_only=False, stack=True, pad_dims=None))
                    queue[-1][key] = aug_list
                else:
                    aug_list = []
                    for aug_id in range(num_augs):
                        meta_list = [each[key][aug_id].data for each in queue]
                        aug_list.append(DC(meta_list, cpu_only=True))
                    queue[-1][key] = aug_list
            else:
                if key != 'img_metas':
                    queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)

        queue = queue[-1]
        return queue

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
        # standard protocal modified from SECOND.Pytorch

        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        l2e_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_translation = info['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        ego2lidar = invert_matrix_egopose_numpy(l2e_matrix)
        input_dict = dict(
            dataset='nuscenes',
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            ego2global=[e2g_matrix for _ in range(6)],
            ego2lidar=[ego2lidar for _ in range(6)],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            scene_name=self.token2name[info['scene_token']],  # Add scene_name for OCC compatibility
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            depth_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            cam_extrinsics_global = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                depth_paths.append(cam_info['data_path'].replace("nuscenes", "nuscenes_depth").replace("jpg", "png"))
                # obtain lidar to image transformation matrix
                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                cam_extrinsics_global.append(lidar2cam_rt @ ego_pose_inv)
                lidar2img_rts.append(lidar2img_rt)

            ego2img_rts = [(lidar2img_rts[i] @ ego2lidar.astype(np.float32)) for i in range(len(lidar2img_rts))]

            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    depth_filename=depth_paths,
                    lidar2img=lidar2img_rts,
                    ego2img=ego2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    cam_extrinsics_global=cam_extrinsics_global,
                    prev_exists=prev_exists,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            annos.update(
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    centers2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore'])
            )
            input_dict['ann_info'] = annos
        vis = False
        if vis:
            import cv2
            import matplotlib.pyplot as plt
            from mmdet3d.core.bbox import LiDARInstance3DBoxes
            import os
            # Get 3D boxes and labels
            gt_bboxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
            gt_labels_3d = annos['gt_labels_3d']

            # Get 2D boxes and labels
            bboxes2d = annos['bboxes']
            labels2d = annos['labels']

            # Create a color map for classes
            colors = plt.cm.get_cmap('hsv', len(self.CLASSES))

            # --- Visualize and save 2D bounding boxes ---
            for img_idx, img_path in enumerate(input_dict['img_filename'][:2]): # Process first two images
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
            for img_idx, (img_path, lidar2img) in enumerate(zip(input_dict['img_filename'][:2], input_dict['lidar2img'][:2])):
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

    def log_table_results(self, results, depth_ranges=None, logger=None):
        """
        将字典结果格式化为漂亮的表格并打印。
        """
        from mmcv.utils import print_log

        # 定义辅助打印函数
        def print_bbox_table(prefix, title):
            rec_str = f"\n{'-'*30} {title} {'-'*30}\n"

            # 打印 Summary
            map_val = results.get(f'{prefix}/mAP', 0.0)
            nds_val = results.get(f'{prefix}/NDS', 0.0)
            rec_str += f"mAP: {map_val:.4f}\n"
            rec_str += f"NDS: {nds_val:.4f}\n\n"

            # 打印表头
            header = f"{'Object Class':<20} {'AP':<6} {'ATE':<6} {'ASE':<6} {'AOE':<6} {'AVE':<6} {'AAE':<6}\n"
            rec_str += header
            rec_str += "-" * len(header.strip()) + "\n"

            # 遍历类别
            for name in self.CLASSES:
                # 计算平均 AP (NuScenes 标准是 0.5, 1.0, 2.0, 4.0 的平均)
                ap_list = []
                for dist in [0.5, 1.0, 2.0, 4.0]:
                    key = f'{prefix}/{name}_AP_dist_{dist}'
                    if key in results:
                        ap_list.append(results[key])

                # 如果没有该类别的 AP 数据，跳过
                if not ap_list:
                    continue

                ap_val = sum(ap_list) / len(ap_list)

                # 获取各项误差
                ate = results.get(f'{prefix}/{name}_trans_err', 1.0)
                ase = results.get(f'{prefix}/{name}_scale_err', 1.0)
                aoe = results.get(f'{prefix}/{name}_orient_err', 1.0)
                ave = results.get(f'{prefix}/{name}_vel_err', 1.0)
                aae = results.get(f'{prefix}/{name}_attr_err', 1.0)

                # 格式化行 (处理 nan)
                def fmt(val): return f"{val:.3f}" if not np.isnan(val) else "nan"

                row = f"{name:<20} {fmt(ap_val):<6} {fmt(ate):<6} {fmt(ase):<6} {fmt(aoe):<6} {fmt(ave):<6} {fmt(aae):<6}\n"
                rec_str += row

            print_log(rec_str, logger=logger)

        # 1. 打印标准全范围结果
        print_bbox_table('pts_bbox_NuScenes', 'Standard Evaluation (0-Inf)')

        # 2. 打印分段结果 (如果有)
        if depth_ranges:
            for (min_d, max_d) in depth_ranges:
                prefix = f'pts_bbox_NuScenes_{min_d}m_{max_d}m'
                # 检查该范围是否有结果 (防止该范围无目标导致 key 不存在)
                if f'{prefix}/mAP' in results:
                    print_bbox_table(prefix, f'Range Evaluation ({min_d}m - {max_d}m)')

        # 3. 打印解耦分析结果 (Decoupled Analysis)
        if 'decouple/Global_Cls_Acc' in results:
            rec_str = f"\n{'-'*30} Decoupled Analysis {'-'*30}\n"
            g_acc = results['decouple/Global_Cls_Acc'] * 100
            g_loc = results['decouple/Global_Loc_Err']

            rec_str += f"Global Cls Acc: {g_acc:.2f}%\n"
            rec_str += f"Global Loc Err: {g_loc:.3f}m\n\n"

            header = f"{'Object Class':<20} {'Cls Acc':<10} {'Loc Err':<10}\n"
            rec_str += header
            rec_str += "-" * len(header.strip()) + "\n"

            for name in self.CLASSES:
                acc_key = f'decouple/{name}_Cls_Acc'
                loc_key = f'decouple/{name}_Loc_Err'

                if acc_key in results:
                    acc = results[acc_key] * 100
                    loc = results[loc_key]
                    rec_str += f"{name:<20} {acc:6.2f}%    {loc:.3f}m\n"

            print_log(rec_str, logger=logger)

    def evaluate(self, results, metric='bbox', logger=None, depth_ranges=[[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]], **kwargs):
        """
        Evaluation function for both detection and OCC tasks.
        """
        eval_results = {}

        # 统一将 metric 转为 list 处理
        if not isinstance(metric, list):
            metric = [metric]

        # ------------------- 1. OCC Evaluation -------------------
        if 'occ' in metric:
            occ_metric_config = kwargs.get('occ_eval', {})
            occ_results = self._evaluate_occ(results, occ_metric_config, logger)
            eval_results.update(occ_results)

        # ------------------- 2. Depth Evaluation -------------------
        if 'depth' in metric:
            depth_metric_config = kwargs.get('depth_eval', {})
            depth_results = self._evaluate_depth(results, depth_metric_config, logger)
            eval_results.update(depth_results)

        # ------------------- 3. Camera Pose Evaluation -------------------
        if 'camera' in metric:
            camera_metric_config = kwargs.get('camera_eval', {})
            camera_results = self._evaluate_camera(results, camera_metric_config, logger)
            eval_results.update(camera_results)

        # ------------------- 4. BBox Evaluation (Modified) -------------------
        if 'bbox' in metric:
            # 获取 bbox 相关的参数，排除 occ 和 depth 的参数
            jsonfile_prefix = kwargs.get('jsonfile_prefix', None)
            result_names = kwargs.get('result_names', ['pts_bbox'])
            show = kwargs.get('show', False)
            out_dir = kwargs.get('out_dir', None)
            pipeline = kwargs.get('pipeline', None)
            # 默认 pr_dir 保存到与 jsonfile_prefix 相同的目录
            pr_dir_default = osp.join(osp.dirname(jsonfile_prefix), 'pr_curves') if jsonfile_prefix else 'pr_curves'
            pr_dir = kwargs.get('pr_dir', pr_dir_default)

            if show or out_dir:
                self.show(results, out_dir, show=show, pipeline=pipeline)
            else:
                # [关键修改]：手动调用 format_results 和 _evaluate_single
                # 而不是调用 super().evaluate(...)

                # 1. 格式化结果 (调用父类的 format_results)
                result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

                # 处理 result_files 可能是 dict 或 str 的情况
                if isinstance(result_files, str):
                    result_files = {result_names[0]: result_files}

                metric_data_lists = {}  # Store metric_data_list for PR curves

                for name, res_path in result_files.items():
                    print(f'\nEvaluating bboxes of {name}...')

                    # B. 运行标准评估 + 深度分段评估 (调用之前改写的 _evaluate_single)
                    # 注意：确保你的 _evaluate_single 已经接受 depth_ranges 参数
                    bbox_stats, md_dict = self._evaluate_single(res_path, depth_ranges=depth_ranges, result_name=name, return_md_list=True)
                    eval_results.update(bbox_stats)

                    # Store metric_data_dict for PR curve plotting
                    # md_dict 包含: {'standard': metric_data_list, 'depth_ranges': {(min_dist, max_dist): metric_data_list}}
                    metric_data_lists[name] = md_dict

                    # C. 运行解耦分析 (新功能)
                    # 针对主结果进行分析
                    if name == 'pts_bbox' or len(result_files) == 1:
                        decouple_stats = self._analyze_loc_vs_cls(res_path, dist_thresh=2.0)
                        eval_results.update(decouple_stats)

                # D. 绘制 PR 曲线 (新功能)
                if metric_data_lists and pr_dir is not None:
                    import os
                    os.makedirs(pr_dir, exist_ok=True)
                    for name, md_dict in metric_data_lists.items():
                        self._plot_pr_curves(md_dict, output_dir=pr_dir, result_name=name, depth_ranges=depth_ranges)

                # 3. 清理临时文件
                if tmp_dir is not None:
                    tmp_dir.cleanup()

        if 'bbox' in metric:
            self.log_table_results(eval_results, depth_ranges=depth_ranges, logger=logger)

        return eval_results

    def _analyze_loc_vs_cls(self, result_path, dist_thresh=2.0):
        """
        解耦定位与分类分析。
        只根据距离（dist_thresh）进行匹配，忽略类别，从而计算：
        1. 在位置对上的情况下，分类对的概率是多少 (Classification Accuracy)
        2. 在位置对上的情况下，中心点距离误差是多少 (Location Error)
        """
        print(f"\n{'='*20} Decoupled Analysis (Thresh={dist_thresh}m) {'='*20}")
        from nuscenes import NuScenes

        # 1. 加载 NuScenes 数据库 (用于获取 GT)
        # 注意：这里可能会有轻微的耗时，但在评估阶段可接受
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)

        # 2. 加载预测结果
        try:
            with open(result_path, 'r') as f:
                predictions = json.load(f)['results']
        except Exception as e:
            print(f"Failed to load result file for decoupled analysis: {e}")
            return {}

        # 统计容器
        stats = {
            'total_spatial_matches': 0,
            'correct_class_matches': 0,
            'total_loc_error': 0.0,
            'class_details': {c: {'matches': 0, 'correct': 0, 'loc_err': 0.0} for c in self.CLASSES}
        }

        # 遍历每个 Sample
        for sample_token, preds in predictions.items():
            if len(preds) == 0:
                continue

            # --- 获取 GT ---
            sample = nusc.get('sample', sample_token)
            gt_boxes = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                # 映射类别名 (例如 vehicle.car -> car)
                if ann['category_name'] in self.NameMapping:
                    mapped_name = self.NameMapping[ann['category_name']]
                    if mapped_name in self.CLASSES:
                        # 构造简单的 Box 对象，保留 Global 坐标
                        box = NuScenesBox(ann['translation'], ann['size'], Quaternion(ann['rotation']), label=self.CLASSES.index(mapped_name))
                        # 为了后续方便，把 label 名字存进去
                        box.name = mapped_name
                        gt_boxes.append(box)

            if len(gt_boxes) == 0:
                continue

            # --- 几何匹配 (Greedy Matching) ---
            # 提取 GT 和 Pred 的中心点 (X, Y)
            gt_centers = np.array([b.center[:2] for b in gt_boxes])
            pred_centers = np.array([p['translation'][:2] for p in preds])

            # 计算距离矩阵 [Num_GT, Num_Pred]
            dists = np.linalg.norm(gt_centers[:, None, :] - pred_centers[None, :, :], axis=2)

            assigned_preds = set()

            for i in range(len(gt_boxes)):
                min_dist = 1e6
                match_idx = -1

                # 寻找最近的 Pred (忽略类别)
                for j in range(len(preds)):
                    if j in assigned_preds:
                        continue
                    if dists[i, j] < min_dist:
                        min_dist = dists[i, j]
                        match_idx = j

                # 如果距离小于阈值，视为“空间上匹配成功”
                if match_idx != -1 and min_dist < dist_thresh:
                    assigned_preds.add(match_idx)

                    gt_name = gt_boxes[i].name
                    pred_name = preds[match_idx]['detection_name']

                    # 更新统计
                    stats['total_spatial_matches'] += 1
                    stats['total_loc_error'] += min_dist
                    stats['class_details'][gt_name]['matches'] += 1
                    stats['class_details'][gt_name]['loc_err'] += min_dist

                    # 检查分类是否正确
                    if gt_name == pred_name:
                        stats['correct_class_matches'] += 1
                        stats['class_details'][gt_name]['correct'] += 1

        # --- 生成报告和返回字典 ---
        ret_dict = {}
        if stats['total_spatial_matches'] > 0:
            avg_cls_acc = (stats['correct_class_matches'] / stats['total_spatial_matches'])
            avg_loc_err = stats['total_loc_error'] / stats['total_spatial_matches']

            print(f"Total Spatially Matched Objects (dist<{dist_thresh}m): {stats['total_spatial_matches']}")
            print(f"Global Classification Accuracy: {avg_cls_acc*100:.2f}%")
            print(f"Global Location Error (L2):     {avg_loc_err:.3f} m")
            print("-" * 65)
            print(f"{'Class':<20} | {'Cls Acc':<10} | {'Loc Err':<10} | {'Count'}")
            print("-" * 65)

            # 记录到 eval_results 字典中，方便日志记录
            ret_dict['decouple/Global_Cls_Acc'] = avg_cls_acc
            ret_dict['decouple/Global_Loc_Err'] = avg_loc_err

            for cls_name in self.CLASSES:
                c_stats = stats['class_details'][cls_name]
                if c_stats['matches'] > 0:
                    c_acc = c_stats['correct'] / c_stats['matches']
                    l_err = c_stats['loc_err'] / c_stats['matches']
                    print(f"{cls_name:<20} | {c_acc*100:6.2f}%    | {l_err:6.3f}m    | {c_stats['matches']}")

                    # 详细指标
                    ret_dict[f'decouple/{cls_name}_Cls_Acc'] = c_acc
                    ret_dict[f'decouple/{cls_name}_Loc_Err'] = l_err
            print("="*65 + "\n")
        else:
            print("No spatial matches found.")

        return ret_dict

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox',
                         depth_ranges=None,  # 新增参数
                         return_md_list=False): # 新增参数
        """Evaluation for a single model in nuScenes protocol."""
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }

        # 1. 初始化评估器，加载所有数据
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)

        # 2. 运行标准评估 (0-Inf)
        print("Running standard evaluation...")
        # Capture metric_data_list for PR curve plotting
        from nuscenes.eval.detection.data_classes import DetectionMetricDataList
        metrics, metric_data_list = nusc_eval.evaluate()

        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'

        # 记录标准指标
        self._parse_and_add_metrics(detail, metrics.serialize(), metric_prefix)

        # 3. 运行深度分段评估 (如果提供了 depth_ranges)
        depth_range_md_lists = {}  # Store metric_data_list for each depth range
        if depth_ranges is not None:
            # 备份原始的 GT 和 Pred 框，因为后续过滤会修改它们
            # 注意：EvalBoxes 对象比较大，deepcopy 可能稍慢，但在评估阶段通常可以接受
            original_gt_boxes = copy.deepcopy(nusc_eval.gt_boxes)
            original_pred_boxes = copy.deepcopy(nusc_eval.pred_boxes)

            for (min_dist, max_dist) in depth_ranges:
                range_prefix = f"{metric_prefix}_{min_dist}m_{max_dist}m"
                print(f"\nEvaluating for range: {min_dist}m to {max_dist}m ...")

                # 过滤 GT 和 Pred
                # 重要：必须同时过滤 GT 和 Pred。
                # 如果只过滤 Pred 不过滤 GT，远处的 GT 会被视为漏检 (False Negative)，导致 Recall 极低。
                nusc_eval.gt_boxes = filter_eval_boxes_by_range(nusc, original_gt_boxes, min_dist, max_dist)
                nusc_eval.pred_boxes = filter_eval_boxes_by_range(nusc, original_pred_boxes, min_dist, max_dist)

                # 运行评估计算
                # 注意：这里我们只运行 metrics 计算，不重新加载数据
                try:
                    metrics_summary, metric_data_list_range = nusc_eval.evaluate()
                    # 也可以保存这一段的详细结果
                    # nusc_eval.render_curves(os.path.join(output_dir, f'range_{min_dist}_{max_dist}'))

                    # 将分段结果解析并加入到最终的 detail 字典中
                    # 修改 key 名称加上距离前缀
                    self._parse_and_add_metrics(detail, metrics_summary.serialize(), range_prefix)

                    # Store metric_data_list for this depth range
                    depth_range_md_lists[(min_dist, max_dist)] = metric_data_list_range
                except Exception as e:
                    print(f"Warning: Evaluation failed for range {min_dist}-{max_dist}. Maybe no boxes found? Error: {e}")

        if return_md_list:
            # Return dictionary containing standard and depth-range metric_data_lists
            md_dict = {
                'standard': metric_data_list,
                'depth_ranges': depth_range_md_lists
            }
            return detail, md_dict
        else:
            return detail

    def _parse_and_add_metrics(self, detail_dict, metrics, prefix):
        """Helper to parse nuscene metrics and add to dict with prefix."""
        for name in self.CLASSES:
            if name in metrics['label_aps']:
                for k, v in metrics['label_aps'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail_dict['{}/{}_AP_dist_{}'.format(prefix, name, k)] = val
            if name in metrics['label_tp_errors']:
                for k, v in metrics['label_tp_errors'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail_dict['{}/{}_{}'.format(prefix, name, k)] = val

        if 'tp_errors' in metrics:
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail_dict['{}/{}'.format(prefix, self.ErrNameMapping[k])] = val

        detail_dict['{}/NDS'.format(prefix)] = metrics['nd_score']
        detail_dict['{}/mAP'.format(prefix)] = metrics['mean_ap']

    def _evaluate_camera(self, results, camera_config, logger):
        from ..models.utils.pose_enc import pose_encoding_to_extri_intri
        from ..models.utils.geometry import closed_form_inverse_se3
        from ..models.utils.rotation import mat_to_quat

        print_log('Starting camera evaluation', logger=logger)

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

        print_log(f"Average camera evaluation metrics: {camera_results}", logger=logger)

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
        if depth_config is None:
            depth_config = {}

        print_log('Starting depth evaluation', logger=logger)

        vis_depth = depth_config.get('vis_depth', False)
        tmpdepth_dir = depth_config.get('vis_dir', 'tmpdepth')

        save_depth = depth_config.get('save_depth', False)
        save_depth_dir = depth_config.get('save_depth_dir', 'saved_depths')

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
            if save_depth:
                import os
                depth_pred_np = depth_pred.detach().cpu().numpy()

                for cam_idx, (cam_type, cam_info) in enumerate(info['cams'].items()):
                    if cam_idx >= depth_pred_np.shape[0]:
                        continue

                    img_path = cam_info['data_path']
                    img_name = os.path.basename(img_path)

                    cam_save_dir = os.path.join(save_depth_dir, 'samples', cam_type)
                    os.makedirs(cam_save_dir, exist_ok=True)

                    save_name = img_name.replace('.jpg', '.npy')
                    save_path = os.path.join(cam_save_dir, save_name)

                    np.save(save_path, depth_pred_np[cam_idx])
            depth_gt = None
            depth_mask = None
            if 'depth_map' in result_dict:
                depth_gt = result_dict['depth_map'].detach().cpu().numpy()
                depth_mask = result_dict['depth_map_mask'].detach().cpu().numpy()
            else:
                depth_paths = []
                for cam_type, cam_info in info['cams'].items():
                    depth_path = cam_info['data_path'].replace("nuscenes", "nuscenes_depth").replace("jpg", "png")
                    depth_paths.append(depth_path)
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

    def _save_depth_visualizations(self, depth_pred, depth_gt, sample_token, scene_name, output_dir):
        """
        Save depth visualizations for inspection

        Args:
            depth_pred (np.ndarray): Predicted depth
            depth_gt (np.ndarray): Ground truth depth
            sample_token (str): Sample token
            scene_name (str): Scene name
            output_dir (str): Output directory
        """
        import matplotlib.pyplot as plt

        # Get the data info for this sample using O(1) lookup via token2idx
        idx = self.token2idx.get(sample_token)
        if idx is None:
            print_log(f'Could not find data info for sample {sample_token}', logger=None)
            return
        data_info = self.data_infos[idx]

        # Create visualization for each camera view
        N = depth_pred.shape[0]

        for cam_idx in range(N):
            pred_cam = depth_pred[cam_idx]
            gt_cam = depth_gt[cam_idx]

            # Get camera info and original image
            cam_types = list(data_info['cams'].keys())
            if cam_idx >= len(cam_types):
                continue

            cam_type = cam_types[cam_idx]
            cam_info = data_info['cams'][cam_type]
            img_path = cam_info['data_path']

            # Load original image
            try:
                import cv2
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print_log(f'Could not load image {img_path}: {str(e)}', logger=None)
                continue

            # Create figure with 3 rows: original image, predicted depth, ground truth depth
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            fig.suptitle(f'Depth Comparison - {scene_name}_{sample_token} - Camera {cam_idx} ({cam_type})', fontsize=16)

            # Original image
            axes[0, 0].imshow(img_rgb)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            # Original image with depth overlay (simple visualization)
            # Ensure depth map has same dimensions as image for overlay
            # depth_overlay = cv2.applyColorMap(cv2.convertScaleAbs(pred_cam, alpha=255/70), cv2.COLORMAP_JET)

            # # Resize depth overlay to match image dimensions if needed
            # if depth_overlay.shape[:2] != img.shape[:2]:
            #     depth_overlay = cv2.resize(depth_overlay, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # overlay = cv2.addWeighted(img, 0.6, depth_overlay, 0.4, 0)
            # axes[0, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            # axes[0, 1].set_title('Image with Predicted Depth Overlay')
            axes[0, 1].axis('off')

            # Camera info
            axes[0, 2].text(0.1, 0.5, f'Camera Type: {cam_type}\nImage Path: {img_path.split("/")[-1]}',
                           transform=axes[0, 2].transAxes, fontsize=12, verticalalignment='center')
            axes[0, 2].set_title('Camera Info')
            axes[0, 2].axis('off')

            # Predicted depth
            pred_vis = pred_cam.copy()
            pred_mask = pred_vis > 0
            if pred_mask.any():
                pred_min, pred_max = pred_vis[pred_mask].min(), pred_vis[pred_mask].max()
                pred_norm = (pred_vis - pred_min) / (pred_max - pred_min)

                axes[1, 0].imshow(pred_norm, cmap='gray')
                axes[1, 0].set_title(f'Predicted Depth (Gray) - Range: [{pred_min:.2f}, {pred_max:.2f}]m')
                axes[1, 0].axis('off')

                im1 = axes[1, 1].imshow(pred_vis, cmap='jet', vmin=pred_min, vmax=pred_max)
                axes[1, 1].set_title('Predicted Depth (Color)')
                axes[1, 1].axis('off')
                plt.colorbar(im1, ax=axes[1, 1], label='Depth (m)')

                # Histogram
                axes[1, 2].hist(pred_vis[pred_mask].flatten(), bins=50, alpha=0.7)
                axes[1, 2].set_title('Predicted Depth Histogram')
                axes[1, 2].set_xlabel('Depth (m)')
                axes[1, 2].set_ylabel('Frequency')

            # Ground truth depth
            gt_vis = gt_cam.copy()
            gt_mask = gt_vis > 0
            if gt_mask.any():
                gt_min, gt_max = gt_vis[gt_mask].min(), gt_vis[gt_mask].max()
                gt_norm = (gt_vis - gt_min) / (gt_max - gt_min)

                axes[2, 0].imshow(gt_norm, cmap='gray')
                axes[2, 0].set_title(f'Ground Truth Depth (Gray) - Range: [{gt_min:.2f}, {gt_max:.2f}]m')
                axes[2, 0].axis('off')

                im2 = axes[2, 1].imshow(gt_vis, cmap='jet', vmin=gt_min, vmax=gt_max)
                axes[2, 1].set_title('Ground Truth Depth (Color)')
                axes[2, 1].axis('off')
                plt.colorbar(im2, ax=axes[2, 1], label='Depth (m)')

                # Histogram
                axes[2, 2].hist(gt_vis[gt_mask].flatten(), bins=50, alpha=0.7)
                axes[2, 2].set_title('Ground Truth Depth Histogram')
                axes[2, 2].set_xlabel('Depth (m)')
                axes[2, 2].set_ylabel('Frequency')

            plt.tight_layout()

            # Save visualization
            vis_path = f'{output_dir}/{scene_name}_{sample_token}_cam{cam_idx}_vis.png'
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()

    def _load_depth_file(self, filepath):
        """Load a single depth file."""
        try:
            # Try to load as 16-bit PNG depth
            depth = load_16big_png_depth(filepath)
            return depth
        except Exception as e:
            # Fallback: try to load as regular image
            import cv2
            depth = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise ValueError(f"Could not load depth file: {filepath}")
            return depth

    def visualize_depth(self, index=0, output_dir='depth_vis', save_colormap=True, save_3d_projection=False):
        """
        Visualize depth maps from the dataset

        Args:
            index (int): Index of the sample to visualize
            output_dir (str): Directory to save visualization results
            save_colormap (bool): Whether to save depth as colormap
            save_3d_projection (bool): Whether to save 3D point cloud projection
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.mplot3d import Axes3D

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get data info
        info = self.data_infos[index]
        scene_name = self.token2name[info['scene_token']]
        sample_token = info['token']

        print(f"Visualizing depth for sample {index}, scene: {scene_name}, token: {sample_token}")

        # Load depth files
        depth_paths = []
        img_paths = []
        camera_names = []

        for cam_type, cam_info in info['cams'].items():
            depth_path = cam_info['data_path'].replace("nuscenes", "nuscenes_depth").replace("jpg", "png")
            img_path = cam_info['data_path']

            if os.path.exists(depth_path):
                depth_paths.append(depth_path)
                img_paths.append(img_path)
                camera_names.append(cam_type)

        if not depth_paths:
            print(f"No depth files found for sample {index}")
            return

        # Process each camera view
        for i, (depth_path, img_path, cam_name) in enumerate(zip(depth_paths, img_paths, camera_names)):
            # Load depth map
            depth_map = self._load_depth_file(depth_path)

            # Load corresponding image
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Depth Visualization - {cam_name} - Sample {index}', fontsize=16)

            # Original image
            axes[0, 0].imshow(img_rgb)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            # Depth map as grayscale
            depth_vis = depth_map.copy()
            # Normalize for visualization
            depth_min, depth_max = depth_vis[depth_vis > 0].min(), depth_vis[depth_vis > 0].max()
            depth_vis = (depth_vis - depth_min) / (depth_max - depth_min)
            axes[0, 1].imshow(depth_vis, cmap='gray')
            axes[0, 1].set_title(f'Depth Map (Grayscale) - Range: [{depth_min:.2f}, {depth_max:.2f}]m')
            axes[0, 1].axis('off')

            # Depth map as colormap
            if save_colormap:
                im = axes[1, 0].imshow(depth_map, cmap='jet', vmin=depth_min, vmax=depth_max)
                axes[1, 0].set_title('Depth Map (Colormap)')
                axes[1, 0].axis('off')
                fig.colorbar(im, ax=axes[1, 0], label='Depth (m)')

            # 3D point cloud projection
            if save_3d_projection:
                # Create 3D point cloud from depth
                h, w = depth_map.shape
                cam_info = info['cams'][cam_name]

                # Get camera intrinsics
                intrinsic = cam_info['cam_intrinsic']

                # Create pixel coordinate grid
                u, v = np.meshgrid(np.arange(w), np.arange(h))

                # Filter valid depth values
                valid_mask = (depth_map > 0) & (depth_map < 70)  # Filter reasonable depth range

                # Convert depth to 3D points
                z = depth_map[valid_mask]
                u_valid = u[valid_mask]
                v_valid = v[valid_mask]

                # Back-project to 3D camera coordinates
                fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]

                x_cam = (u_valid - cx) * z / fx
                y_cam = (v_valid - cy) * z / fy

                # Transform to world coordinates
                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)

                # Transform points
                points_cam = np.vstack((x_cam, y_cam, z, np.ones_like(z)))
                points_lidar = cam2lidar_rt @ points_cam

                # Plot 3D point cloud
                ax = fig.add_subplot(224, projection='3d')
                ax.scatter(points_lidar[0], points_lidar[1], points_lidar[2],
                          c=z, cmap='jet', s=0.5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('3D Point Cloud (Lidar Frame)')
                ax.set_box_aspect([1, 1, 0.4])  # Adjust aspect ratio

                # Set reasonable axis limits
                max_range = np.array([
                    points_lidar[0].max() - points_lidar[0].min(),
                    points_lidar[1].max() - points_lidar[1].min(),
                    points_lidar[2].max() - points_lidar[2].min()
                ]).max() / 2.0

                mid_x = (points_lidar[0].max() + points_lidar[0].min()) * 0.5
                mid_y = (points_lidar[1].max() + points_lidar[1].min()) * 0.5
                mid_z = (points_lidar[2].max() + points_lidar[2].min()) * 0.5

                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

            plt.tight_layout()

            # Save visualization
            output_path = os.path.join(output_dir, f'depth_vis_{cam_name}_sample_{index}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved depth visualization to {output_path}")
            plt.close()

            # Also save depth as raw numpy array for further analysis
            depth_output_path = os.path.join(output_dir, f'depth_raw_{cam_name}_sample_{index}.npz')
            np.savez(depth_output_path, depth=depth_map)
            print(f"Saved raw depth data to {depth_output_path}")

    def visualize_depth_sequence(self, start_idx=0, length=10, output_dir='depth_seq_vis'):
        """
        Visualize a sequence of depth maps

        Args:
            start_idx (int): Starting index of the sequence
            length (int): Length of the sequence to visualize
            output_dir (str): Directory to save visualization results
        """
        import os
        import matplotlib.pyplot as plt

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print(f"Visualizing depth sequence from index {start_idx}, length {length}")

        # Get sequence indices
        indices = list(range(start_idx, min(start_idx + length, len(self.data_infos))))

        # Visualize each frame
        for idx in indices:
            self.visualize_depth(
                index=idx,
                output_dir=output_dir,
                save_colormap=True,
                save_3d_projection=False  # Skip 3D projection for sequence to save time
            )

        print(f"Sequence visualization complete. Check {output_dir} for results.")


    def _evaluate_occ(self, results, occ_config, logger):
        """
        Evaluate occupancy prediction results.

        Args:
            results (list): List of OCC prediction results
            occ_config (dict): OCC evaluation configuration
            logger (logging.Logger): Logger for output

        Returns:
            dict: OCC evaluation results
        """
        if occ_config is None:
            occ_config = {}

        if 'occ_gt_root' in occ_config:
            occ_gt_root = occ_config['occ_gt_root']
            print_log('Using OCC ground truth for mIoU and RayIoU evaluation', logger=logger)

            # Import necessary modules
            from projects.mmdet3d_plugin.datasets.ego_pose_dataset import EgoPoseDataset
            from projects.mmdet3d_plugin.models.utils.occ_utils import sparse2dense
            from torch.utils.data import DataLoader
            import os

            occ_class_names = occ_config.get('class_names', [])
            free_id = 17

            # Prepare ground truth data and lidar origins
            occ_gts, occ_preds, lidar_origins = [], [], []

            # Get sample tokens
            sample_tokens = [info['token'] for info in self.data_infos]

            with open(self.data_root + 'v1.0-trainval/scene.json', 'r') as f:
                scenes = json.load(f)          # list[dict]
            token2name = {s['token']: s['name'] for s in scenes}

            # Load ground truth data and lidar origins
            for batch in DataLoader(EgoPoseDataset(self.data_infos), num_workers=8):
                token = batch[0][0]
                output_origin = batch[1]

                data_id = sample_tokens.index(token)
                info = self.data_infos[data_id]

                occ_path = os.path.join(occ_config['occ_gt_root'], token2name[info['scene_token']], info['token'], 'labels.npz')
                occ_gt = np.load(occ_path, allow_pickle=True)
                gt_semantics = occ_gt['semantics']
                occ_pred = results[data_id]['occ_pred']
                sem_pred = torch.from_numpy(occ_pred['sem_pred'])  # [B, N]
                occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))  # [B, N, 3]

                occ_size = list(gt_semantics.shape)
                sem_pred, _ = sparse2dense(occ_loc, sem_pred, dense_shape=occ_size, empty_value=free_id)
                sem_pred = sem_pred.squeeze(0).numpy()

                lidar_origins.append(output_origin)
                occ_gts.append(gt_semantics)
                occ_preds.append(sem_pred)

            eval_results = self.eval_miou(results, **occ_config)
            eval_results.update(
                main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names))
            return eval_results

        return {}

    def eval_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        from tqdm import tqdm
        from projects.mmdet3d_plugin.models.utils.occ_utils import sparse2dense

        print('\nStarting Evaluation...')
        metric = Metric_mIoU_Occ3D(
            use_image_mask=eval_kwargs.get('use_image_mask', True),
            use_lidar_mask=eval_kwargs.get('use_lidar_mask', False),
            num_classes=eval_kwargs.get('num_classes', 18),
        )

        occ_root = eval_kwargs.get('occ_gt_root', 'data/nuscenes/gts/')

        for i in tqdm(range(len(occ_results))):
            result_dict = occ_results[i]
            if isinstance(result_dict, dict) and 'occ_pred' in result_dict:
                result_dict = result_dict['occ_pred']

            info = self.get_data_info(i)
            token = info.get('sample_token', info.get('sample_idx'))
            scene_name = info['scene_name']
            occ_file = osp.join(occ_root, scene_name, token, 'labels.npz')
            occ_infos = np.load(occ_file)

            occ_labels = occ_infos['semantics']
            mask_lidar = occ_infos['mask_lidar'].astype(np.bool_)
            mask_camera = occ_infos['mask_camera'].astype(np.bool_)

            occ_pred, _ = sparse2dense(
                torch.from_numpy(result_dict['occ_loc'].astype(np.int64)),
                torch.from_numpy(result_dict['sem_pred']),
                dense_shape=list(occ_labels.shape),
                empty_value=17)
            occ_pred = occ_pred.squeeze(0).numpy()

            metric.add_batch(occ_pred, occ_labels, mask_lidar, mask_camera)

        return {'mIoU': metric.count_miou()}

    def _plot_pr_curves(self, md_dict, output_dir='pr_curves', result_name='pts_bbox', depth_ranges=None):
        """
        绘制 Precision-Recall 曲线并保存。

        Args:
            md_dict (dict): 包含标准评估和深度分段评估的 metric 数据字典
                         {'standard': metric_data_list, 'depth_ranges': {(min_dist, max_dist): metric_data_list}}
            output_dir (str): 输出目录
            result_name (str): 结果名称
            depth_ranges (list): 深度分段范围列表，如 [[0, 10], [10, 20], ...]
        """
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        import os

        print(f"\n{'='*40}")
        print(f"Plotting PR Curves for {result_name}...")
        print(f"{'='*40}")

        # 获取 metric data list
        metric_data_list = md_dict.get('standard', None)
        depth_range_md_lists = md_dict.get('depth_ranges', {})

        # 获取配置信息
        dist_ths = self.eval_detection_configs.dist_ths
        class_names = self.CLASSES

        # 设置绘图参数
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 13
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11

        # 颜色映射（使用不同的颜色区分距离阈值）
        dist_colors = {
            0.5: '#FF6B6B',  # 红色
            1.0: '#4ECDC4',  # 青色
            2.0: '#45B7D1',  # 蓝色
            4.0: '#96CEB4',  # 绿色
        }

        # 线型
        dist_linestyles = {
            0.5: '-',  # 实线
            1.0: '--',  # 虚线
            2.0: '-.',  # 点划线
            4.0: ':',  # 点线
        }
        dist_labels = {
            0.5: 'dist_th=0.5m',
            1.0: 'dist_th=1.0m',
            2.0: 'dist_th=2.0m',
            4.0: 'dist_th=4.0m',
        }

        # ===========================================
        # 1. 构建查找字典: (class_name, dist_th) -> DetectionMetricData
        # ===========================================
        # DetectionMetricDataList 使用 get_dist_data(dist_th) 获取每个距离阈值的所有类别数据
        md_lookup = {}
        for dist_th in dist_ths:
            # get_dist_data returns List[Tuple[DetectionMetricData, class_name]]
            dist_data = metric_data_list.get_dist_data(dist_th)
            for md, cls in dist_data:
                md_lookup[(cls, dist_th)] = md

        # ===========================================
        # 2. 绘制每个类别的 PR 曲线（多距离阈值）
        # ===========================================
        for class_name in class_names:
            fig, ax = plt.subplots(figsize=(10, 8))

            has_data = False
            for dist_th in dist_ths:
                # 获取 metric data
                key = (class_name, dist_th)
                md = md_lookup.get(key)

                if md is None:
                    continue

                # 获取数据
                recall = md.recall
                precision = md.precision

                # 过滤有效数据（非零精度）
                valid_mask = precision > 0
                if valid_mask.sum() > 1:
                    has_data = True
                    ax.plot(
                        recall[valid_mask],
                        precision[valid_mask],
                        color=dist_colors.get(dist_th, '#000000'),
                        linestyle=dist_linestyles.get(dist_th, '-'),
                        linewidth=2.5,
                        label=dist_labels.get(dist_th, f'dist_th={dist_th}m'),
                        alpha=0.8
                    )

            if not has_data:
                plt.close(fig)
                continue

            ax.set_xlabel('Recall', fontweight='bold')
            ax.set_ylabel('Precision', fontweight='bold')
            ax.set_title(f'Precision-Recall Curve - {class_name}', fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', framealpha=0.9)
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
            ax.set_aspect('equal')

            # 添加统计信息
            if has_data:
                # 添加文本框显示关键指标
                textstr = 'Distance Thresholds:\n'
                for dist_th in dist_ths:
                    key = (class_name, dist_th)
                    md = md_lookup.get(key)
                    if md is not None:
                        # 计算平均精度
                        valid_mask = md.precision > 0
                        if valid_mask.sum() > 0:
                            ap = np.trapz(md.precision[valid_mask], md.recall[valid_mask])
                            textstr += f'Dist {dist_th}m: AP={ap:.3f}\n'

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)

            plt.tight_layout()
            save_path = osp.join(output_dir, f'{result_name}_pr_{class_name}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

        # ===========================================
        # 3. 绘制类别 PR 压缩图（所有类别在同一图）
        # ===========================================
        # 为每个距离阈值创建一个图
        for dist_th in dist_ths:
            fig, ax = plt.subplots(figsize=(14, 10))

            palette = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

            for idx, class_name in enumerate(class_names):
                key = (class_name, dist_th)
                md = md_lookup.get(key)

                if md is None:
                    continue

                recall = md.recall
                precision = md.precision
                valid_mask = precision > 0

                if valid_mask.sum() > 1:
                    ax.plot(
                        recall[valid_mask],
                        precision[valid_mask],
                        linewidth=2.0,
                        label=class_name,
                        color=palette[idx],
                        alpha=0.75
                    )

            ax.set_xlabel('Recall', fontweight='bold')
            ax.set_ylabel('Precision', fontweight='bold')
            ax.set_title(f'All Classes PR Curve (Distance Threshold: {dist_th}m)',
                         fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='lower left', ncol=2, framealpha=0.9)
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
            ax.set_aspect('equal')

            plt.tight_layout()
            save_path = osp.join(output_dir, f'{result_name}_pr_all_classes_dist{dist_th}m.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

        # ===========================================
        # 4. 绘制置信度分布分析图
        # ===========================================
        fig, axes = plt.subplots(len(class_names), 1, figsize=(14, 3 * len(class_names)))
        if len(class_names) == 1:
            axes = [axes]

        for idx, class_name in enumerate(class_names):
            ax = axes[idx]
            key = (class_name, 1.0)  # 使用 dist_th=1.0m 作为参考
            md = md_lookup.get(key)

            if md is not None and md.precision.max() > 0:
                confidence = md.confidence[md.precision > 0]
                precision = md.precision[md.precision > 0]
                recall = md.recall[md.precision > 0]

                ax.scatter(confidence, precision, c=recall, cmap='viridis',
                          s=30, alpha=0.6, edgecolors='none')
                ax.set_xlabel('Confidence', fontweight='bold')
                ax.set_ylabel('Precision', fontweight='bold')
                ax.set_title(f'{class_name} - Confidence vs Precision (colored by Recall)',
                            fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')

                cbar = plt.colorbar(ax.collections[0], ax=ax)
                cbar.set_label('Recall', rotation=270, labelpad=15)
            else:
                ax.text(0.5, 0.5, f'No valid data for {class_name}',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        save_path = osp.join(output_dir, f'{result_name}_confidence_analysis.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        # ===========================================
        # 5. 绘制综合分析图（4个子图）
        # ===========================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # (a) 各类别在多个距离阈值下的 PR 曲线（使用 dist_th=1.0m）
        ax = axes[0, 0]
        palette = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
        for idx, class_name in enumerate(class_names):
            key = (class_name, 1.0)
            md = md_lookup.get(key)
            if md is not None and md.precision.max() > 0:
                valid_mask = md.precision > 0
                ax.plot(md.recall[valid_mask], md.precision[valid_mask],
                       linewidth=2, label=class_name, color=palette[idx], alpha=0.7)
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title('All Classes (dist_th=1.0m)', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower left', fontsize=8, ncol=2)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])

        # (b) 单个类别的 PR 曲线（显示不同距离阈值）- 选取car类
        ax = axes[0, 1]
        class_name = 'car'
        for dist_th in dist_ths:
            key = (class_name, dist_th)
            md = md_lookup.get(key)
            if md is not None and md.precision.max() > 0:
                valid_mask = md.precision > 0
                ax.plot(md.recall[valid_mask], md.precision[valid_mask],
                       linewidth=2.5, label=dist_labels.get(dist_th),
                       color=dist_colors.get(dist_th),
                       linestyle=dist_linestyles.get(dist_th), alpha=0.9)
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title(f'{class_name} - Multiple Distance Thresholds', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])

        # (c) 各类别最大 Recall 对比
        ax = axes[1, 0]
        max_recalls = []
        class_labels = []
        for class_name in class_names:
            key = (class_name, 1.0)
            md = md_lookup.get(key)
            if md is not None:
                max_recalls.append(md.max_recall)
                class_labels.append(class_name.replace('_', '\n'))

        if max_recalls:
            colors = ['green' if r > 0.8 else 'orange' if r > 0.6 else 'red' for r in max_recalls]
            bars = ax.bar(range(len(class_labels)), max_recalls, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=0, ha='center')
            ax.set_ylabel('Max Recall', fontweight='bold')
            ax.set_title('Maximum Recall by Class (dist_th=1.0m)', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim([0, 1.05])

            # 添加数值标签
            for bar, val in zip(bars, max_recalls):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # (d) 各类别平均精度对比
        ax = axes[1, 1]
        aps = []
        for class_name in class_names:
            key = (class_name, 1.0)
            md = md_lookup.get(key)
            if md is not None and md.precision.max() > 0:
                # 计算曲线下面积
                valid_mask = md.precision > 0
                ap = np.trapz(md.precision[valid_mask], md.recall[valid_mask])
                aps.append(ap)
            else:
                aps.append(0.0)

        if aps:
            colors = ['green' if a > 0.6 else 'orange' if a > 0.4 else 'red' for a in aps]
            bars = ax.bar(range(len(class_labels)), aps, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=0, ha='center')
            ax.set_ylabel('Average Precision (AP)', fontweight='bold')
            ax.set_title('Average Precision by Class (dist_th=1.0m)', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim([0, 1.05])

            # 添加数值标签
            for bar, val in zip(bars, aps):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            # 添加 mAP 线
            mean_ap = np.mean(aps) if aps else 0.0
            ax.axhline(y=mean_ap, color='red', linestyle='--', linewidth=2,
                      label=f'mAP: {mean_ap:.3f}')
            ax.legend(loc='lower right')

        plt.suptitle(f'{result_name} - PR Curve Analysis Summary',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        save_path = osp.join(output_dir, f'{result_name}_pr_summary.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        print(f"\nPR curves saved to: {output_dir}")
        print(f"  - Individual class PR curves: {result_name}_pr_*.png")
        print(f"  - All classes by distance: {result_name}_pr_all_classes_dist*.png")
        print(f"  - Confidence analysis: {result_name}_confidence_analysis.png")
        print(f"  - Summary plots: {result_name}_pr_summary.png")

        # ===========================================
        # 5. 绘制深度分段 PR 曲线（新功能）
        # ===========================================
        if depth_range_md_lists and depth_ranges is not None:
            print(f"\n{'='*40}")
            print(f"Plotting Depth-Range PR Curves...")
            print(f"{'='*40}")

            # 使用 colormap 动态生成深度范围颜色映射
            cmap = plt.get_cmap('RdYlGn_r')

            def get_depth_color(depth_tuple):
                """根据深度范围在列表中的位置获取对应颜色"""
                if isinstance(depth_tuple, list):
                    depth_tuple = tuple(depth_tuple)
                if depth_tuple in depth_ranges:
                    idx = depth_ranges.index(depth_tuple)
                    color = cmap(idx / max(1, len(depth_ranges) - 1))
                    return color
                return '#000000'  # 默认黑色（理论上不应触发）

            # 为每个深度范围创建子目录
            depth_pr_dir = osp.join(output_dir, 'depth_ranges')
            os.makedirs(depth_pr_dir, exist_ok=True)

            # 构建深度范围的查找字典
            depth_md_lookups = {}
            for (min_dist, max_dist), md_list in depth_range_md_lists.items():
                md_lookup = {}
                for dist_th in dist_ths:
                    # get_dist_data returns List[Tuple[DetectionMetricData, class_name]]
                    dist_data = md_list.get_dist_data(dist_th)
                    for md, cls in dist_data:
                        md_lookup[(cls, dist_th)] = md
                depth_md_lookups[(min_dist, max_dist)] = md_lookup

            # (a) 各类别的深度分段 PR 曲线
            for class_name in class_names:
                fig, ax = plt.subplots(figsize=(12, 8))

                has_data = False
                for (min_dist, max_dist) in depth_ranges:
                    md_lookup = depth_md_lookups.get((min_dist, max_dist), None)
                    if md_lookup is None:
                        continue

                    # 使用 dist_th=1.0m 作为参考距离阈值
                    key = (class_name, 1.0)
                    md = md_lookup.get(key)

                    if md is not None and md.precision.max() > 0:
                        has_data = True
                        valid_mask = md.precision > 0
                        color = get_depth_color((min_dist, max_dist))
                        ax.plot(
                            md.recall[valid_mask],
                            md.precision[valid_mask],
                            linewidth=2.5,
                            label=f'{min_dist}-{max_dist}m',
                            color=color,
                            alpha=0.8
                        )

                if has_data:
                    ax.set_xlabel('Recall', fontweight='bold')
                    ax.set_ylabel('Precision', fontweight='bold')
                    ax.set_title(f'Precision-Recall Curve - {class_name}\n(Segmented by Depth Range)',
                                fontsize=15, fontweight='bold')
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(loc='upper right', framealpha=0.9)
                    ax.set_xlim([0, 1.05])
                    ax.set_ylim([0, 1.05])
                    ax.set_aspect('equal')
                    plt.tight_layout()
                    save_path = osp.join(depth_pr_dir, f'{result_name}_pr_depth_{class_name}.png')
                    plt.savefig(save_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.close(fig)

            # (b) 所有类别的深度分段 PR 曲线对比（使用 2x3 子图布局）
            num_plots = len(depth_ranges)
            num_cols = min(3, num_plots)
            num_rows = (num_plots + num_cols - 1) // num_cols

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows))
            if num_plots == 1:
                axes = [axes]
            elif num_rows == 1:
                axes = [axes] if num_cols == 1 else list(axes)
            else:
                axes = axes.flatten()

            for idx, (min_dist, max_dist) in enumerate(depth_ranges):
                ax = axes[idx]
                md_lookup = depth_md_lookups.get((min_dist, max_dist), None)

                if md_lookup is not None:
                    palette = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
                    for c_idx, class_name in enumerate(class_names):
                        key = (class_name, 1.0)
                        md = md_lookup.get(key)
                        if md is not None and md.precision.max() > 0:
                            valid_mask = md.precision > 0
                            ax.plot(md.recall[valid_mask], md.precision[valid_mask],
                                   linewidth=2, label=class_name, color=palette[c_idx], alpha=0.7)

                ax.set_xlabel('Recall', fontweight='bold')
                ax.set_ylabel('Precision', fontweight='bold')
                ax.set_title(f'Depth Range: {min_dist}-{max_dist}m', fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlim([0, 1.05])
                ax.set_ylim([0, 1.05])
                ax.legend(loc='lower left', fontsize=8, ncol=2)

            # 隐藏多余的子图
            for idx in range(len(depth_ranges), len(axes)):
                axes[idx].axis('off')

            plt.suptitle(f'{result_name} - All Classes PR Curves by Depth Range',
                        fontsize=16, fontweight='bold', y=1.0)
            plt.tight_layout()
            save_path = osp.join(depth_pr_dir, f'{result_name}_pr_depth_all_classes.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            # (c) 单个类别的深度分段对比图（选择 car 类）
            fig, ax = plt.subplots(figsize=(12, 8))
            class_name = 'car'

            for (min_dist, max_dist) in depth_ranges:
                md_lookup = depth_md_lookups.get((min_dist, max_dist), None)
                if md_lookup is None:
                    continue

                # 使用 dist_th=1.0m 作为参考
                key = (class_name, 1.0)
                md = md_lookup.get(key)

                if md is not None and md.precision.max() > 0:
                    valid_mask = md.precision > 0
                    color = get_depth_color((min_dist, max_dist))
                    ax.plot(
                        md.recall[valid_mask],
                        md.precision[valid_mask],
                        linewidth=2.5,
                        label=f'{min_dist}-{max_dist}m',
                        color=color,
                        alpha=0.9
                    )

            ax.set_xlabel('Recall', fontweight='bold')
            ax.set_ylabel('Precision', fontweight='bold')
            ax.set_title(f'{class_name} - PR Curves Segmented by Depth Range',
                        fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', framealpha=0.9)
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
            ax.set_aspect('equal')
            plt.tight_layout()
            save_path = osp.join(depth_pr_dir, f'{result_name}_pr_depth_car.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            # (d) 深度分段对比图 - AP 随距离变化
            fig, ax = plt.subplots(figsize=(14, 8))

            # 为每个深度范围准备数据
            depth_labels = [f'{d[0]}-{d[1]}m' for d in depth_ranges]
            bar_width = 0.6
            x = np.arange(len(depth_ranges))

            # 为每个类别绘制柱状图
            num_classes_to_show = min(10, len(class_names))  # 最多显示10个类别
            palette = plt.cm.tab10(np.linspace(0, 1, num_classes_to_show))

            for idx in range(num_classes_to_show):
                class_name = class_names[idx]
                aps_per_depth = []

                for (min_dist, max_dist) in depth_ranges:
                    md_lookup = depth_md_lookups.get((min_dist, max_dist), None)
                    ap = 0.0
                    if md_lookup is not None:
                        key = (class_name, 1.0)
                        md = md_lookup.get(key)
                        if md is not None and md.precision.max() > 0:
                            valid_mask = md.precision > 0
                            ap = np.trapz(md.precision[valid_mask], md.recall[valid_mask])
                    aps_per_depth.append(ap)

                offset = (idx - num_classes_to_show / 2) * (bar_width / num_classes_to_show)
                bars = ax.bar(x + offset, aps_per_depth, bar_width / num_classes_to_show,
                            label=class_name, color=palette[idx], alpha=0.7, edgecolor='black', linewidth=0.5)

                # 添加数值标签（只对最高的几个）
                for bar, val in zip(bars, aps_per_depth):
                    if val >= 0.1:  # 只显示较大的值，避免拥挤
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                               f'{val:.2f}', ha='center', va='bottom', fontsize=6)

            ax.set_xlabel('Depth Range', fontweight='bold')
            ax.set_ylabel('Average Precision (AP)', fontweight='bold')
            ax.set_title(f'AP by Depth Range (dist_th=1.0m)', fontsize=15, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(depth_labels)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim([0, 1.05])
            ax.legend(loc='lower left', fontsize=8, ncol=2)

            plt.tight_layout()
            save_path = osp.join(depth_pr_dir, f'{result_name}_pr_depth_ap_comparison.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            # (e) 深度分段热力图（类别 vs 深度范围）
            fig, ax = plt.subplots(figsize=(12, 8))

            # 构建 AP 矩阵 [num_classes, num_depth_ranges]
            ap_matrix = []
            for class_name in class_names:
                row = []
                for (min_dist, max_dist) in depth_ranges:
                    md_lookup = depth_md_lookups.get((min_dist, max_dist), None)
                    ap = 0.0
                    if md_lookup is not None:
                        key = (class_name, 1.0)
                        md = md_lookup.get(key)
                        if md is not None and md.precision.max() > 0:
                            valid_mask = md.precision > 0
                            ap = np.trapz(md.precision[valid_mask], md.recall[valid_mask])
                    row.append(ap)
                ap_matrix.append(row)

            ap_matrix = np.array(ap_matrix)

            # 绘制热力图
            im = ax.imshow(ap_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            # 设置坐标轴标签
            ax.set_xticks(np.arange(len(depth_ranges)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels([f'{d[0]}-{d[1]}m' for d in depth_ranges])
            ax.set_yticklabels(class_names)
            ax.set_xlabel('Depth Range', fontweight='bold')
            ax.set_ylabel('Class', fontweight='bold')
            ax.set_title(f'AP Heatmap by Depth Range\n(dist_th=1.0m)',
                        fontsize=15, fontweight='bold')

            # 添加数值标签
            for i in range(len(class_names)):
                for j in range(len(depth_ranges)):
                    val = ap_matrix[i, j]
                    text_color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=text_color, fontsize=9, fontweight='bold')

            # 添加 colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Average Precision (AP)', rotation=270, labelpad=15, fontweight='bold')

            plt.tight_layout()
            save_path = osp.join(depth_pr_dir, f'{result_name}_pr_depth_heatmap.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            # (f) 深度分段综合分析图
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))

            # (a) car 类别的深度分段 PR 曲线
            ax = axes[0, 0]
            class_name = 'car'
            for (min_dist, max_dist) in depth_ranges:
                md_lookup = depth_md_lookups.get((min_dist, max_dist), None)
                if md_lookup is None:
                    continue

                key = (class_name, 1.0)
                md = md_lookup.get(key)
                if md is not None and md.precision.max() > 0:
                    valid_mask = md.precision > 0
                    color = get_depth_color((min_dist, max_dist))
                    ax.plot(md.recall[valid_mask], md.precision[valid_mask],
                           linewidth=2.5, label=f'{min_dist}-{max_dist}m',
                           color=color, alpha=0.9)

            ax.set_xlabel('Recall', fontweight='bold')
            ax.set_ylabel('Precision', fontweight='bold')
            ax.set_title(f'(a) Car - PR by Depth Range', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])

            # (b) pedestrian 类别的深度分段 PR 曲线
            ax = axes[0, 1]
            class_name = 'pedestrian'
            for (min_dist, max_dist) in depth_ranges:
                md_lookup = depth_md_lookups.get((min_dist, max_dist), None)
                if md_lookup is None:
                    continue

                key = (class_name, 1.0)
                md = md_lookup.get(key)
                if md is not None and md.precision.max() > 0:
                    valid_mask = md.precision > 0
                    color = get_depth_color((min_dist, max_dist))
                    ax.plot(md.recall[valid_mask], md.precision[valid_mask],
                           linewidth=2.5, label=f'{min_dist}-{max_dist}m',
                           color=color, alpha=0.9)

            ax.set_xlabel('Recall', fontweight='bold')
            ax.set_ylabel('Precision', fontweight='bold')
            ax.set_title(f'(b) Pedestrian - PR by Depth Range', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])

            # (c) 各类别平均 AP 按深度范围变化
            ax = axes[1, 0]

            # 计算每个深度范围的平均 AP
            depth_range_aps = []
            for (min_dist, max_dist) in depth_ranges:
                md_lookup = depth_md_lookups.get((min_dist, max_dist), None)
                class_aps = []
                if md_lookup is not None:
                    for class_name in class_names:
                        key = (class_name, 1.0)
                        md = md_lookup.get(key)
                        if md is not None and md.precision.max() > 0:
                            valid_mask = md.precision > 0
                            ap = np.trapz(md.precision[valid_mask], md.recall[valid_mask])
                            class_aps.append(ap)
                mAP = np.mean(class_aps) if class_aps else 0.0
                depth_range_aps.append(mAP)

            bars = ax.bar(range(len(depth_ranges)), depth_range_aps,
                        color=[get_depth_color(d) for d in depth_ranges],
                        alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_xticks(range(len(depth_ranges)))
            ax.set_xticklabels(depth_labels)
            ax.set_ylabel('Mean AP', fontweight='bold')
            ax.set_title(f'(c) Mean AP by Depth Range', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim([0, 1.05])

            for bar, val in zip(bars, depth_range_aps):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # (d) 各类别在不同深度范围下的最大 Recall
            ax = axes[1, 1]

            recall_matrix = []
            for class_name in class_names:
                row = []
                for (min_dist, max_dist) in depth_ranges:
                    md_lookup = depth_md_lookups.get((min_dist, max_dist), None)
                    max_recall = 0.0
                    if md_lookup is not None:
                        key = (class_name, 1.0)
                        md = md_lookup.get(key)
                        if md is not None:
                            max_recall = md.max_recall
                    row.append(max_recall)
                recall_matrix.append(row)

            recall_matrix = np.array(recall_matrix)

            # 计算每个深度范围的平均 max recall
            mean_recalls = np.mean(recall_matrix, axis=0) if recall_matrix.size > 0 else np.zeros(len(depth_ranges))

            bars = ax.bar(range(len(depth_ranges)), mean_recalls,
                        color=[get_depth_color(d) for d in depth_ranges],
                        alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_xticks(range(len(depth_ranges)))
            ax.set_xticklabels(depth_labels)
            ax.set_ylabel('Mean Max Recall', fontweight='bold')
            ax.set_title(f'(d) Mean Max Recall by Depth Range', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim([0, 1.05])

            for bar, val in zip(bars, mean_recalls):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.suptitle(f'{result_name} - Depth Range PR Analysis Summary',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            save_path = osp.join(depth_pr_dir, f'{result_name}_pr_depth_summary.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            print(f"\nDepth-range PR curves saved to: {depth_pr_dir}")
            print(f"  - Per-class depth curves: {result_name}_pr_depth_*.png")
            print(f"  - All classes depth curves: {result_name}_pr_depth_all_classes.png")
            print(f"  - Car depth curves: {result_name}_pr_depth_car.png")
            print(f"  - AP comparison: {result_name}_pr_depth_ap_comparison.png")
            print(f"  - AP heatmap: {result_name}_pr_depth_heatmap.png")
            print(f"  - Depth summary: {result_name}_pr_depth_summary.png")


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            # print(f"nus, idx: {idx}")
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

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
