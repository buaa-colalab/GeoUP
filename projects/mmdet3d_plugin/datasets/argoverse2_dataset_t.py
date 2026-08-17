# Copyright (c) 2023 megvii-model. All Rights Reserved.

import pandas as pd
import torch
import numpy as np
import cv2
from mmdet.datasets import DATASETS
from av2.evaluation.detection.constants import CompetitionCategories
from pathlib import Path
from .argoverse2_dataset import Argoverse2Dataset
import math
from mmcv.parallel import DataContainer as DC
import random
from .av2_utils import DetectionCfg
from .av2_eval_util import evaluate
from av2.utils.io import read_feather
from os import path as osp
from mmcv.utils import print_log

LABEL_ATTR = (
    "tx_m","ty_m","tz_m","length_m","width_m","height_m","qw","qx","qy","qz",
)

@DATASETS.register_module()
class Argoverse2DatasetT(Argoverse2Dataset):
    CLASSES = tuple(x.value for x in CompetitionCategories)
    
    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, interval_test=False, vggt_mode=False, max_dist=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        self.vggt_mode = vggt_mode
        self.max_dist = max_dist

        if interval_test:
            data_infos = self.data_infos
            s1, s2, s3, s4, s5 = data_infos[::5], data_infos[1::5], data_infos[2::5], data_infos[3::5], data_infos[4::5]
            data_infos = s1 + s2 + s3 + s4 + s5 
            self.data_infos = data_infos
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []
        scene_id = None

        curr_sequence = -1
        for idx in range(len(self.data_infos)):
            if self.data_infos[idx]['scene_id'] != scene_id:
                # Not first frame and # of sweeps is 0 -> new sequence
                scene_id = self.data_infos[idx]['scene_id']
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
                # assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                
                self.flag = np.array(new_flags, dtype=np.int64)

    def _build_depth_path(self, image_path):
        """Build the AV2 depth path that corresponds to a camera image path."""
        depth_dir = 'depth' if self.max_dist is None else 'depth_limit'
        return str(image_path).replace('.jpg', '.npy').replace('cameras', depth_dir)


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
    
    @staticmethod
    def _first_test_aug_value(value):
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    def union2one_test(self, queue):
        for key in self.collect_keys:
            values = [self._first_test_aug_value(each[key]) for each in queue]
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each.data for each in values]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each.data for each in values], cpu_only=True)

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
        
        city_SE3_ego = info['city_SE3_ego_lidar_t'] # ego -> global
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = city_SE3_ego.rotation
        transform_matrix[:3, 3] = city_SE3_ego.translation

        nus_lidar_to_av2_lidar = np.array([[0, 1, 0, 0],
                                            [-1,  0, 0, 0],
                                            [0,  0, 1, 0],
                                            [0,  0, 0, 1]])

        ego_pose =  transform_matrix @ nus_lidar_to_av2_lidar

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        pts_filename = Path(self.split)/ info['scene_id']  / 'sensors' /  'lidar' / f"{info['lidar_timestamp_ns']}.feather"
        input_dict = dict(
            dataset='av2',
            pts_filename=self.data_root / pts_filename,
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            scene_token=info['scene_id'],
            timestamp=index,
            lidar_timestamp=info['lidar_timestamp_ns'],
        )

        if self.modality['use_camera']:
            image_paths = []
            depth_paths = []
            image_raw_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            cam_extrinsics_global = []
            city_SE3_ego_lidar_t = info['city_SE3_ego_lidar_t']
            for cam_type, cam_info in info['cam_infos'].items():
                if cam_info is None:
                    print('WARNING: no camera data for {}'.format(info['scene_id']))
                    return None
                img_timestamp.append(cam_info['cam_timestamp_ns']/ 1e9)
                image_path = self.data_root / cam_info['fpath']
                image_paths.append(image_path)
                depth_path = self._build_depth_path(image_path)
                depth_paths.append(depth_path)
                image_raw_paths.append(cam_info['fpath'])
                # obtain lidar to image transformation matrix
                city_SE3_ego_cam_t = cam_info['city_SE3_ego_cam_t']
                ego_SE3_cam = cam_info['ego_SE3_cam']
                ego_cam_t_SE3_ego_lidar_t = city_SE3_ego_cam_t.inverse().compose(city_SE3_ego_lidar_t) #ego2glo_lidar -> glo2ego_cam
                cam_SE3_ego_cam_t = ego_SE3_cam.inverse().compose(ego_cam_t_SE3_ego_lidar_t) #ego -> cam
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = cam_SE3_ego_cam_t.rotation
                transform_matrix[:3, 3] = cam_SE3_ego_cam_t.translation
                transform_matrix = transform_matrix  @ nus_lidar_to_av2_lidar

                intrinsic = cam_info['intrinsics']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ transform_matrix)
                intrinsics.append(viewpad)
                extrinsics.append(transform_matrix)
                lidar2img_rts.append(lidar2img_rt)
                cam_extrinsics_global.append(transform_matrix @ ego_pose_inv) 
            
            if not self.test_mode:
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None
            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                    depth_filename=depth_paths,
                    cam_extrinsics_global=cam_extrinsics_global,
                ))
        if not self.test_mode or True:
            annos = self.get_ann_info(index, input_dict)
            gt2d_infos = info['gt2d_infos']
            # gt2d_infos = self.remove_classes_2d(gt2d_infos)
            if self.max_dist:
                gt2d_infos = self.filter_gt2d_infos_by_distance(gt2d_infos, self.max_dist)
            annos.update( 
                dict(
                    bboxes=gt2d_infos['gt_2dbboxes'],
                    labels=gt2d_infos['gt_2dlabels'],
                    centers2d=gt2d_infos['centers2d'],
                    depths=gt2d_infos['depths'],
                    bboxes_ignore=None)
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
            for img_idx, img_path in enumerate(input_dict['img_filename']): # Process first two images
                img_2d = cv2.imread(img_path)
                if len(bboxes2d) > img_idx and len(labels2d) > img_idx:
                    for bbox, label in zip(bboxes2d[img_idx], labels2d[img_idx]):
                        color = tuple(c * 255 for c in colors(label)[:3])
                        cv2.rectangle(img_2d, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        class_name = self.CLASSES[label]
                        cv2.putText(img_2d, class_name, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                output_path_2d = f'vis/av2/result_2d_{img_idx}_av2.jpg'
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

                output_path_3d = f'vis/av2/result_3d_{img_idx}_av2.jpg'
                cv2.imwrite(output_path_3d, img_3d)
                print(f"Saved 3D visualization to {output_path_3d}")
        # breakpoint()
            
        return input_dict

    def filter_gt2d_infos_by_distance(self, gt2d_infos, max_dist):
        """Filter ground truth 2D annotations by distance.

        Args:
        gt2d_infos (dict): Ground truth 2D annotations.
        max_dist (float): Maximum distance for filtering.

        Returns:
        dict: Filtered ground truth 2D annotations.
        """
        cam_num = len(gt2d_infos['gt_2dlabels'])
        filtered_infos = {}
        index_list = [np.where(depths < max_dist)[0] for depths in gt2d_infos['depths']]

        for key in gt2d_infos.keys():
                filtered_infos[key] = [gt2d_infos[key][cam_id][index_list[cam_id]] for cam_id in range(cam_num)] 
        return filtered_infos

    def remove_classes_2d(self, ann_info, classes_to_remove=['STROLLER', 'WHEELCHAIR', 'WHEELED_DEVICE', 'DOG', 'SIGN', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN']):
        img_filtered_annotations = {}
        av2_class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
               'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
               'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
               'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
               'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
               'WHEELCHAIR', 'WHEELED_DEVICE','WHEELED_RIDER']
        cam_num = len(ann_info['gt_2dlabels'])
        relevant_annotation_indices = [[] for _ in range(cam_num)]
        for cam_id in range(cam_num):
            relevant_annotation_indices[cam_id] = [
                i for i, x in enumerate(ann_info['gt_2dlabels'][cam_id]) if av2_class_names[x] not in classes_to_remove
            ]
        
        for cam_id in range(cam_num):
            for key in ann_info.keys():
                if img_filtered_annotations.get(key, None) is None:
                    shape = ann_info[key][0].shape
                    if len(shape) == 1:
                        new_shape = (0,)
                    else:
                        new_shape = (0, *shape[1:])
                    empty_array = np.empty(new_shape, dtype=ann_info[key][0].dtype)
                    img_filtered_annotations[key] = [empty_array for _ in range(cam_num)]
                if len(relevant_annotation_indices[cam_id]) == 0:
                    continue
                img_filtered_annotations[key][cam_id] = (ann_info[key][cam_id][relevant_annotation_indices[cam_id]])
        av2_to_nuscenes = {
            'REGULAR_VEHICLE': 'car',
            'LARGE_VEHICLE': 'truck',
            'TRUCK': 'truck',
            'BOX_TRUCK': 'truck',
            'TRUCK_CAB': 'truck',
            'BUS': 'bus',
            'SCHOOL_BUS': 'bus',
            'ARTICULATED_BUS': 'bus',
            'VEHICULAR_TRAILER': 'trailer',
            'MESSAGE_BOARD_TRAILER': 'trailer',
            'CONSTRUCTION_CONE': 'traffic_cone',
            'CONSTRUCTION_BARREL': 'barrier',
            'BOLLARD': 'barrier',
            'BICYCLE': 'bicycle',
            'MOTORCYCLE': 'motorcycle',
            'BICYCLIST': 'pedestrian',  # 骑手→行人（nuScenes 无 rider）
            'MOTORCYCLIST': 'pedestrian',
            'WHEELED_RIDER': 'pedestrian',
            'PEDESTRIAN': 'pedestrian',
            # 以下无对应，映射为 None 或 ignore
            'STROLLER': None,
            'WHEELCHAIR': None,
            'WHEELED_DEVICE': None,
            'DOG': None,
            'SIGN': None,
            'STOP_SIGN': None,
            'MOBILE_PEDESTRIAN_CROSSING_SIGN': None,
        }
        for cam_id in range(cam_num):
            labels = img_filtered_annotations['gt_2dlabels'][cam_id]
            if len(labels) == 0:
                continue
            name = [av2_to_nuscenes[av2_class_names[x]] for x in labels]
            img_filtered_annotations['gt_2dlabels'][cam_id] = np.array([self.CLASSES.index(x) if x is not None else -1 for x in name])
        return img_filtered_annotations


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            data = self.prepare_test_data(idx)
            count = 1
            while data is None:
                data = self.prepare_test_data(idx + count)
                count += 1
            return data
        while True:
            # print(f'av2, idx: {idx}')
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
        import copy
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        new_results = []
        for result in results:
            new_result = copy.deepcopy(result)
            bbox_3d = result['boxes_3d'].tensor
            x, y, z = bbox_3d[:, 0:1], bbox_3d[:, 1:2], bbox_3d[:, 2:3]
            # size = bbox_3d[:, 3:6]
            yaw = bbox_3d[:, 6:7]
            new_x, new_y, new_z = y, -x, z
            # new_size = size[:, [1, 0, 2]]
            new_yaw = yaw - np.pi / 2
            new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi
            bbox_3d = torch.cat([new_x, new_y, new_z, bbox_3d[:, 3:6], new_yaw, bbox_3d[:, 7:]], dim=1)

            label_3d = result['labels_3d']
            # # car 0->0, pedestrian 8->1, cyclist 7->2, other *->3
            # label_mapping = {0: 0, 8: 1, 7: 2}
            # new_label_3d = []
            # for label in label_3d:
            #     if label in label_mapping:
            #         new_label_3d.append(label_mapping[label])
            #     else:
            #         new_label_3d.append(0)
            # new_label_3d = torch.tensor(new_label_3d).to(label_3d)

            new_result['boxes_3d'] = LiDARInstance3DBoxes(bbox_3d, box_dim=bbox_3d.shape[-1])
            # new_result['pts_bbox']['labels_3d'] = new_label_3d
            new_results.append(new_result)
        return new_results
    
    def visualize_predictions(
        self,
        dts,
        gts,
        output_dir,
        num_samples_to_viz=10,
        score_thresh=0.3
    ):
        """
        在BEV图上可视化预测结果和真值。
        此版本专门处理 dts 是 gts 的子集的情况。

        Args:
            dts (pd.DataFrame): 格式化后的预测结果 (可能只是一个子集).
            gts (pd.DataFrame): 格式化后的完整真值.
            output_dir (str): 保存可视化图片的目录.
            num_samples_to_viz (int): 可视化的样本数量.
            score_thresh (float): 用于过滤预测框的分数阈值.
        """
        import matplotlib.pyplot as plt
        import mmcv
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        print(f"\nStarting visualization for subset of predictions...")
        mmcv.mkdir_or_exist(output_dir)

        # ==================== 核心修改在这里 ====================
        # 1. 从预测结果(dts)中获取唯一的样本ID，而不是从全部真值(gts)中获取。
        #    这确保了我们只可视化那些我们有预测结果的样本。
        unique_samples_with_preds = dts.index.unique()

        if unique_samples_with_preds.empty:
            print("Warning: The 'dts' (predictions) DataFrame is empty. No samples to visualize.")
            return
        
        # 2. 如果预测的样本数大于我们想可视化的数量，就从中随机抽样。
        #    否则，就可视化所有有预测的样本。
        if len(unique_samples_with_preds) > num_samples_to_viz:
            indices = np.random.choice(len(unique_samples_with_preds), num_samples_to_viz, replace=False)
            samples_to_viz = unique_samples_with_preds[indices]
        else:
            samples_to_viz = unique_samples_with_preds
        # ==========================================================

        print(f"Found {len(unique_samples_with_preds)} samples with predictions. Visualizing {len(samples_to_viz)} of them.")

        for log_id, timestamp_ns in mmcv.track_iter_progress(samples_to_viz):
            # 查找并加载点云
            info = next((item for item in self.data_infos if item['scene_id'] == log_id and item['lidar_timestamp_ns'] == timestamp_ns), None)
            if info is None:
                print(f"Warning: Could not find data_info for {log_id}/{timestamp_ns}")
                continue
            
            pts_filename = Path(self.data_root) / self.split / info['scene_id'] / 'sensors' / 'lidar' / f"{info['lidar_timestamp_ns']}.feather"
            points_df = read_feather(pts_filename)
            points = np.stack([points_df["x"], points_df["y"], points_df["z"]], axis=-1)

            # 获取当前样本的预测(dts)和真值(gts)
            # 使用 try-except 增加鲁棒性
            try:
                # 从完整的gts中精确查找当前样本的真值
                sample_gts = gts.loc[[(log_id, timestamp_ns)]]
            except KeyError:
                print(f"Warning: No ground truth found for sample {log_id}/{timestamp_ns}. Skipping visualization for this sample.")
                continue

            # 从dts子集中获取当前样本的预测
            sample_dts = dts.loc[[(log_id, timestamp_ns)]]
            sample_dts = sample_dts[sample_dts['score'] > score_thresh]

            # 将DataFrame格式的盒子转换为8个角点的格式以便绘图
            def get_corners_from_df(df):
                if df.empty:
                    return np.zeros((0, 8, 3))
                
                xyz = df[['tx_m', 'ty_m', 'tz_m']].values
                lwh = df[['length_m', 'width_m', 'height_m']].values
                # 确保四元数是 (w, x, y, z) 顺序
                quat_wxyz = df[['qw', 'qx', 'qy', 'qz']].values
                
                # 使用MMDetection3D的LiDARInstance3DBoxes和四元数来获取角点
                boxes = LiDARInstance3DBoxes(
                    torch.from_numpy(np.hstack([xyz, lwh])),
                    box_dim=6
                )
                boxes.q = torch.from_numpy(quat_wxyz).float()
                return boxes.corners.numpy()
         
            gt_corners = get_corners_from_df(sample_gts)
            dt_corners = get_corners_from_df(sample_dts)

            # --- 绘图 ---
            fig, ax = plt.subplots(figsize=(12, 12))
            
            ax.scatter(points[:, 0], points[:, 1], s=0.2, c='gray', alpha=0.5)

            # 绘制GT盒子 (绿色)
            for corners in gt_corners:
                # corners是(8, 3)，绘制底座
                for i in range(4):
                    ax.plot([corners[i, 0], corners[(i + 1) % 4, 0]], 
                            [corners[i, 1], corners[(i + 1) % 4, 1]], color='green', linewidth=2)
                # 绘制方向（从中心指向前方的线）
                front_center = corners[[0, 1]].mean(axis=0)
                rear_center = corners[[2, 3]].mean(axis=0)
                center = (front_center + rear_center) / 2
                ax.plot([center[0], front_center[0]], [center[1], front_center[1]], color='lime', linewidth=2)

            # 绘制DT盒子 (红色)
            for corners in dt_corners:
                # 绘制底座
                for i in range(4):
                    ax.plot([corners[i, 0], corners[(i + 1) % 4, 0]], 
                            [corners[i, 1], corners[(i + 1) % 4, 1]], color='red', linewidth=1.5, linestyle='--')
                # 绘制方向
                front_center = corners[[0, 1]].mean(axis=0)
                rear_center = corners[[2, 3]].mean(axis=0)
                center = (front_center + rear_center) / 2
                ax.plot([center[0], front_center[0]], [center[1], front_center[1]], color='magenta', linewidth=1.5)

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_title(f"Log: {log_id}\nTimestamp: {timestamp_ns}")
            ax.set_xlabel("X (m) - Forward")
            ax.set_ylabel("Y (m) - Left")
            plt.grid(True)
            
            output_path = Path(output_dir) / f"{log_id}_{timestamp_ns}.png"
            plt.savefig(output_path, dpi=150)
            plt.close(fig)

        print(f"Visualizations saved to {output_dir}")

    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 load_from=None,
                 jsonfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 eval_range_m=None,
                 **kwargs):
        # from av2.evaluation.detection.utils import DetectionCfg
        # from av2.evaluation.detection.eval import evaluate
        # from av2.utils.io import read_all_annotations, read_feather

        # dts = self.format_results(results, jsonfile_prefix, submission_prefix)

        # result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
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
            if load_from is None:
                dts = self.format_results(results, jsonfile_prefix, submission_prefix)
            else:
                dts = pd.read_feather(load_from)
                dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()
                print(f'Result is loaded from {load_from}.')

            val_anno_path = osp.join(self.data_root, 'val_anno.feather')
            gts = read_feather(val_anno_path)

            gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")

            # viz_output_dir = Path(jsonfile_prefix).parent / "bev_visualizations" if jsonfile_prefix else Path(out_dir) / "bev_visualizations"
            # self.visualize_predictions(dts, gts, output_dir=str(viz_output_dir))

            valid_uuids_gts = gts.index.tolist()
            valid_uuids_dts = dts.index.tolist()
            valid_uuids = set(valid_uuids_gts) & set(valid_uuids_dts)
            gts = gts.loc[list(valid_uuids)].sort_index()

            categories = set(x.value for x in CompetitionCategories)
            categories &= set(gts["category"].unique().tolist())  # 交集
            split_dir = Path(self.data_root) / self.split
            cfg = DetectionCfg(
                dataset_dir=split_dir,
                categories=tuple(sorted(categories)),
                eval_range_m=[0.0, 50.0] if eval_range_m is None else eval_range_m,
                eval_only_roi_instances=True,
            )
            eval_dts, eval_gts, metrics, recall3d = evaluate(dts.reset_index(), gts.reset_index(), cfg)
            valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
            print(metrics.loc[valid_categories])
            ap_dict = {}
            for index, row in metrics.iterrows():
                ap_dict[index] = row.to_json()
            
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
                depth_paths = []
                for cam_type, cam_info in info['cam_infos'].items():
                    image_path = self.data_root / cam_info['fpath']
                    depth_path = self._build_depth_path(image_path)
                    depth_paths.append(depth_path)
                try:
                    missing_depth_paths = [name for name in depth_paths if not osp.exists(name)]
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
                        import os
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
