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
import torch
from contextlib import nullcontext
from projects.mmdet3d_plugin.core.utils import force_fp32
from mmdet.models import DETECTORS, build_head, build_neck
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from mmcv.parallel import DataContainer

@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    CAMERA_SEQUENCE_KEYS = (
        'img', 'intrinsics', 'lidar2img', 'extrinsics',
        'cam_extrinsics_global', 'point_mask', 'gt_depth', 'depth_map',
        'depth_map_mask')

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 occ_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 camera_head=None,
                 depth_head=None,
                 occ_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 depth_range=70,
                 occ_seq_length=None,
                 multi_dataset_depth_range=None,
                 save_depth=True,
                 pretrained=None):
        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.bf16_enabled = False
        if camera_head is not None:
            self.camera_head = build_head(camera_head)
        else:
            self.camera_head = None
        if depth_head is not None:
            self.depth_head = build_head(depth_head)
        else:
            self.depth_head = None
        occ_train_cfg = train_cfg.occ if train_cfg is not None and hasattr(train_cfg, 'occ') else train_cfg
        occ_test_cfg = test_cfg.occ if test_cfg is not None and hasattr(test_cfg, 'occ') else test_cfg
        occ_head.update(train_cfg=occ_train_cfg)
        occ_head.update(test_cfg=occ_test_cfg)
        self.occ_head = build_head(occ_head)
        self.occ_neck = build_neck(occ_neck)
        if pts_bbox_head is None:
            self.pts_bbox_head = None

        if img_neck is not None:
            self.patch_start_idx = getattr(self.img_neck, 'patch_start_idx', 0)
        else:
            self.patch_start_idx = getattr(self.img_backbone, 'patch_start_idx', 0)

        self.save_depth = save_depth
        self.use_downsample_depth = self.pts_bbox_head.use_downsample_depth if hasattr(self.pts_bbox_head, 'use_downsample_depth') else True
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.prev_scene_token = None
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False
        self.depth_range = depth_range
        self.multi_dataset_depth_range = multi_dataset_depth_range

        self.scene_token_list = [
            None for _ in range(self.img_backbone.seq_info['batch_size'])]
        self.camera_pos_cache = [
            [None for _ in range(self.img_backbone.seq_info['batch_size'])]
            for _ in range(2)]
        self.seq_length = self.img_backbone.seq_info['seq_length']

        # Per batch slot OCC history, aligned with the sequence sampler order.
        self.occ_memory = []
        backbone_occ_seq_length = self.img_backbone.seq_info.get('seq_length', 1)
        self.occ_seq_length = occ_seq_length if occ_seq_length is not None else backbone_occ_seq_length

    def format_camera_sequence(self, data, img):
        B, T, N = img.shape[:3]
        img = img.permute(0, 2, 1, 3, 4, 5).contiguous()
        for key in self.CAMERA_SEQUENCE_KEYS:
            if key == 'img' or key not in data or data[key] is None:
                continue
            value = data[key]
            if not torch.is_tensor(value) or value.dim() < 3:
                continue
            if value.shape[:3] != (B, T, N):
                continue
            dims = (0, 2, 1, *range(3, value.dim()))
            data[key] = value.permute(*dims).contiguous()

        return img

    def select_sequence_frame(self, data, frame_idx, batch_size, num_views,
                              seq_length):
        frame_data = {}
        flattened_groups = batch_size * num_views
        for key, value in data.items():
            if key in ('img_feats_backbone', 'img_feats_backbone_all',
                       'rescale'):
                continue
            if not torch.is_tensor(value):
                frame_data[key] = value
                continue
            if value.dim() >= 3 and value.shape[:3] == (
                    batch_size, num_views, seq_length):
                frame_data[key] = value[:, :, frame_idx].contiguous()
            elif value.dim() >= 2 and value.shape[:2] == (
                    batch_size, seq_length):
                frame_data[key] = value[:, frame_idx].contiguous()
            elif value.dim() >= 2 and value.shape[:2] == (
                    flattened_groups, seq_length):
                frame_data[key] = value[:, frame_idx].contiguous()
            else:
                frame_data[key] = value
        return frame_data

    def format_roi_head_data(self, data):
        roi_data = dict(data)
        if 'intrinsics' not in roi_data or 'img_feats' not in roi_data:
            return roi_data

        intrinsics = roi_data['intrinsics']
        if not torch.is_tensor(intrinsics) or intrinsics.dim() != 4:
            return roi_data

        batch_size, num_views = roi_data['img_feats'].shape[:2]
        if intrinsics.shape[:2] == (batch_size, num_views):
            roi_data['intrinsics'] = intrinsics.flatten(0, 1)
        return roi_data

    def _get_supervision_loss_scale(self, has_supervision, reference=None):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return 1.0

        if torch.is_tensor(reference):
            device = reference.device
        elif isinstance(reference, (list, tuple)):
            ref = next((feat for feat in reference if feat is not None), None)
            device = ref.device if ref is not None else next(self.parameters()).device
        else:
            device = next(self.parameters()).device

        num_supervised = torch.tensor(
            [1.0 if has_supervision else 0.0],
            device=device,
            dtype=torch.float32)
        torch.distributed.all_reduce(
            num_supervised, op=torch.distributed.ReduceOp.SUM)
        return num_supervised.new_tensor(
            float(torch.distributed.get_world_size())) / num_supervised.clamp(min=1.0)

    def _zero_occ_losses(self):
        zero = None
        for module in (self.occ_neck, self.occ_head):
            for param in module.parameters():
                if param.numel() == 0:
                    continue
                value = param.reshape(-1)[0] * 0.0
                zero = value if zero is None else zero + value

        num_layers = self.occ_head.transformer.num_layers
        losses = dict(init_loss_pts=zero, loss_cls=zero, loss_pts=zero)
        for i in range(num_layers - 1):
            losses[f'd{i}.loss_cls'] = zero
            losses[f'd{i}.loss_pts'] = zero
        return losses

    def _normalize_meta_tensor(self, value):
        if isinstance(value, DataContainer):
            value = value.data
        assert isinstance(value, torch.Tensor), f'expected torch.Tensor, got {type(value)}'
        tensor = value.to(dtype=torch.float32).contiguous()
        if tensor.ndim == 2:
            assert tensor.shape == (4, 4), f'unexpected matrix shape: {tuple(tensor.shape)}'
        else:
            assert tensor.ndim == 3 and tensor.shape[-2:] == (4, 4),                 f'unexpected matrix shape: {tuple(tensor.shape)}'
        return tensor

    def _scale_depth_prediction(self, depth, data):
        if self.multi_dataset_depth_range:
            return depth * self.multi_dataset_depth_range[data['dataset'][0]]
        return depth * self.depth_range

    def _collect_train_depth_losses(self, data):
        target = None
        target_mask = None
        if 'depth_map' in data:
            if data.get('depth_map', None) is not None and data.get(
                    'depth_map_mask', None) is not None:
                target = data['depth_map']
                target_mask = data['depth_map_mask']
        elif data.get('gt_depth', None) is not None and data.get(
                'point_mask', None) is not None:
            target = data['gt_depth']
            target_mask = data['point_mask']

        losses = {}
        if target is not None:
            losses.update(self.depth_head.loss(
                data['depth'], data['depth_conf'], target, target_mask))

        data['depth'] = self._scale_depth_prediction(data['depth'], data)
        if 'depth_map' in data:
            data['pseudo_depth'] = self._scale_depth_prediction(
                data['pseudo_depth'], data)
        if target is not None:
            for key, value in self.depth_head.loss(
                    data['pseudo_depth'], data['pseudo_depth_conf'], target,
                    target_mask).items():
                losses[f'{key}_pseudo_depth'] = value

        if 'depth_map' not in data:
            data['pseudo_depth'] = self._scale_depth_prediction(
                data['pseudo_depth'], data)
        return losses

    def _prepare_train_depth_maps(self, data, B, N, T, image_h, image_w,
                                  feat_h, feat_w):
        data['depth_vggt'] = data['pseudo_depth'].squeeze(-1)
        if self.depth_head.down_ratio == 1 and self.use_downsample_depth:
            data['depth_vggt'] = torch.nn.functional.interpolate(
                data['depth_vggt'],
                scale_factor=1 / self.stride,
                mode='area')

        if self.use_downsample_depth:
            depth_h, depth_w = feat_h, feat_w
        else:
            depth_h, depth_w = image_h, image_w
        data['depth_vggt'] = data['depth_vggt'].reshape(
            B, N, T, depth_h, depth_w)

    def _save_test_depth_outputs(self, data, bbox_list, img_metas):
        for i, result_dict in enumerate(bbox_list):
            if i >= data['depth_vggt'].shape[0]:
                continue
            if self.save_depth:
                result_dict['depth_pred'] = data['depth_vggt'][i]
                result_dict['depth_conf'] = data['depth_conf'][i]
                if 'depth_map' in data:
                    result_dict['depth_map'] = data['depth_map'][i]
                    result_dict['depth_map_mask'] = data['depth_map_mask'][i]
            if 'ida_mat' in img_metas[i]:
                result_dict['ida_mat'] = torch.tensor(img_metas[i]['ida_mat'])
                result_dict['filename'] = img_metas[i]['filename']

    def _prepare_test_depth_maps(self, data, B, N, image_h, image_w, feat_h,
                                 feat_w):
        data['depth_vggt'] = self._scale_depth_prediction(
            data['pseudo_depth'], data).squeeze(-1)
        if self.depth_head.down_ratio == 1 and self.use_downsample_depth:
            data['depth_vggt'] = torch.nn.functional.interpolate(
                data['depth_vggt'],
                scale_factor=1 / self.stride,
                mode='area')

        if self.use_downsample_depth:
            depth_h, depth_w = feat_h, feat_w
        else:
            _, _, depth_h, depth_w = data['img'].shape
        data['depth_vggt'] = data['depth_vggt'].view(B, N, depth_h, depth_w)

    def _get_occ_batch_metas(self, img_metas):
        if len(img_metas) == 1 and isinstance(img_metas[0], tuple):
            return list(img_metas[0])
        if len(img_metas) > 0 and isinstance(img_metas[0], list):
            return [m[0] for m in img_metas]
        return img_metas

    def prepare_occ_inputs(self, backbone_feats, img, img_metas, data):
        batch_metas = self._get_occ_batch_metas(img_metas)
        batch_size = len(batch_metas)
        num_views = img.shape[1]
        occ_seq_length = self.occ_seq_length

        current_backbone_feats = []
        for feat in backbone_feats:
            if feat is None:
                current_backbone_feats.append(None)
                continue
            feat = feat.reshape(
                batch_size, num_views, feat.shape[1], *feat.shape[2:])
            current_backbone_feats.append(feat[:, :, 0].contiguous())

        current_img = img.unsqueeze(2).contiguous()
        current_occ_feats = self.occ_neck(current_backbone_feats, current_img)
        current_occ_feats = [
            feat.reshape(batch_size, num_views, *feat.shape[1:]).contiguous()
            for feat in current_occ_feats
        ]
        while len(self.occ_memory) < batch_size:
            self.occ_memory.append(
                dict(
                    scene_token=None,
                    metas=[],
                    features=[[] for _ in current_occ_feats]))

        reset_flags = [False for _ in range(batch_size)]
        prev_exists = data.get('prev_exists', None)
        if prev_exists is not None:
            if isinstance(prev_exists, DataContainer):
                prev_exists = prev_exists.data
            if not torch.is_tensor(prev_exists):
                prev_exists = torch.as_tensor(prev_exists)
            prev_exists = prev_exists.detach()
            prev_exists = prev_exists.reshape(batch_size, -1)[:, 0]
            reset_flags = (~prev_exists.to(device='cpu', dtype=torch.bool)).tolist()

        feature_history_by_level = [[] for _ in current_occ_feats]
        img_metas_occ = []

        for i, meta in enumerate(batch_metas):
            current_meta = meta.copy()
            for key in ('ego2img', 'ego2occ', 'ego2global'):
                current_meta[key] = self._normalize_meta_tensor(
                    current_meta[key])
            memory = self.occ_memory[i]

            reset_memory = (
                reset_flags[i] or
                memory.get('scene_token', None) != current_meta['scene_token'])
            if reset_memory:
                memory['scene_token'] = current_meta['scene_token']
                memory['metas'] = []
                memory['features'] = [[] for _ in current_occ_feats]
            elif len(memory['features']) != len(current_occ_feats):
                memory['features'] = [[] for _ in current_occ_feats]

            memory['scene_token'] = current_meta['scene_token']
            memory['metas'].insert(0, current_meta)
            memory['metas'] = memory['metas'][:occ_seq_length]
            meta_history = memory['metas'][:occ_seq_length]
            while len(meta_history) < occ_seq_length:
                meta_history.append(meta_history[-1])

            ego2global_curr = current_meta['ego2global']
            ego2img_history = []
            for history_meta in meta_history:
                ego2global_hist = history_meta['ego2global']
                ego2img = history_meta['ego2img']
                ego_hist_to_curr = torch.matmul(
                    self.invert_rigid_transform(ego2global_hist),
                    ego2global_curr)
                ego2img_history.append(torch.matmul(ego2img, ego_hist_to_curr))

            occ_meta = current_meta.copy()
            ego2img_history = [
                value.unsqueeze(0) if value.ndim == 2 else value
                for value in ego2img_history
            ]
            occ_meta['ego2img'] = torch.cat(
                ego2img_history, dim=0).contiguous()
            img_metas_occ.append(occ_meta)

            for lvl, feat in enumerate(current_occ_feats):
                current_live = feat[i]
                level_memory = memory['features'][lvl]
                current_detached = current_live.detach().clone()
                if (reset_memory or len(level_memory) == 0 or
                        level_memory[0].shape != current_detached.shape):
                    level_memory = [current_detached]
                else:
                    level_memory = [current_detached] + \
                        level_memory[:occ_seq_length - 1]
                memory['features'][lvl] = level_memory
                padded_level_memory = level_memory[:occ_seq_length]
                while len(padded_level_memory) < occ_seq_length:
                    padded_level_memory.append(padded_level_memory[-1])
                feature_history_by_level[lvl].append(
                    torch.stack(
                        [current_live] + padded_level_memory[1:occ_seq_length],
                        dim=0))

        occ_img_feats = []
        for lvl, feat in enumerate(current_occ_feats):
            level_batch = torch.stack(
                feature_history_by_level[lvl], dim=0).contiguous()
            occ_img_feats.append(
                level_batch.reshape(
                    batch_size,
                    occ_seq_length * num_views,
                    *level_batch.shape[3:]).contiguous())

        return occ_img_feats, img_metas_occ

    def extract_img_feat(self, img, **data):
        """Extract features of images."""
        B, num_view, T, C, H, W = img.shape
        img = self.grid_mask(img.reshape(B * num_view * T, C, H, W))
        img = img.view(B, num_view, T, C, H, W)
        data_for_backbone = {
            k: v for k, v in data.items()
            if k in self.img_backbone.CAMERA_META_KEYS
        }
        data_for_backbone['img_metas'] = data['img_metas']
        img_feats_all = self.img_backbone(img, **data_for_backbone)
        img_feats = [feat[:, :T].contiguous() for feat in img_feats_all]

        img_feats_neck = self.img_neck(img_feats, img)
        feat_map = img_feats_neck[self.position_level]

        _, C, H, W = feat_map.size()
        img_feats_reshaped = feat_map.view(B, num_view, T, C, H, W)
        return img_feats_reshaped, img_feats, img_feats_all

    @staticmethod
    def invert_rigid_transform(transform):
        """Invert batched SE(3) transforms without a generic matrix inverse."""
        rotation = transform[..., :3, :3]
        translation = transform[..., :3, 3:4]
        rotation_t = rotation.transpose(-1, -2)

        inverse = transform.new_zeros(transform.shape)
        inverse[..., :3, :3] = rotation_t
        inverse[..., :3, 3:4] = -torch.matmul(rotation_t, translation)
        inverse[..., 3, 3] = 1
        return inverse

    def get_camera_pos(self, extrinsics, intrinsics, scene_token_list):
        B = extrinsics.size(0)
        intrinsics = intrinsics[:, :, :3, :3]
        if len(self.scene_token_list) < B:
            extend_len = B - len(self.scene_token_list)
            self.scene_token_list.extend([None for _ in range(extend_len)])
            for cache in self.camera_pos_cache:
                cache.extend([None for _ in range(extend_len)])

        curr_seq_length = self.img_backbone.curr_seq_length
        cached_extrinsics = []
        cached_intrinsics = []
        for i in range(B):
            is_new_scene = self.scene_token_list[i] != scene_token_list[i]
            self.scene_token_list[i] = scene_token_list[i]
            for cache, current, output in zip(
                    self.camera_pos_cache,
                    (extrinsics[i].clone(), intrinsics[i].clone()),
                    (cached_extrinsics, cached_intrinsics)):
                if is_new_scene or cache[i] is None:
                    cache[i] = current.repeat(self.seq_length, 1, 1)
                else:
                    cache[i] = torch.cat((current, cache[i][:-1]), dim=0)
                output.append(cache[i][:curr_seq_length])

        return (
            torch.stack(cached_extrinsics, dim=0),
            torch.stack(cached_intrinsics, dim=0))


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, training=False, **data):
        if (self.aux_2d_only and not self.training and not training) or not self.with_img_roi_head:
            return {'topk_indexes': None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi

    @force_fp32(apply_to=('img_feats'))
    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location(img_metas, **data)
        roi_data = self.format_roi_head_data(data)
        forward_context = nullcontext() if requires_grad else torch.no_grad()
        if not requires_grad:
            self.eval()
        with forward_context:
            outs_roi = self.forward_roi_head(location, True, **roi_data)
            outs = self.pts_bbox_head(
                location, img_metas, outs_roi['topk_indexes'], **data)
        if not requires_grad:
            self.train()

        if return_losses:
            losses = {}
            has_pts_supervision = gt_bboxes_3d is not None and gt_bboxes_3d[0] is not None
            has_roi_supervision = gt_bboxes is not None and gt_bboxes[0] is not None

            if not has_pts_supervision:
                gt_bboxes_3d = [
                    LiDARInstance3DBoxes(data['img_feats'].new_zeros((0, 9)), box_dim=9)
                    for _ in img_metas
                ]
                gt_labels_3d = [
                    data['img_feats'].new_zeros((0,), dtype=torch.long)
                    for _ in img_metas
                ]
            if not has_roi_supervision:
                batch_size, num_views = data['img_feats'].shape[:2]
                gt_bboxes = [[data['img_feats'].new_zeros((0, 4)) for _ in range(num_views)] for _ in range(batch_size)]
                gt_labels = [[data['img_feats'].new_zeros((0,), dtype=torch.long) for _ in range(num_views)] for _ in range(batch_size)]
                centers2d = [[data['img_feats'].new_zeros((0, 2)) for _ in range(num_views)] for _ in range(batch_size)]
                depths = [[data['img_feats'].new_zeros((0,)) for _ in range(num_views)] for _ in range(batch_size)]

            use_multi_dataset_pts = (
                getattr(self.pts_bbox_head, 'multi_dataset_pc_range', None) is not None or
                getattr(self.pts_bbox_head, 'multi_dataset_num_classes', None) is not None)

            if use_multi_dataset_pts:
                loss_inputs = [
                    gt_bboxes_3d, gt_labels_3d, outs,
                    data['dataset'][0]
                ]
            else:
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            pts_losses = self.pts_bbox_head.loss(*loss_inputs)
            if not has_pts_supervision:
                pts_losses = {
                    key: value * 0 if torch.is_tensor(value) else value
                    for key, value in pts_losses.items()
                }
            losses.update(pts_losses)

            if self.with_img_roi_head:
                if use_multi_dataset_pts:
                    loss2d_inputs = [
                        gt_bboxes, gt_labels, centers2d, depths, outs_roi,
                        img_metas, data['dataset'][0]
                    ]
                else:
                    loss2d_inputs = [
                        gt_bboxes, gt_labels, centers2d, depths, outs_roi,
                        img_metas
                    ]
                roi_losses = self.img_roi_head.loss(*loss2d_inputs)
                if not has_roi_supervision:
                    roi_losses = {
                        key: value * 0 if torch.is_tensor(value) else value
                        for key, value in roi_losses.items()
                    }
                losses.update(roi_losses)

            return losses
        return None

    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                if key not in data:
                    continue
                data[key] = list(zip(*data[key]))

            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            if self.pts_bbox_head is not None:
                self.pts_bbox_head.reset_memory()
            self.occ_memory = []
            self.test_flag = False
        if data.get('dataset', None) is not None:
            data['dataset'] = torch.where(
                data['dataset'] == 4, torch.full_like(data['dataset'], 3),
                data['dataset'])
        img = self.format_camera_sequence(data, img)

        data['img_feats'], data['img_feats_backbone'], data[
            'img_feats_backbone_all'] = self.extract_img_feat(
                img,
                img_metas=self._get_occ_batch_metas(img_metas),
                **data)
        # data['img_feats']: B, N, T, C, H, W(Train), B, N, C, H, W(Test T = 1)
        # data['img_feats_backbone']: List[(B * N, T, H * W + cam_token + reg, C)]

        all_loss = {}

        B, N, T, C, H, W = img.shape
        _, _, _, _, h, w = data['img_feats'].shape

        for key in ['cam_extrinsics_global', 'point_mask', 'gt_depth',
                    'depth_map', 'depth_map_mask', 'intrinsics']:
            if key in data and data[key] is not None:
                data[key] = data[key].flatten(0, 1)
        aux_img = img.flatten(0, 1)

        if self.camera_head is not None:
            camera_feats = data['img_feats_backbone_all']
            camera_extrinsics_global = data['cam_extrinsics_global']
            camera_intrinsics = data['intrinsics']
            camera_point_mask = data['point_mask']

            pose_enc_list = self.camera_head(camera_feats)
            scene_token_list = []
            for i in range(B):
                scene_token = img_metas[0][i]['scene_token']
                scene_token_list.extend([scene_token] * N)

            cam_extrinsics, cam_intrinsics = self.get_camera_pos(
                camera_extrinsics_global, camera_intrinsics,
                scene_token_list)
            cam_extrinsics_global_first_inv = self.invert_rigid_transform(
                cam_extrinsics[:, :1])
            cam_extrinsics = torch.matmul(
                cam_extrinsics, cam_extrinsics_global_first_inv)
            trans = cam_extrinsics[:, :, :3, 3] / self.depth_range
            cam_extrinsics[:, :, :3, 3] = trans
            all_loss.update(self.camera_head.loss(
                pose_enc_list, camera_point_mask, cam_extrinsics,
                cam_intrinsics, aux_img))

        if self.depth_head is not None:
            depth_outputs = self.depth_head(
                data['img_feats_backbone'],
                images=aux_img,
                intrinsics=data['intrinsics'],
                patch_start_idx=self.patch_start_idx)
            data['depth'], data['depth_conf'], data['pseudo_depth'], data[
                'pseudo_depth_conf'] = depth_outputs
            all_loss.update(self._collect_train_depth_losses(data))
            self._prepare_train_depth_maps(data, B, N, T, H, W, h, w)


        has_occ_supervision = voxel_semantics is not None and mask_camera is not None
        occ_loss_scale = self._get_supervision_loss_scale(
            has_occ_supervision, data['img_feats_backbone_all'])
        if has_occ_supervision:
            occ_img_feats, img_metas_occ = self.prepare_occ_inputs(
                data['img_feats_backbone_all'],
                img=img[:, :, 0],
                img_metas=img_metas,
                data=data)
            occ_outs = self.occ_head(occ_img_feats, img_metas_occ)
            occ_losses = self.occ_head.loss(
                voxel_semantics, mask_camera, occ_outs)
        else:
            occ_losses = self._zero_occ_losses()

        occ_losses = {
            key: value * occ_loss_scale
            for key, value in occ_losses.items()
        }
        all_loss.update(occ_losses)
        if self.pts_bbox_head is not None:
            frame_idx = 0
            data_t = self.select_sequence_frame(data, frame_idx, B, N, T)
            data_t['img'] = img[:, :, frame_idx].contiguous()
            losses = self.forward_pts_train(
                None if gt_bboxes_3d is None else gt_bboxes_3d[frame_idx],
                None if gt_labels_3d is None else gt_labels_3d[frame_idx],
                None if gt_bboxes is None else gt_bboxes[frame_idx],
                None if gt_labels is None else gt_labels[frame_idx],
                img_metas[frame_idx],
                None if centers2d is None else centers2d[frame_idx],
                None if depths is None else depths[frame_idx],
                requires_grad=True,
                return_losses=True,
                **data_t)
            all_loss.update({
                f'frame_{frame_idx}_' + key: value
                for key, value in losses.items()
            })

        return all_loss


    def forward_test(self, img_metas, rescale, **data):
        self.test_flag = True

        for key in list(data.keys()):
            val = data[key]
            if isinstance(val, DataContainer):
                val = val.data
            while isinstance(val, (list, tuple)):
                val = val[0]
            data[key] = val

        while isinstance(img_metas, (list, tuple)):
            img_metas = img_metas[0]
        img_metas = [img_metas]

        return self.simple_test(img_metas, **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        location = self.prepare_location(img_metas, **data)
        roi_data = self.format_roi_head_data(data)
        outs_roi = self.forward_roi_head(location, **roi_data)
        topk_indexes = outs_roi['topk_indexes']

        B = len(img_metas)
        if not isinstance(self.prev_scene_token, list):
            self.prev_scene_token = [None for _ in range(B)]
        elif len(self.prev_scene_token) < B:
            self.prev_scene_token.extend([None for _ in range(B - len(self.prev_scene_token))])

        prev_exists = data['img'].new_zeros(B, 1)
        reset_memory = False
        for i in range(B):
            if img_metas[i]['scene_token'] != self.prev_scene_token[i]:
                self.prev_scene_token[i] = img_metas[i]['scene_token']
                prev_exists[i, 0] = 0.0
                reset_memory = True
            else:
                prev_exists[i, 0] = 1.0

        data['prev_exists'] = prev_exists
        if reset_memory:
            self.pts_bbox_head.reset_memory()

        outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
        use_multi_dataset_pts = (
            getattr(self.pts_bbox_head, 'multi_dataset_pc_range', None) is not None or
            getattr(self.pts_bbox_head, 'multi_dataset_num_classes', None) is not None)
        if use_multi_dataset_pts:
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, data['dataset'][0])
        else:
            bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, **data):
        """Test function without augmentation."""
        data['img'] = self.format_camera_sequence(data, data['img'])
        B, N, T = data['img'].shape[:3]
        data['img_feats'], data['img_feats_backbone'], data[
            'img_feats_backbone_all'] = self.extract_img_feat(
                img_metas=img_metas, **data)
        if data['img_feats'].dim() == 6:
            data['img_feats'] = data['img_feats'][:, :, 0]

        current_data = self.select_sequence_frame(data, 0, B, N, T)
        for key, value in current_data.items():
            data[key] = value
        if isinstance(img_metas[0], list):
            img_metas = [m[0] for m in img_metas]
        if data.get('dataset', None) is not None:
            data['dataset'] = torch.where(
                data['dataset'] == 4, torch.full_like(data['dataset'], 3),
                data['dataset'])
        B, N, C, H, W = data['img_feats'].shape        # B N C H W
        bbox_list = [dict() for i in range(len(img_metas))]
        if self.camera_head is not None:
            camera_feats = data['img_feats_backbone_all']

            pose_enc_list = self.camera_head(camera_feats)
            scene_token_list = []
            for batch_idx in range(B):
                scene_token = img_metas[batch_idx]['scene_token']
                scene_token_list.extend([scene_token] * N)

            camera_extrinsics_global = data['cam_extrinsics_global'].reshape(
                B * N, 1, *data['cam_extrinsics_global'].shape[2:])
            camera_intrinsics = data['intrinsics'].reshape(
                B * N, 1, *data['intrinsics'].shape[2:])
            cam_extrinsics, cam_intrinsics = self.get_camera_pos(
                camera_extrinsics_global, camera_intrinsics,
                scene_token_list)
            cam_extrinsics_global_first_inv = self.invert_rigid_transform(
                cam_extrinsics[:, :1])
            cam_extrinsics = torch.matmul(
                cam_extrinsics, cam_extrinsics_global_first_inv)
            trans = cam_extrinsics[:, :, :3, 3] / self.depth_range
            cam_extrinsics[:, :, :3, 3] = trans
            bbox_list[0]['cam_pose_pred'] = pose_enc_list[-1]
            bbox_list[0]['cam_extrinsics'] = cam_extrinsics
            bbox_list[0]['img_hw'] =  data['img'].shape[-2:]

        if self.depth_head is not None:
            depth_outputs = self.depth_head(
                data['img_feats_backbone'],
                images=data['img'],
                intrinsics=data['intrinsics'],
                patch_start_idx=self.patch_start_idx)
            data['depth'], data['depth_conf'], data['pseudo_depth'], data[
                'pseudo_depth_conf'] = depth_outputs
            data['depth_vggt'] = self._scale_depth_prediction(
                data['depth'], data)
            self._save_test_depth_outputs(data, bbox_list, img_metas)
            self._prepare_test_depth_maps(
                data, B, N, data['img'].shape[-2], data['img'].shape[-1],
                H, W)

        # Handle Detection task
        if self.pts_bbox_head is not None:
            bbox_pts = self.simple_test_pts(img_metas, **data)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict.update(pts_bbox)

        occ_results = self.simple_test_occ(img_metas, **data)
        for i, result_dict in enumerate(bbox_list):
            if i < len(occ_results):
                result_dict['occ_pred'] = occ_results[i]

        return bbox_list

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def simple_test_occ(self, img_metas, **data):
        occ_img_feats, img_metas_occ = self.prepare_occ_inputs(
            data['img_feats_backbone_all'],
            img=data['img'],
            img_metas=img_metas,
            data=data)
        outs = self.occ_head(occ_img_feats, img_metas_occ)
        results = self.occ_head.get_occ(outs, img_metas_occ)
        for res in results:
            if res['sem_pred'].ndim == 1:
                res['sem_pred'] = res['sem_pred'][None, ...]
            if res['occ_loc'].ndim == 2:
                res['occ_loc'] = res['occ_loc'][None, ...]
        return results
