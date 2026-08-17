import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob

# from mmcv.runner import force_fp32
from projects.mmdet3d_plugin.core.utils import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, normalize_bbox_center, normalize_far_bbox_center
import torch.nn.functional as F
from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from mmcv.cnn import Conv2d, Linear
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from torch.distributions import Beta


@HEADS.register_module()
class StreamPETRHeadVGGT(StreamPETRHead):
    _version = 1

    def __init__(self,
                 # 3dppe part
                 positional_encoding=dict(
                     type='SinePositionalEncoding3D', num_feats=128, normalize=True),
                 use_sigmoid=True,
                 use_detach=False,
                 share_pe_encoder=False,
                 with_2dpe_only=False,
                 with_pos_info=False,
                 with_position=True,
                 with_multiview=True,
                 with_vggt_depth=False,
                 raydn_group=1,
                 raydn_num=5,
                 raydn_alpha=8,
                 raydn_beta=2,
                 raydn_radius=3,
                 multi_dataset_pc_range=None,
                 multi_dataset_position_range=None,
                 multi_dataset_num_classes=None,
                 split_kitti_waymo_cls_head=False,
                 init_query=None,
                 use_downsample_depth=True,
                 use_conv_downsample=False,
                 use_dpt_feat=False,
                 use_cam_info=False,
                 use_dpt_offset=False,
                 use_masked_3dppe=False,
                 masked_3dppe_mode='soft',
                 masked_3dppe_quantile=0.7,
                 masked_3dppe_temperature=12.0,
                 masked_3dppe_min_scale=0.1,
                 normalize=False,
                 normalize_far=False,
                 use_multi_reg_head=False,
                 **kwargs):
        self.use_multi_reg_head = use_multi_reg_head
        self.normalize = normalize
        self.normalize_far = normalize_far
        self.use_dpt_feat = use_dpt_feat
        self.use_cam_info = use_cam_info
        self.use_dpt_offset = use_dpt_offset
        self.use_masked_3dppe = use_masked_3dppe
        self.masked_3dppe_mode = masked_3dppe_mode
        self.masked_3dppe_quantile = masked_3dppe_quantile
        self.masked_3dppe_temperature = masked_3dppe_temperature
        self.masked_3dppe_min_scale = masked_3dppe_min_scale
        self.use_downsample_depth = use_downsample_depth
        self.use_conv_downsample = use_conv_downsample
        self.use_sigmoid = use_sigmoid
        self.use_detach = use_detach  # detach depth score
        self.share_pe_encoder = share_pe_encoder
        self.with_2dpe_only = with_2dpe_only
        self.with_pos_info = with_pos_info
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.init_query = init_query
        self.with_vggt_depth = with_vggt_depth
        self.multi_dataset_pc_range = multi_dataset_pc_range
        self.multi_dataset_position_range = multi_dataset_position_range
        self.split_kitti_waymo_cls_head = split_kitti_waymo_cls_head
        self.multi_dataset_num_classes = self._prepare_multi_dataset_num_classes(
            multi_dataset_num_classes)
        if self.multi_dataset_num_classes:
            self.multi_dataset_cls_out_channels = []
            for num_class in self.multi_dataset_num_classes:
                if kwargs['loss_cls']['use_sigmoid']:
                    cls_out_channels = num_class
                else:
                    cls_out_channels = num_class + 1
                self.multi_dataset_cls_out_channels.append(cls_out_channels)
        else:
            self.multi_dataset_cls_out_channels = None
        self.raydn_group=raydn_group
        self.raydn_num=raydn_num
        self.raydn_alpha=raydn_alpha
        self.raydn_beta=raydn_beta
        self.raydn_radius=raydn_radius
        self.raydn_sampler = Beta(raydn_alpha, raydn_beta)

        super(StreamPETRHeadVGGT, self).__init__(**kwargs)

        self.positional_encoding = build_positional_encoding(
            positional_encoding)

    def _prepare_multi_dataset_num_classes(self, multi_dataset_num_classes):
        if multi_dataset_num_classes is None:
            return None

        multi_dataset_num_classes = list(multi_dataset_num_classes)
        if self.split_kitti_waymo_cls_head:
            if len(multi_dataset_num_classes) == 3:
                return [
                    multi_dataset_num_classes[0],
                    multi_dataset_num_classes[1],
                    multi_dataset_num_classes[1],
                    multi_dataset_num_classes[2],
                ]
            if len(multi_dataset_num_classes) == 4:
                return multi_dataset_num_classes
            raise ValueError(
                'Expected 3 or 4 entries in multi_dataset_num_classes when '
                'split_kitti_waymo_cls_head=True, got '
                f'{len(multi_dataset_num_classes)}.')

        if len(multi_dataset_num_classes) != 3:
            raise ValueError(
                'Expected 3 entries in multi_dataset_num_classes when '
                'split_kitti_waymo_cls_head=False, got '
                f'{len(multi_dataset_num_classes)}.')
        return multi_dataset_num_classes

    def _normalize_cls_dataset_idx(self, dataset_idx):
        if torch.is_tensor(dataset_idx):
            dataset_idx = int(dataset_idx.item())
        elif isinstance(dataset_idx, np.ndarray):
            dataset_idx = int(dataset_idx.item())
        elif isinstance(dataset_idx, (list, tuple)):
            if len(dataset_idx) != 1:
                raise ValueError(
                    f'Expected a scalar dataset index, got {dataset_idx}.')
            return self._normalize_cls_dataset_idx(dataset_idx[0])

        if dataset_idx == 4:
            dataset_idx = 3
        return dataset_idx

    def _get_cls_head_idx(self, dataset_idx):
        dataset_idx = self._normalize_cls_dataset_idx(dataset_idx)
        if dataset_idx == 0:
            return 0
        if dataset_idx == 1:
            return 1
        if dataset_idx == 2:
            return 2 if self.split_kitti_waymo_cls_head else 1
        if dataset_idx == 3:
            return 3 if self.split_kitti_waymo_cls_head else 2
        raise ValueError(f'Unsupported dataset index for cls head: {dataset_idx}')

    def _touch_module_params_zero(self, module, ref_tensor):
        zero = ref_tensor.new_zeros(())
        for param in module.parameters():
            if param.requires_grad and param.numel() > 0:
                zero = zero + param.reshape(-1)[0].to(
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype) * 0.0
        return ref_tensor + zero

    def _get_num_classes_for_dataset(self, dataset_idx):
        return self.multi_dataset_num_classes[self._get_cls_head_idx(dataset_idx)]

    def _get_cls_out_channels_for_dataset(self, dataset_idx):
        return self.multi_dataset_cls_out_channels[self._get_cls_head_idx(dataset_idx)]

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = nn.Sequential(
                Conv2d(self.in_channels, self.embed_dims, kernel_size=1),
                nn.ReLU(),
                Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
            )
            # self.input_proj = Conv2d(
            #     self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = nn.Sequential(
                Conv2d(self.in_channels, self.embed_dims, kernel_size=1),
                nn.ReLU(),
                Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
            )
            # self.input_proj = Conv2d(
            #     self.in_channels, self.embed_dims, kernel_size=1)

        if self.multi_dataset_cls_out_channels is not None:
            multi_dataset_fc_cls = []
            for cls_out_channel in self.multi_dataset_cls_out_channels:
                cls_branch = []
                for _ in range(self.num_reg_fcs):
                    cls_branch.append(Linear(self.embed_dims, self.embed_dims))
                    cls_branch.append(nn.LayerNorm(self.embed_dims))
                    cls_branch.append(nn.ReLU(inplace=True))
                if self.normedlinear:
                    cls_branch.append(NormedLinear(
                        self.embed_dims, cls_out_channel))
                else:
                    cls_branch.append(Linear(self.embed_dims, cls_out_channel))
                fc_cls = nn.Sequential(*cls_branch)
                multi_dataset_fc_cls.append(fc_cls)
        else:
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            if self.normedlinear:
                cls_branch.append(NormedLinear(
                    self.embed_dims, self.cls_out_channels))
            else:
                cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
            fc_cls = nn.Sequential(*cls_branch)

        if self.use_multi_reg_head:
            multi_dataset_reg_branch = []
            self.dataset_num = 4
            for _ in range(self.dataset_num):
                reg_branch = []
                for _ in range(self.num_reg_fcs):
                    reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                    reg_branch.append(nn.ReLU())
                reg_branch.append(Linear(self.embed_dims, self.code_size))
                reg_branch = nn.Sequential(*reg_branch)
                multi_dataset_reg_branch.append(reg_branch)

        else:
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

        if self.multi_dataset_cls_out_channels is not None:
            self.multi_dataset_cls_branches = nn.ModuleList(nn.ModuleList([fc_cls for _ in range(self.num_pred)]) for fc_cls in multi_dataset_fc_cls)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred)])
        if self.use_multi_reg_head:
            self.multi_dataset_reg_branches = nn.ModuleList(nn.ModuleList([reg_branch for _ in range(self.num_pred)]) for reg_branch in multi_dataset_reg_branch)
        else:
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred)])

        # self.position_encoder = nn.Sequential(
        #     nn.Linear(self.position_dim, self.embed_dims*4),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dims*4, self.embed_dims),
        # )
        
        if self.share_pe_encoder:
            position_encoder = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
            if self.with_position:
                self.position_encoder = position_encoder
            self.query_embedding = position_encoder
        else:
            if self.with_position:
                # self.position_dim = 3 * self.depth_num      # D*3 3:(x, y, z)
                self.position_encoder = nn.Sequential(
                    nn.Linear(self.embed_dims*3//2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        if self.use_downsample_depth is False and self.use_conv_downsample:
            # sample depth feature based on self.stride
            self.depth_downsample = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=self.stride, stride=self.stride)
        if self.use_dpt_feat:
            dpt_channels = 128
            self.dpt_pe_proj = Conv2d(dpt_channels, self.embed_dims, kernel_size=1)
            self.dpt_value_proj = Conv2d(dpt_channels, self.embed_dims, kernel_size=1)
            if self.use_cam_info:
                self.cam_encoder = nn.Sequential(
                    nn.Linear(16, self.embed_dims // 2),
                    nn.LayerNorm(self.embed_dims // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims // 2, self.embed_dims),
                    nn.LayerNorm(self.embed_dims)
                )
        if self.use_dpt_offset:
            dpt_channels = 128
            self.depth_offset_head = nn.Sequential(
                nn.Conv2d(dpt_channels, 64, 1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1)
            )
        if self.init_query is not None:
            ref_points = torch.from_numpy(np.load(self.init_query)) 
        else:
            ref_points = None
        self.reference_points = nn.Embedding(self.num_query, 3, _weight=ref_points)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        # self.query_embedding = nn.Sequential(
        #     nn.Linear(self.embed_dims*3//2, self.embed_dims),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dims, self.embed_dims),
        # )

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

        if self.with_pos_info:
            self.extra_position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )
        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims *
                          4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
            )
        if self.with_2dpe_only:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
            )
        
    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        self.transformer.init_weights()
        if self.multi_dataset_cls_out_channels is not None:
            for cls_branches in self.multi_dataset_cls_branches:
                if self.loss_cls.use_sigmoid:
                    bias_init = bias_init_with_prob(0.01)
                    for m in cls_branches:
                        nn.init.constant_(m[-1].bias, bias_init)
        else:
            if self.loss_cls.use_sigmoid:
                bias_init = bias_init_with_prob(0.01)
                for m in self.cls_branches:
                    nn.init.constant_(m[-1].bias, bias_init)
        
        if self.use_dpt_feat:
            if hasattr(self, 'cam_encoder'):
                for m in self.cam_encoder.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.weight, 1.0)
                        nn.init.constant_(m.bias, 0.0)
                        
    def prepare_for_dn(self, batch_size, reference_points, img_metas, data):
        has_dn_meta = self.training and self.with_dn and img_metas is not None
        if has_dn_meta:
            for img_meta in img_metas:
                if not isinstance(img_meta, dict):
                    has_dn_meta = False
                    break
                gt_bboxes_3d = img_meta.get('gt_bboxes_3d', None)
                gt_labels_3d = img_meta.get('gt_labels_3d', None)
                if gt_bboxes_3d is None or gt_labels_3d is None or \
                        not hasattr(gt_bboxes_3d, '_data') or \
                        not hasattr(gt_labels_3d, '_data'):
                    has_dn_meta = False
                    break

        if has_dn_meta:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [torch.ones_like(t, device=reference_points.device)
                     for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in targets]
        
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])
        
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            total_raydn_num = self.raydn_num * self.raydn_group
            known_indice = known_indice.repeat(self.scalar+total_raydn_num, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar+total_raydn_num, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()
            if self.multi_dataset_num_classes:
                num_classes = self._get_num_classes_for_dataset(data['dataset'][0])
            else:
                num_classes = self.num_classes

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.real_pc_range[0:3]) / (self.real_pc_range[3:6] - self.real_pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = num_classes
            
           # Ray Denoising
            for g_id in range(self.raydn_group):
                raydn_known_labels = labels.repeat(self.raydn_num, 1).view(-1).long().to(reference_points.device)
                raydn_known_bboxs = boxes.repeat(self.raydn_num, 1).to(reference_points.device)
                raydn_known_bbox_center = raydn_known_bboxs[:, :3].clone()
                raydn_known_bbox_scale = raydn_known_bboxs[:, 3:6].clone()
                noise_scale = raydn_known_bbox_scale[:, :].mean(dim=-1) / 2
                noise_step = (self.raydn_sampler.sample([noise_scale.shape[0]]).to(reference_points.device) * 2 - 1.0) * self.raydn_radius

                noise_scale = noise_scale.view(self.raydn_num, -1)
                noise_step = noise_step.view(self.raydn_num, -1)
                min_value, min_index = noise_step.abs().min(dim=0)
                reset_mask = min_value.abs() > self.split
                reset_value = (torch.rand(reset_mask.sum()).to(reference_points.device) * 2 - 1) * self.split     
                min_value[reset_mask] = reset_value           
                noise_step.scatter_(0, min_index.unsqueeze(0), min_value.unsqueeze(0))
                mask = torch.zeros_like(noise_step)
                mask.scatter_(0, min_index.unsqueeze(0), 1)
                mask = mask < 1
                mask = mask.view(-1)
                raydn_known_labels[mask] = num_classes

                raydn_known_bbox_center = raydn_known_bbox_center.view(self.raydn_num, -1, 3)
                ori_raydn_known_bbox_center = raydn_known_bbox_center.clone()
                for view_id in range(data['lidar2img'].shape[1]):
                    raydn_known_bbox_center_copy = torch.cat([ori_raydn_known_bbox_center.clone(), ori_raydn_known_bbox_center.new_ones((ori_raydn_known_bbox_center.shape[0], ori_raydn_known_bbox_center.shape[1], 1))], dim=-1)
                    tmp_p = raydn_known_bbox_center_copy.new_zeros(raydn_known_bbox_center_copy.shape)
                    for batch_id in range(data['lidar2img'].shape[0]):
                        tmp_p[:, sum(known_num[:batch_id]): sum(known_num[:batch_id+1])] = (data['lidar2img'][batch_id][view_id] @ raydn_known_bbox_center_copy[:, sum(known_num[:batch_id]): sum(known_num[:batch_id+1])].permute(0, 2, 1)).permute(0, 2, 1)

                    z_mask = tmp_p[..., 2] > 0 # depth > 0
                    tmp_p[..., :2] = tmp_p[..., :2] / (tmp_p[..., 2:3] + z_mask.unsqueeze(-1) * 1e-6 - (~z_mask).unsqueeze(-1) * 1e-6)
                    pad_h, pad_w = img_metas[0]['pad_shape'][0][:2] #(320, 800) #(640, 1600)
                    hw_mask = (
                        (tmp_p[..., 0] < pad_w)
                        & (tmp_p[..., 0] >= 0)
                        & (tmp_p[..., 1] < pad_h)
                        & (tmp_p[..., 1] >= 0)
                    ) # 0 < u < h and 0 < v < w
                    valid_mask = torch.logical_and(hw_mask, z_mask)
                    tmp_p[..., 2] += noise_scale*noise_step
                    tmp_p[..., :2] = tmp_p[..., :2] * tmp_p[..., 2:3]
                    proj_back = raydn_known_bbox_center_copy.new_zeros(raydn_known_bbox_center_copy.shape)
                    for batch_id in range(data['lidar2img'].shape[0]):
                        proj_back[:, sum(known_num[:batch_id]): sum(known_num[:batch_id+1])] = (data['lidar2img'][batch_id][view_id].inverse() @ tmp_p[:, sum(known_num[:batch_id]): sum(known_num[:batch_id+1])].permute(0, 2, 1)).permute(0, 2, 1)
                    raydn_known_bbox_center[valid_mask.unsqueeze(-1).repeat(1, 1, 3)] = proj_back[..., :3][valid_mask.unsqueeze(-1).repeat(1, 1, 3)]
                raydn_known_bbox_center = raydn_known_bbox_center.view(-1, 3)
                raydn_known_bbox_center[..., 0:3] = (raydn_known_bbox_center[..., 0:3] - self.real_pc_range[0:3]) / (self.real_pc_range[3:6] - self.real_pc_range[0:3])
                raydn_known_bbox_center = raydn_known_bbox_center.clamp(min=0.0, max=1.0)
                
                known_labels = torch.cat([known_labels, raydn_known_labels], dim=0)
                known_bbox_center = torch.cat([known_bbox_center, raydn_known_bbox_center], dim=0)
            known_bboxs = boxes.repeat(self.scalar+total_raydn_num, 1).to(reference_points.device)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * (self.scalar+total_raydn_num))
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar+total_raydn_num)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                # if i == self.scalar - 1:
                #     attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            for i in range(self.raydn_group):
                attn_mask[single_pad * (self.scalar + i*self.raydn_num):single_pad * (self.scalar + (i + 1)*self.raydn_num), single_pad * (self.scalar + (i + 1)*self.raydn_num):pad_size] = True
                attn_mask[single_pad * (self.scalar + i*self.raydn_num):single_pad * (self.scalar + (i + 1)*self.raydn_num), :single_pad * (self.scalar + i*self.raydn_num)] = True

            # update dn mask for temporal modeling
            query_size = pad_size + self.num_query + self.num_propagated
            tgt_size = pad_size + self.num_query + self.memory_len
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def forward(self, memory_center, img_metas, topk_indexes=None,  **data):
        # zero init the memory bank
        if self.multi_dataset_pc_range is not None:
            self.real_pc_range = torch.tensor(self.multi_dataset_pc_range[data['dataset'][0]]).to(data['img_feats'])
        else:
            self.real_pc_range = self.pc_range
        self.pre_update_memory(data)

        x = data['img_feats']
        B, N, C, H, W = x.shape
        x = x.flatten(0, 1)
        dpt_feats_flat = None
        if 'dpt_feats' in data and (self.use_dpt_feat or self.use_dpt_offset):
            dpt_feats_flat = data['dpt_feats'].flatten(0, 1)
        if self.use_dpt_feat:
            dpt_embed_val = self.dpt_value_proj(dpt_feats_flat)
            x = x + dpt_embed_val
        if self.with_vggt_depth:
            depth_map_pred = data['depth_vggt']
        else:
            depth_map_pred = data['depth_map']  # for gt depth test(Upper limit of 3dppe) 

        x = self.input_proj(x)
        x = x.view(B, N, C, H, W)

        if self.with_position:
            # 3D PE: (B, N_view, embed_dims, H, W)
            if self.use_detach:
                depth_map = depth_map_pred.detach()
            else:
                depth_map = depth_map_pred
            # depth_map = depth_map.view(B, N, H, W)

            if self.use_dpt_offset:
                depth_offset = self.depth_offset_head(dpt_feats_flat.detach())
                depth_offset = depth_offset.view(B, N, H, W)
                depth_map = depth_map + depth_offset
            masks = x.new_zeros((B, N, H, W))
            coords_position_embeding = self.position_embeding_3dppe(
                data, x, img_metas, depth_map)
            pe_mask = self.build_3dppe_mask(data, depth_map, B, N, H, W)
            if pe_mask is not None:
                coords_position_embeding = coords_position_embeding * pe_mask.unsqueeze(2)
            pos_embed = coords_position_embeding
            if self.use_dpt_feat:
                dpt_embed = self.dpt_pe_proj(dpt_feats_flat)
                if self.use_cam_info:
                    lidar2img = data['lidar2img'] 
                    cam_params = lidar2img.reshape(B * N, 16).float()
                    cam_embeds = self.cam_encoder(cam_params) 
                    cam_embeds = cam_embeds.view(B * N, self.embed_dims, 1, 1)
                    dpt_embed = dpt_embed + cam_embeds
                dpt_embed = dpt_embed.view(B, N, self.embed_dims, H, W)
                pos_embed = pos_embed + dpt_embed
            if self.with_multiview:
                # (B, N_view, num_feats*3=embed_dims*3/2, H, W)
                sin_embed = self.positional_encoding(masks)
                # (B, N_view, num_feats*3=embed_dims*3/2, H, W) --> (B*N_view, num_feats*3=embed_dims*3/2, H, W)
                # --> (B*N_view, embed_dims, H, W) --> (B, N_view, embed_dims, H, W)
                sin_embed = self.adapt_pos3d(
                    sin_embed.flatten(0, 1)).view(x.size())
                # (B, N_view, embed_dims, H, W)
                pos_embed = pos_embed + sin_embed
            elif self.with_2dpe_only:
                pos_embeds = []
                for i in range(N):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(
                    sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embed = pos_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(
                    pos_embed.flatten(0, 1)).view(x.size())
            elif self.with_2dpe_only:
                pos_embeds = []
                for i in range(N):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)
            else:
                pos_embed = x.new_zeros(x.size())

        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = topk_gather(memory, topk_indexes)

        pos_embed = pos_embed.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        pos_embed = topk_gather(pos_embed, topk_indexes)

        # first 256 tokens ->  memory   , last 644 query -> current
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            B, reference_points, img_metas, data)
        # dim = 128*3 384 -> query_embed 256
        query_pos = self.query_embedding(
            pos2posemb3d(inverse_sigmoid(reference_points)))
        tgt = torch.zeros_like(query_pos)

        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(
            query_pos, tgt, reference_points)
        outs_dec, _ = self.transformer(
            memory, tgt, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        reference = inverse_sigmoid(reference_points)
        if self.multi_dataset_cls_out_channels is not None:
            dataset_idx = self._normalize_cls_dataset_idx(data['dataset'][0])
        for lvl in range(outs_dec.shape[0]):
            assert reference.shape[-1] == 3
            if self.multi_dataset_cls_out_channels is not None:
                cls_idx = self._get_cls_head_idx(dataset_idx)
                outputs_class = self.multi_dataset_cls_branches[cls_idx][lvl](outs_dec[lvl])
                for i in range(len(self.multi_dataset_cls_branches)):
                    if i != cls_idx:
                        outputs_class = self._touch_module_params_zero(
                            self.multi_dataset_cls_branches[i][lvl],
                            outputs_class)
                        
            else:
                outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            if self.use_multi_reg_head:
                tmp = self.multi_dataset_reg_branches[dataset_idx][lvl](outs_dec[lvl])
                for i in range(self.dataset_num):
                    if i != dataset_idx:
                        tmp = self._touch_module_params_zero(
                            self.multi_dataset_reg_branches[i][lvl],
                            tmp)
            else:
                tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (
            all_bbox_preds[..., 0:3] * (self.real_pc_range[3:6] - self.real_pc_range[0:3]) + self.real_pc_range[0:3])

        # update the memory bank
        self.post_update_memory(
            data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:,
                                                :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:,
                                                :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes'] = (
                output_known_class, output_known_coord)
            outs = {
                'all_cls_scores': outputs_class,
                'all_bbox_preds': outputs_coord,
                'dn_mask_dict': mask_dict,

            }
        else:
            outs = {
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict': None,
            }

        return outs

    def build_3dppe_mask(self, data, depth_map, B, N, H, W):
        if not self.use_masked_3dppe:
            return None
        if 'depth_conf' not in data or data['depth_conf'] is None:
            return None

        depth_conf = data['depth_conf']
        if depth_conf.shape[-1] == 1:
            depth_conf = depth_conf.squeeze(-1)

        if depth_conf.dim() == 4 and depth_conf.shape[0] == B and depth_conf.shape[1] == N:
            depth_conf = depth_conf
        elif depth_conf.dim() == 3 and depth_conf.shape[0] == B * N:
            depth_conf = depth_conf.view(B, N, depth_conf.shape[-2], depth_conf.shape[-1])
        elif depth_conf.dim() == 3 and depth_conf.shape[0] == N and B == 1:
            depth_conf = depth_conf.unsqueeze(0)
        else:
            return None

        conf = depth_conf.flatten(0, 1).unsqueeze(1)
        if conf.shape[-2:] != (H, W):
            conf = F.interpolate(conf, size=(H, W), mode='bilinear', align_corners=False)
        conf = conf.view(B, N, H, W)

        valid_mask = (depth_map > 1e-5).to(conf.dtype)
        conf_flat = conf.view(B * N, -1)
        conf_threshold = torch.quantile(
            conf_flat, self.masked_3dppe_quantile, dim=1, keepdim=True)
        conf_threshold = conf_threshold.view(B, N, 1, 1)

        if self.masked_3dppe_mode == 'hard':
            pe_mask = (conf > conf_threshold).to(conf.dtype)
        else:
            pe_mask = torch.sigmoid(
                (conf - conf_threshold) * self.masked_3dppe_temperature)
            if self.masked_3dppe_min_scale > 0:
                pe_mask = self.masked_3dppe_min_scale + \
                    (1 - self.masked_3dppe_min_scale) * pe_mask

        return pe_mask * valid_mask

    def position_embeding_3dppe(self, data, img_feats, img_metas, depth_map=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats.shape
        
        # Map to the original scale to get the corresponding pixel coordinates.
        if self.use_downsample_depth:
            co_h, co_w = H, W
            coords_h = torch.arange(
                H, device=img_feats[0].device).float() * pad_h / H      # (H, )
            coords_w = torch.arange(
                W, device=img_feats[0].device).float() * pad_w / W      # (W, )
        else:
            co_h, co_w = pad_h, pad_w
            coords_h = torch.arange(
                pad_h, device=img_feats[0].device).float()
            coords_w = torch.arange(
                pad_w, device=img_feats[0].device).float()    # (W, )
        depth_map = depth_map.view(B, N, co_h, co_w)
        # (2, W, H)  --> (W, H, 2)    2: (u, v)
        coords = torch.stack(torch.meshgrid(
            [coords_w, coords_h])).permute(1, 2, 0).contiguous()
        coords = coords.view(1, 1, co_w, co_h, 2).expand(
            B, N, co_w, co_h, 2)       # (B, N_view, W, H, 2)

        depth_map = depth_map.permute(
            0, 1, 3, 2).contiguous()      # (B, N_view, W, H)

        depth_map = depth_map.unsqueeze(dim=-1)     # (B, N_view, W, H, 1)
        # (B, N_view, W, H, 2)    (du, dv)
        coords = coords * \
            torch.maximum(depth_map, torch.ones_like(depth_map) * eps)
        # (B, N_view, W, H, 3)   (du, dv, d)
        coords = torch.cat([coords, depth_map], dim=-1)
        # (B, N_view, W, H, 4)   (du, dv, d, 1)
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)

        lidar2imgs = data['lidar2img']
        img2lidars = lidar2imgs.inverse()

        coords = coords.unsqueeze(dim=-1)       # (B, N_view, W, H, 4, 1)
        # (B, N_view, 1, 1, 4, 4) --> (B, N_view, W, H, 4, 4)
        img2lidars = img2lidars.view(B, N, 1, 1, 4, 4).expand(
            B, N, co_w, co_h, 4, 4)

        # The frustum points corresponding to each pixel in the image are projected into the lidar system with img2lidars..
        # (B, N_view, W, H, D, 4, 4) @ (B, N_view, W, H, D, 4, 1) --> (B, N_view, W, H, D, 4, 1)
        # --> (B, N_view, W, H, D, 3)   3: (x, y, z)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        # With the help of position range, the 3D coordinates are normalized.
        if self.multi_dataset_position_range:
            position_range = torch.tensor(self.multi_dataset_position_range[data['dataset'][0]]).to(coords3d)
        else:
            position_range = self.position_range
        # limit the range of the coordinates
        x_clamped = torch.clamp(coords3d[..., 0:1], min=position_range[0], max=position_range[3])
        y_clamped = torch.clamp(coords3d[..., 1:2], min=position_range[1], max=position_range[4])
        z_clamped = torch.clamp(coords3d[..., 2:3], min=position_range[2], max=position_range[5])
        coords3d = torch.cat([x_clamped, y_clamped, z_clamped], dim=-1)
        # normalize the coordinates
        coords3d[..., 0:3] = (coords3d[..., 0:3] - position_range[0:3]) / (
            position_range[3:6] - position_range[0:3])  # norm 0~1

        # (B, N_view, W, H, D, 3) --> (B, N_view, D, 3, H, W) --> (B*N_view, D*3, H, W)
        coords3d = coords3d.permute(0, 1, 3, 2, 4).contiguous().view(
            B*N, co_h, co_w, 3)      # (B*N_view, H, W, 3)
        eps = 1e-5
        coords3d = coords3d.clamp(min=eps, max=1-eps)
        coords3d = inverse_sigmoid(coords3d)    # (B*N_view, H, W, 3)
        # 3D position embedding(PE)
        coords_position_embeding = self.position_encoder(
            pos2posemb3d(coords3d))  # (B*N_view, H, W, embed_dims)
        coords_position_embeding = coords_position_embeding.permute(
            0, 3, 1, 2).contiguous()    # (B*N_view, embed_dims, H, W)
        if not self.use_downsample_depth:
            if self.use_conv_downsample:
                coords_position_embeding = self.depth_downsample(coords_position_embeding)
            else:
                coords_position_embeding = F.interpolate(coords_position_embeding, scale_factor=1/self.stride, mode='bilinear')

        return coords_position_embeding.view(B, N, self.embed_dims, H, W)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             dataset=None,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
       
        all_dataset_list = [dataset for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_dataset_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(
                preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [
                known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]

            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list,
                all_num_tgts_list, all_dataset_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                               dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, all_dataset_list,
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                               dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()
                num_dec_layer += 1

        return loss_dict

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    dataset_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list, dataset_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None,
                           dataset=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        if gt_bboxes.shape[1] == 7:
            bbox_pred = bbox_pred[:, :8]
            match_cost = self.match_costs[:8]
        else:
            match_cost = self.match_costs
        if dataset is not None:
            assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, match_cost, self.match_with_velo, dataset, self.normalize, self.normalize_far)
        else:
            assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, match_cost, self.match_with_velo, self.normalize, self.normalize_far)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if self.multi_dataset_num_classes:
            num_classes = self._get_num_classes_for_dataset(dataset)
        else:
            num_classes = self.num_classes
        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    dataset=None,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # if self.multi_dataset_pc_range is not None:
        #     real_pc_range = self.multi_dataset_pc_range[dataset_list]
        #     real_pc_min = real_pc_range[:3]  # [min_x, min_y, min_z]
        #     real_pc_max = real_pc_range[3:]  # [max_x, max_y, max_z]

        #     x_in_range = (bbox_preds[..., 0] >= real_pc_min[0]) & (bbox_preds[..., 0] <= real_pc_max[0])
        #     y_in_range = (bbox_preds[..., 1] >= real_pc_min[1]) & (bbox_preds[..., 1] <= real_pc_max[1])
        #     z_in_range = (bbox_preds[..., 2] >= real_pc_min[2]) & (bbox_preds[..., 2] <= real_pc_max[2])
        #     total_mask_in_range = x_in_range & y_in_range & z_in_range
        #     bbox_preds = bbox_preds[total_mask_in_range].unsqueeze(0)
        #     cls_scores = cls_scores[total_mask_in_range].unsqueeze(0)

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        dataset_list = [dataset for _ in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list, dataset_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        if self.multi_dataset_cls_out_channels is not None:
            cls_out_channels = self._get_cls_out_channels_for_dataset(dataset)
        else:
            cls_out_channels = self.cls_out_channels
        cls_scores = cls_scores.reshape(-1, cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.real_pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        if self.normalize:
            bbox_preds = normalize_bbox_center(bbox_preds, self.real_pc_range)
            normalized_bbox_targets = normalize_bbox_center(normalized_bbox_targets, self.real_pc_range)
        
        if self.normalize_far:
            bbox_preds = normalize_far_bbox_center(bbox_preds, self.real_pc_range)
            normalized_bbox_targets = normalize_far_bbox_center(normalized_bbox_targets, self.real_pc_range)

        if normalized_bbox_targets.shape[1] == 8:
            bbox_weights = bbox_weights * self.code_weights[:8]
            bbox_preds = bbox_preds[isnotnan, :8]
        else:
            bbox_weights = bbox_weights * self.code_weights
            bbox_preds = bbox_preds[isnotnan, :10]

        loss_bbox = self.loss_bbox(
                bbox_preds, normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

   
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None,
                    dataset=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # if self.multi_dataset_pc_range is not None:
        #     real_pc_range = self.multi_dataset_pc_range[dataset_list]
        #     real_pc_min = real_pc_range[:3]  # [min_x, min_y, min_z]
        #     real_pc_max = real_pc_range[3:]  # [max_x, max_y, max_z]

        #     x_in_range = (bbox_preds[..., 0] >= real_pc_min[0]) & (bbox_preds[..., 0] <= real_pc_max[0])
        #     y_in_range = (bbox_preds[..., 1] >= real_pc_min[1]) & (bbox_preds[..., 1] <= real_pc_max[1])
        #     z_in_range = (bbox_preds[..., 2] >= real_pc_min[2]) & (bbox_preds[..., 2] <= real_pc_max[2])
        #     total_mask_in_range = x_in_range & y_in_range & z_in_range
        #     bbox_preds = bbox_preds[total_mask_in_range]
        #     cls_scores = cls_scores[total_mask_in_range]

        # classification loss
        if self.multi_dataset_cls_out_channels is not None:
            cls_out_channels = self._get_cls_out_channels_for_dataset(dataset)
        else:
            cls_out_channels = self.cls_out_channels
        cls_scores = cls_scores.reshape(-1, cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split  * self.split ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.real_pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        if self.normalize:
            bbox_preds = normalize_bbox_center(bbox_preds, self.real_pc_range)
            normalized_bbox_targets = normalize_bbox_center(normalized_bbox_targets, self.real_pc_range)

        if self.normalize_far:
            bbox_preds = normalize_far_bbox_center(bbox_preds, self.real_pc_range)
            normalized_bbox_targets = normalize_far_bbox_center(normalized_bbox_targets, self.real_pc_range)

        if normalized_bbox_targets.shape[1] == 8:
            bbox_weights = bbox_weights[:, :8] * self.code_weights[:8]
            bbox_preds = bbox_preds[isnotnan, :8]
        else:
            bbox_weights = bbox_weights * self.code_weights
            bbox_preds = bbox_preds[isnotnan, :10]
        
        loss_bbox = self.loss_bbox(
                bbox_preds, normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point -
                                self.real_pc_range[:3]) / (self.real_pc_range[3:6] - self.real_pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(
            inverse_sigmoid(temp_reference_point)))
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(
            0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)

        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(
                reference_points[..., :3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat(
                [self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(
            torch.zeros_like(reference_points[..., :1])))
        temp_pos += self.time_embedding(
            pos2posemb1d(self.memory_timestamp).float())

        # TODO:
        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat(
                [query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat(
                [reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(
                0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, dataset_idx=None, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        if self.multi_dataset_num_classes:
            num_classes = self._get_num_classes_for_dataset(dataset_idx)
            preds_dicts = self.bbox_coder.decode(preds_dicts, dataset_idx, num_classes)
        else:
            preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
