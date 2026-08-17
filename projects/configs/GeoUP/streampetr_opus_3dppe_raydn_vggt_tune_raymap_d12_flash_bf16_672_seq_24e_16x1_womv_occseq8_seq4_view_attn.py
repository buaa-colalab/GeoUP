_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
depth_range = 90
voxel_size = [0.2, 0.2, 8]

_dim_ = 256
_num_points_ = 2
_num_groups_ = 4
_num_layers_ = 5
_occ_num_frames_ = 8
_num_queries_ = 4800
point_cloud_range_occ = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size_occ = [0.4, 0.4, 0.4]
occ_size = [200, 200, 16]
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

num_gpus = 16
batch_size = 1
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 24

queue_length = 1
num_frame_losses = 1
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                'img_timestamp', 'ego_pose', 'ego_pose_inv']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Petr3D',
    occ_seq_length=_occ_num_frames_,
    position_level=1,
    stride=14,
    depth_range=depth_range,
    img_backbone=dict(
        type='AggregatorVGGT',
        depth=12,
        frozen=False,
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpts/GeoUP/stage-1/low_resolution.pth',
            prefix='img_backbone'
        )
    ),
    img_neck=dict(
        type='VGGTDetectionNeck',
        backbone_channel=1024 * 2,
        in_indices=[2, 5, 8, 11],
        in_channels=[256, 512, 1024, 1024 * 2],
        patch_size=14,
        use_pan=True,
        pan_target_level=1,
        out_channels=256,
        num_outs=4),
    img_roi_head=dict(
        type='FocalHead',
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        stride=14,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss',
                             reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type='HungarianAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0,
                              box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))
    ),
    camera_head=dict(
        type='CameraHead',
        dim_in=2048,
        loss_type='l1',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpts/GeoUP/stage-1/low_resolution.pth',
            prefix='camera_head'
        )
    ),
    depth_head=dict(
        type='DPTHeadPseudo',
        dim_in=2048,
        intermediate_layer_idx=[2, 5, 8, 11],
        output_dim=2,
        activation='exp',
        conf_activation='expp1',
        gradient_loss_fn='grad',
        use_intrinsics=True,
        use_full_loss=True,
        valid_range=0.98,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpts/GeoUP/stage-1/low_resolution.pth',
            prefix='depth_head'
        )
    ),
    pts_bbox_head=dict(
        type='StreamPETRHeadVGGT',
        sync_cls_avg_factor=True,
        num_classes=10,
        stride=14,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        match_with_velo=True,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        with_vggt_depth=True,
        with_multiview=False,
        transformer=dict(
            type='PETRTemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),),
    occ_neck=dict(
        type='VGGTOccNeck',
        backbone_channel=2048,
        in_indices=[2, 5, 8, 11],
        in_channels=[256, 512, 1024, 2048],
        patch_size=14,
        out_channels=_dim_,
        num_outs=4),
    occ_head=dict(
        type='OPUSV2Head',
        num_classes=len(occ_class_names),
        in_channels=_dim_,
        num_query=_num_queries_,
        pc_range=point_cloud_range_occ,
        voxel_size=voxel_size_occ,
        transformer=dict(
            type='OPUSV2Transformer',
            embed_dims=_dim_,
            num_layers=_num_layers_,
            num_frames=_occ_num_frames_,
            num_points=_num_points_,
            num_groups=_num_groups_,
            num_refines=[1, 2, 4, 8, 16],
            num_pt_channels=32,
            scales=[0.5],
            pc_range=point_cloud_range_occ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5)
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range),
        ),
        occ=dict(
            cls_weights=[
                10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
        )
    ),
    test_cfg=dict(
        occ=dict(score_thr=0.25),
        pts=dict()
    )
)


dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

file_client_args = dict(backend='disk')


ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (224, 672),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewDepthFromFiles',
        to_float32=True,
        max_dist=depth_range,
        use_all_depth=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_bbox=True, with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage',
         data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[1, 1],
         reverse_angle=True,
         training=True,
         ),
    dict(type='LoadOccAnnotations',
         occ_path=data_root + '/occ_gts',
    ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=112),
    dict(type='PETRFormatBundle3D', class_names=class_names,
         collect_keys=collect_keys + ['prev_exists', 'cam_extrinsics_global']),
    dict(
        type='Collect3D',
        keys=['gt_depth', 'cam_extrinsics_global', 'point_mask', 'gt_bboxes_3d',
              'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d',
              'depths', 'prev_exists', 'voxel_semantics', 'mask_camera'] + collect_keys,
        meta_keys=('lidar2img', 'ego2img', 'ego2global', 'ego2occ', 'filename',
                   'ida_mat', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'scene_token', 'gt_bboxes_3d',
                   'gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewDepthFromFiles',
        to_float32=True,
        max_dist=depth_range,
        use_all_depth=True),
    dict(type='ResizeCropFlipRotImage',
         data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=112),
    dict(type='LoadOccAnnotations',
         occ_path=data_root + '/occ_gts',
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys + ['cam_extrinsics_global'],
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img', 'gt_depth', 'cam_extrinsics_global', 'point_mask',
                      'voxel_semantics', 'mask_camera'] + collect_keys,
                meta_keys=('lidar2img', 'ego2img', 'ego2global', 'ego2occ',
                           'filename', 'ida_mat', 'ori_shape', 'img_shape',
                           'pad_shape', 'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_temporal_infos_train.pkl',
        num_frame_losses=num_frame_losses,
        seq_split_num=2,
        seq_mode=True,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas',
                                     'gt_depth', 'point_mask',
                                     'cam_extrinsics_global'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas',
                                     'cam_extrinsics_global'],
        queue_length=queue_length,
        ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas',
                                     'cam_extrinsics_global'],
        queue_length=queue_length,
        ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)


optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01),
            'img_neck.projects': dict(lr_mult=0.01),
            'img_neck.resize_layers': dict(lr_mult=0.01),
            'sampling_offset': dict(lr_mult=0.1),
            'depth_head': dict(lr_mult=0.1),
            'camera_head': dict(lr_mult=0.1),
            'occ_neck.projects': dict(lr_mult=0.01),
            'occ_neck.resize_layers': dict(lr_mult=0.01),
            'occ_head': dict(lr_mult=1.0),
        }),
    weight_decay=0.01)

optimizer_config = dict(type='BF16OptimizerHook',
                        grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

evaluation = dict(
    interval=num_iters_per_epoch * num_epochs,
    pipeline=test_pipeline,
    metric=['bbox', 'depth', 'occ', 'camera'],
    depth_eval=dict(
        max_depth=500,
    ),
    occ_eval=dict(
        class_names=occ_class_names + ['free'],
        occ_gt_root=data_root + 'occ_gts'
    )
)
find_unused_parameters = False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
resume_from = None

dist_params = dict(
    backend='nccl',
    timeout=5400
)
