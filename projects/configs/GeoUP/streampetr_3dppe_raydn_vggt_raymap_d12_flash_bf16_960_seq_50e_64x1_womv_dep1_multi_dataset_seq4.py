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
use_all_depth = True

_dim_ = 256
_num_points_ = 2
_num_groups_ = 4
_num_layers_ = 5
_num_frames_ = 8
_num_queries_ = 4800
point_cloud_range_occ = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size_occ = [0.4, 0.4, 0.4] # Consistent with occ_size [200, 200, 16]
occ_size = [200, 200, 16]
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

# 'nuscenes': 0,
# 'kitti': 1,
# 'waymo': 2,
# 'av2': 3,
multi_dataset_sample_ratios = [8, 1, 9, 8, 3]
# multi_dataset_sample_ratios = [1]
multi_dataset_pc_range = [
    [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    [-40, 0, -3, 40, 70.4, 1],
    [-74.88, -74.88, -2, 74.88, 74.88, 4],
    [-152.4, -152.4, -5.0, 152.4, 152.4, 5.0],
]
multi_dataset_position_range = [
    [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    [-50, 0, -6, 50, 80.4, 4],
    [-85, -85, -4, 85, 85, 8],
    [-152.4, -152.4, -5.0, 152.4, 152.4, 5.0],
]
multi_dataset_depth_range = [
    61.2, 80.4, 85, 64
]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
multi_dataset_class_names = [
    [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    ['Car', 'Pedestrian', 'Cyclist'],
    ['Car', 'Pedestrian', 'Cyclist'],
    ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
    'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
    'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
    'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
    'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
    'WHEELCHAIR', 'WHEELED_DEVICE','WHEELED_RIDER']
]

nuscenes_dataset_len = 28130
kitti_dataset_len = 3712
waymo_dataset_len = 31155
av2_dataset_len = 27473
ddad_dataset_len = 12650
num_gpus = 64
batch_size = 1
num_iters_per_epoch = (nuscenes_dataset_len + kitti_dataset_len + waymo_dataset_len + av2_dataset_len + ddad_dataset_len) // (num_gpus * batch_size)
num_epochs = 50
max_view = 7
val_loss_interval_iters = num_iters_per_epoch * 2
queue_length = 1
num_frame_losses = 1
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                'img_timestamp', 'ego_pose', 'ego_pose_inv', 'dataset', 'cam_extrinsics_global']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Petr3D',
    occ_seq_length=8,
    position_level=1,
    stride=14,
    depth_range=depth_range,
    img_backbone=dict(
        type='AggregatorVGGT',
        patch_size=14,
        depth=12,
        frozen=False,
        with_cp=True,
        seq_info=dict(
            batch_size=batch_size * num_gpus,
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpts/GeoUP/stage-1/high_resolution.pth',
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
        split_kitti_waymo_cls_head=True,
        sync_cls_avg_factor=True,
        num_classes=10,
        multi_dataset_num_classes=[10, 3, 26],
        in_channels=256,
        stride=14,
        use_intrinsics=True,
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
    depth_head=dict(
        type='DPTHeadPseudo',
        dim_in=2048,
        intermediate_layer_idx=[2, 5, 8, 11],
        output_dim=2,
        activation="exp",
        conf_activation="expp1",
        gradient_loss_fn="grad",
        use_intrinsics=True,
        use_full_loss=True,
        valid_range=0.98,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpts/GeoUP/stage-1/high_resolution.pth',
            prefix='depth_head'
        )
    ),
    camera_head=dict(
        type='CameraHead',
        dim_in=2048,
        loss_type="l1",
        camera_weight=5.0,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpts/GeoUP/stage-1/high_resolution.pth',
            prefix='camera_head'
        )
    ),
    pts_bbox_head=dict(
        type='StreamPETRHeadVGGT',
        split_kitti_waymo_cls_head=True,
        sync_cls_avg_factor=True,
        normalize_far=True,
        multi_dataset_pc_range=multi_dataset_pc_range,
        multi_dataset_position_range=multi_dataset_position_range,
        multi_dataset_num_classes=[10, 3, 26],
        num_classes=10,
        stride=14,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        match_with_velo=True,
        scalar=10,  # noise groups
        noise_scale=1.0,
        dn_weight=1.0,  # dn loss weight
        split=0.75,  # positive rate
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
                    with_cp=True,  # use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoderMultiDataset',
            multi_dataset_post_center_range=multi_dataset_position_range,
            multi_dataset_pc_range=multi_dataset_pc_range,
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
        in_indices=[2, 5, 8, 11], # Adapted for depth 12 backbone
        in_channels=[256, 512, 1024, 2048], # Assuming constant width 2048 for AggregatorVGGT
        patch_size=14,
        out_channels=_dim_,
        num_outs=4),
    occ_head=dict(
        type='OPUSV2Head',
        num_classes=len(occ_class_names),
        in_channels=_dim_,
        # with_cp=True,
        num_query=_num_queries_,
        pc_range=point_cloud_range_occ,
        voxel_size=voxel_size_occ,
        # pfn_channels=[128, 256], # Increased capacity for better performance
        transformer=dict(
            type='OPUSV2Transformer',
            embed_dims=_dim_,
            num_layers=_num_layers_,
            num_frames=_num_frames_,
            num_points=_num_points_,
            num_groups=_num_groups_,
            num_refines=[1, 2, 4, 8, 16], # Adapted for 5 layers
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
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3DMultiDataset',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', weight=0.0),
            multi_dataset_pc_range=multi_dataset_pc_range),),
        occ=dict(
            cls_weights=[
            10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
        )),
    test_cfg=dict(
        occ=dict(score_thr=0.25),
        pts=dict()
    ))


file_client_args = dict(backend='disk')


nus_ida_aug_conf = {
    # PETR / StreamPETR / BEVFusion commonly use 256x704 or 320x800 on nuScenes.
    "resize_lim": (0.45, 0.65),
    "final_dim": (308, 798),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
kitti_ida_aug_conf = {
    # Keep the repo's KITTI max image size convention (1242x375), rounded down
    # to the largest size divisible by 14.
    "resize_lim": (1.0, 1.0),
    "final_dim": (364, 1232),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 375,
    "W": 1242,
    "rand_flip": True,
}

av2_ida_aug_conf = {
    # Follow Far3D's large-image AV2 setting (1536x1536), rounded down to the
    # nearest size divisible by 14 for the ViT patch embed.
    "resize_lim": (0.45, 0.55),
    "final_dim": (630, 952),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 1550,
    "W": 2048,
    "rand_flip": False,
}

waymo_ida_aug_conf = {
    # MV-FCOS3D++ uses 1248x832 / 1536x1024 crops with about 1.5:1 aspect ratio.
    "resize_lim": (0.55, 0.66),
    "final_dim": (630, 952),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "rand_flip": True,
}

ddad_ida_aug_conf = {
    # No common public DDAD detection ida config was found, so keep its native ~1.59:1
    # camera ratio and match the overall token budget used above.
    "resize_lim": (0.5, 0.7),
    "final_dim": (630, 952),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 1216,
    "W": 1936,
    "rand_flip": True,
}

dataset_type = 'ConcatDataset'
nuscenes_dataset_type = 'CustomNuScenesDataset'
nuscenes_data_root = './data/nuscenes/'
waymo_dataset_type = 'CustomWayMoDataset'
waymo_data_root = './data/waymo/kitti_format/'
kitti_dataset_type = 'CustomKittiDataset'
kitti_data_root = './data/kitti/'
av2_dataset_type = 'Argoverse2DatasetT'
av2_data_root = './data/argoverse/'
ddad_dataset_type = 'CustomDDADDataset'
ddad_data_root = './data/DDAD/'

nus_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewDepthFromFiles', to_float32=True, max_dist=depth_range, use_all_depth=use_all_depth),
    # dict(type='LoadPointsFromFile',
    #      coord_type='LIDAR',
    #      load_dim=5,
    #      use_dim=5,
    #      file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
         with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=multi_dataset_pc_range[0]),
    dict(type='ObjectNameFilter', classes=multi_dataset_class_names[0]),
    dict(type='ResizeCropFlipRotImage',
         data_aug_conf=nus_ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[1, 1],
         reverse_angle=True,
         training=True,
         ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(type='LoadOccAnnotations',
         occ_path=nuscenes_data_root + '/occ_gts',
    ),
    dict(type='PETRFormatBundle3D', class_names=multi_dataset_class_names[0],
         collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_depth', 'point_mask', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists', 'voxel_semantics', 'mask_camera'] + collect_keys,
         meta_keys=('lidar2img', 'ego2img', 'ego2global', 'ego2occ', 'filename', 'ida_mat', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
]

av2_train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesV1', to_float32=True),
    dict(type='LoadMultiViewDepthFromNpyFiles', to_float32=True, max_dist=depth_range, use_all_depth=use_all_depth),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
         with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=multi_dataset_pc_range[3]),
    dict(type='ObjectNameFilter', classes=multi_dataset_class_names[3]),
    dict(type='AV2ResizeCropFlipRotImageV2',
         data_aug_conf=av2_ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True,
         ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(type='PETRFormatBundle3D', class_names=multi_dataset_class_names[3],
         collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_depth', 'point_mask', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
         meta_keys=('ida_mat', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
]

kitti_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewDepthFromNpyFiles', to_float32=True, max_dist=depth_range, use_all_depth=use_all_depth),
    # dict(type='LoadPointsFromFile',
    #      coord_type='LIDAR',
    #      load_dim=5,
    #      use_dim=5,
    #      file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
         with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=multi_dataset_pc_range[1]),
    dict(type='ObjectNameFilter', classes=multi_dataset_class_names[1]),
    dict(type='ResizeCropFlipRotImage',
         data_aug_conf=kitti_ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True,
         ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(type='PETRFormatBundle3D', class_names=multi_dataset_class_names[1],
         collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_depth', 'point_mask', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
         meta_keys=('ida_mat', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
]
waymo_train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesV1', to_float32=True),
    dict(type='LoadMultiViewDepthFromNpyFiles', to_float32=True, max_dist=depth_range, use_all_depth=use_all_depth),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
         with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=multi_dataset_pc_range[2]),
    dict(type='ObjectNameFilter', classes=multi_dataset_class_names[2]),
    dict(type='WaymoResizeCropFlipRotImage',
         data_aug_conf=waymo_ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True,
         ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(type='PETRFormatBundle3D', class_names=multi_dataset_class_names[2],
         collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_depth', 'point_mask', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
         meta_keys=('ida_mat', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
]
nus_test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage',
         data_aug_conf=nus_ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(type='LoadOccAnnotations',
         occ_path=nuscenes_data_root + '/occ_gts',
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=multi_dataset_class_names[0],
                with_label=False),
            dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera'] + collect_keys,
                 meta_keys=('lidar2img', 'ego2img', 'ego2global', 'ego2occ', 'filename', 'ida_mat', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

ddad_train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewDepthFromNpyFiles', to_float32=True, max_dist=depth_range, use_all_depth=use_all_depth),
    # dict(type='LoadPointsFromFile',
    #      coord_type='LIDAR',
    #      load_dim=5,
    #      use_dim=5,
    #      file_client_args=file_client_args),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
    #      with_label=True, with_bbox_depth=True),
    # dict(type='ObjectRangeFilter', point_cloud_range=multi_dataset_pc_range[1]),
    # dict(type='ObjectNameFilter', classes=multi_dataset_class_names[1]),
    dict(type='ResizeCropFlipRotImage',
         data_aug_conf=ddad_ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True,
         ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(type='PETRFormatBundle3D', class_names=multi_dataset_class_names[1],
         collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_depth', 'point_mask', 'img', 'prev_exists'] + collect_keys,
         meta_keys=('ida_mat', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
]

kitti_test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage',
         data_aug_conf=kitti_ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
                 meta_keys=('ida_mat', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

waymo_test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesV1', to_float32=True),
    dict(type='WaymoResizeCropFlipRotImage',
         data_aug_conf=waymo_ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
                 meta_keys=('ida_mat', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

av2_test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesV1', to_float32=True),
    dict(type='AV2ResizeCropFlipRotImageV2',
         data_aug_conf=av2_ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
                 meta_keys=('ida_mat', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

nus_val_loss_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewDepthFromFiles',
        to_float32=True,
        max_dist=depth_range,
        use_all_depth=use_all_depth),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=multi_dataset_pc_range[0]),
    dict(type='ObjectNameFilter', classes=multi_dataset_class_names[0]),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=nus_ida_aug_conf,
        training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14, use_all_depth=use_all_depth),
    dict(type='LoadOccAnnotations',
         occ_path=nuscenes_data_root + '/occ_gts',
    ),
    dict(
        type='PETRFormatBundle3D',
        class_names=multi_dataset_class_names[0],
        collect_keys=collect_keys + ['prev_exists']),
    dict(
        type='Collect3D',
        keys=[
            'gt_depth', 'point_mask', 'gt_bboxes_3d', 'gt_labels_3d',
            'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths',
            'prev_exists', 'voxel_semantics', 'mask_camera'
        ] + collect_keys,
        meta_keys=('lidar2img', 'ego2img', 'ego2global', 'ego2occ', 'filename', 'ida_mat', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
]

nuscenes_dataset_val=dict(type=nuscenes_dataset_type, pipeline=nus_test_pipeline, collect_keys=collect_keys +
             ['img', 'img_metas'], queue_length=queue_length, ann_file=nuscenes_data_root + 'nuscenes2d_temporal_infos_val.pkl', classes=multi_dataset_class_names[0], modality=input_modality)

kitti_dataset_val=dict(type=kitti_dataset_type, pipeline=kitti_test_pipeline, split='val', seq_length=30, seq_mode=True, data_root=kitti_data_root, collect_keys=collect_keys +
             ['img', 'img_metas'], queue_length=queue_length, ann_file=kitti_data_root + 'kitti_infos_val.pkl', classes=multi_dataset_class_names[1], modality=input_modality)

waymo_dataset_val=dict(type=waymo_dataset_type, pipeline=waymo_test_pipeline, split='val', data_root=waymo_data_root, test_seq_interval=5, collect_keys=collect_keys +
             ['img', 'img_metas'], queue_length=queue_length, ann_file=waymo_data_root + 'kitti_infos_val.pkl', classes=multi_dataset_class_names[2], modality=input_modality)


av2_dataset_val=dict(
        type=av2_dataset_type,
        pipeline=av2_test_pipeline,
        data_root=av2_data_root,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        ann_file=av2_data_root + 'av2_val_infos_mini_new.pkl',
        split='val',
        load_interval=1,
        classes=multi_dataset_class_names[3],
        modality=input_modality,
        test_mode=True,
        interval_test=True)

nuscenes_dataset = dict(
    type=nuscenes_dataset_type,
    data_root=nuscenes_data_root,
    ann_file=nuscenes_data_root + 'nuscenes2d_temporal_infos_train.pkl',
    num_frame_losses=num_frame_losses,
    seq_split_num=2,
    seq_mode=True,
    pipeline=nus_train_pipeline,
    classes=multi_dataset_class_names[0],
    modality=input_modality,
    collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas', 'gt_depth', 'point_mask'],
    queue_length=queue_length,
    test_mode=False,
    use_valid_flag=True,
    box_type_3d='LiDAR')

waymo_dataset = dict(
    type=waymo_dataset_type,
    data_root=waymo_data_root,
    ann_file=waymo_data_root + 'kitti_infos_train.pkl',
    split='train',
    num_frame_losses=num_frame_losses,
    load_interval=5,
    seq_split_num=2,
    seq_mode=True,
    pipeline=waymo_train_pipeline,
    classes=multi_dataset_class_names[2],
    modality=input_modality,
    collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas', 'gt_depth', 'point_mask'],
    queue_length=queue_length,
    test_mode=False,
    # use_valid_flag=True,
    box_type_3d='LiDAR')

kitti_dataset = dict(
    type=kitti_dataset_type,
    data_root=kitti_data_root,
    ann_file=kitti_data_root + 'kitti_infos_train.pkl',
    split='train',
    num_frame_losses=num_frame_losses,
    seq_length=30,
    seq_mode=True,
    pipeline=kitti_train_pipeline,
    classes=multi_dataset_class_names[1],
    modality=input_modality,
    collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas', 'gt_depth', 'point_mask'],
    queue_length=queue_length,
    test_mode=False,
    # use_valid_flag=True,
    box_type_3d='LiDAR')

av2_dataset = dict(
        type=av2_dataset_type,
        data_root=av2_data_root,
        ann_file=av2_data_root + 'av2_train_infos_mini_new.pkl',
        split='train',
        load_interval=1,
        num_frame_losses=num_frame_losses,
        seq_split_num=2,
        seq_mode=True,
        pipeline=av2_train_pipeline,
        classes=multi_dataset_class_names[3],
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas', 'gt_depth', 'point_mask'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=False,
        interval_test=True,
        box_type_3d='LiDAR')


ddad_dataset = dict(
        type=ddad_dataset_type,
        data_root=ddad_data_root,
        ann_file=ddad_data_root + 'ddad_infos_train.pkl',
        # split='train',
        # num_frame_losses=num_frame_losses,
        # seq_split_num=2,
        seq_mode=True,
        pipeline=ddad_train_pipeline,
        classes=multi_dataset_class_names[3],
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas', 'gt_depth', 'point_mask'],
        queue_length=queue_length,
        test_mode=False,
        # use_valid_flag=False,
        # interval_test=True,
        box_type_3d='LiDAR')

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        datasets=[nuscenes_dataset, kitti_dataset, waymo_dataset, av2_dataset, ddad_dataset],
        # datasets=[av2_dataset],
        separate_eval=True,),
    val_loss=dict(
        type=nuscenes_dataset_type,
        data_root=nuscenes_data_root,
        ann_file=nuscenes_data_root + 'nuscenes2d_temporal_infos_val.pkl',
        num_frame_losses=num_frame_losses,
        seq_split_num=2,
        seq_mode=True,
        pipeline=nus_val_loss_pipeline,
        classes=multi_dataset_class_names[0],
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas', 'gt_depth', 'point_mask'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR',
        samples_per_gpu=1),
    val=nuscenes_dataset_val,
    test=nuscenes_dataset_val,
    shuffler_sampler=dict(type='MultiDatasetSeqSampler', dataset_ratios=multi_dataset_sample_ratios),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

val_loss = dict(
    interval=val_loss_interval_iters,
    by_epoch=False,
    prefix='val',
)

optimizer = dict(
    # _delete_=True,
    type='Muon',
    lr=6e-4,
    weight_decay=0.01,
    momentum=0.95,
    nesterov=True,
    ns_steps=3,
    min_muon_numel=65536,
    ns_eps=1e-7,
    muon_lr_scale=0.2,
    min_muon_ndim=2,
    adamw_betas=(0.9, 0.95),
    adamw_eps=1e-8,
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
)

optimizer_config = dict(type='BF16OptimizerHook',
                        loss_scale='dynamic', grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-2,
)

evaluation = dict(
    interval=num_iters_per_epoch * num_epochs,
    pipeline=nus_test_pipeline,
    # Multi-task evaluation metrics
    metric=['bbox', 'camera', 'depth', 'occ'],  # Evaluate both detection and OCC tasks
    # OCC evaluation configuration - will be passed as kwargs to dataset.evaluate()
    depth_eval=dict(
        max_depth=500,
    ),
    occ_eval=dict(
        class_names=occ_class_names + ['free'],
        occ_gt_root=nuscenes_data_root + 'occ_gts'
    )
)
# when use checkpoint, find_unused_parameters must be False
find_unused_parameters = False
checkpoint_config = dict(interval=10000, max_keep_ckpts=3)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
resume_from = None

dist_params = dict(
    backend='nccl',
    timeout=54000
)
# mAP: 0.4975
# mATE: 0.5851
# mASE: 0.2601
# mAOE: 0.3840
# mAVE: 0.2396
# mAAE: 0.1996
# NDS: 0.5819
# Eval time: 118.5s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.686   0.382   0.146   0.069   0.247   0.197
# truck   0.459   0.605   0.205   0.104   0.200   0.199
# bus     0.484   0.720   0.202   0.075   0.421   0.303
# trailer 0.291   0.925   0.230   0.604   0.167   0.133
# construction_vehicle    0.183   0.895   0.458   0.996   0.136   0.370
# pedestrian      0.561   0.591   0.285   0.442   0.292   0.141
# motorcycle      0.495   0.526   0.243   0.423   0.314   0.246
# bicycle 0.478   0.502   0.246   0.598   0.139   0.008
# traffic_cone    0.681   0.359   0.308   nan     nan     nan
# barrier 0.657   0.346   0.278   0.145   nan     nan
