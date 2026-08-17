import numpy as np


class Metric_mIoU_Occ3D():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False):
        if num_classes == 18:
            self.class_names = [
                'others', 'barrier', 'bicycle', 'bus', 'car',
                'construction_vehicle', 'motorcycle', 'pedestrian',
                'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade',
                'vegetation', 'free'
            ]
        elif num_classes == 2:
            self.class_names = ['non-free', 'free']
        else:
            self.class_names = [f'class_{i}' for i in range(num_classes)]

        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) /
            self.occupancy_size[0])
        self.occ_ydim = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) /
            self.occupancy_size[1])
        self.occ_zdim = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) /
            self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """Build confusion matrix."""
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)
        labeled = np.sum(k)
        correct = np.sum(pred[k] == gt[k])

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int),
                minlength=n_cl ** 2).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):
        result = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        result[hist.sum(1) == 0] = float('nan')
        return result

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(
            n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

        if self.num_classes == 2:
            masked_semantics_pred = np.copy(masked_semantics_pred)
            masked_semantics_gt = np.copy(masked_semantics_gt)
            masked_semantics_pred[masked_semantics_pred < 17] = 0
            masked_semantics_pred[masked_semantics_pred == 17] = 1
            masked_semantics_gt[masked_semantics_gt < 17] = 0
            masked_semantics_gt[masked_semantics_gt == 17] = 1

        _, hist = self.compute_mIoU(
            masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += hist

    def count_miou(self):
        miou = self.per_class_iu(self.hist)
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes - 1):
            print(
                f'===> {self.class_names[ind_class]} - IoU = '
                f'{round(miou[ind_class] * 100, 2)}')

        print(
            f'===> mIoU of {self.cnt} samples: '
            f'{round(np.nanmean(miou[:self.num_classes - 1]) * 100, 2)}')

        return round(np.nanmean(miou[:self.num_classes - 1]) * 100, 2)
