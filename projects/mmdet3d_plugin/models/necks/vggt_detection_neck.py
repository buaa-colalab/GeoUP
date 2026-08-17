from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from torch import Tensor

from mmdet.models.builder import NECKS
from .fpn_base import FPNBase

@NECKS.register_module()
class VGGTDetectionNeck(FPNBase):
    """Build detection FPN features from retained VGGT tokens."""

    def __init__(self,
                 backbone_channel: int,
                 num_register_tokens=4,
                 patch_size=14,
                 in_indices = [4, 11, 17, 23],
                 use_pan=False,
                 pan_target_level=None,
                 **kwargs) -> None:
        conv_cfg = kwargs.get('conv_cfg', None)
        norm_cfg = kwargs.get('norm_cfg', None)
        act_cfg = kwargs.get('act_cfg', None)
        super().__init__(**kwargs)
        self.backbone_channel = backbone_channel
        self.in_indices = in_indices
        self.patch_size = patch_size
        self.patch_start_idx = 1 + num_register_tokens
        self.use_pan = use_pan
        self.pan_target_level = pan_target_level

        assert len(self.in_channels) == 4, "in_channels must have two elements."

        self.norm = nn.LayerNorm(backbone_channel)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.backbone_channel,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in self.in_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=self.in_channels[0], out_channels=self.in_channels[0], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=self.in_channels[2], out_channels=self.in_channels[2], kernel_size=3, stride=2, padding=1
                ),
                nn.Conv2d(
                    in_channels=self.in_channels[3], out_channels=self.in_channels[3], kernel_size=7, stride=4, padding=3
                ),
            ]
        )

        if self.use_pan:
            if self.pan_target_level is None:
                self.pan_stages = len(self.in_channels) - 1
            else:
                assert 1 <= self.pan_target_level < len(self.in_channels), \
                    "pan_target_level must be in [1, len(in_channels) - 1]."
                self.pan_stages = self.pan_target_level
            self.pan_downsample_convs = nn.ModuleList(
                [
                    ConvModule(
                        self.out_channels,
                        self.out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                    for _ in range(self.pan_stages)
                ]
            )
            self.pan_out_convs = nn.ModuleList(
                [
                    ConvModule(
                        self.out_channels,
                        self.out_channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                    for _ in range(self.pan_stages)
                ]
            )

    def forward(self, input: List, img: Tensor) -> Tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        inputs = []
        if img.dim() != 6:
            raise ValueError(
                f'VGGTDetectionNeck expects img with shape [B, N, T, C, H, W], '
                f'got {tuple(img.shape)}.')
        B, N, T, _, H, W = img.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out_idx = 0

        for idx in self.in_indices:
            feat = input[idx]
            if feat is None:
                raise ValueError(
                    f'VGGTDetectionNeck requires VGGT feature at layer {idx}, '
                    'but the backbone did not retain it.')
            expected_prefix = (B * N, T)
            if feat.shape[:2] != expected_prefix:
                raise ValueError(
                    f'VGGTDetectionNeck expects feature prefix {expected_prefix}, got '
                    f'{tuple(feat.shape)} at layer {idx}.')
            x = feat[:, :, self.patch_start_idx:]
            x = x.reshape(B * T * N, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[out_idx](x)
            x = self.resize_layers[out_idx](x)
            inputs.append(x)
            out_idx += 1

        outs = list(super().forward(inputs))
        if not self.use_pan:
            return tuple(outs)

        pan_outs = list(outs)
        for i in range(self.pan_stages):
            bottom_up_feat = self.pan_downsample_convs[i](pan_outs[i])
            fused_feat = pan_outs[i + 1] + bottom_up_feat
            pan_outs[i + 1] = self.pan_out_convs[i](fused_feat)

        return tuple(pan_outs)
