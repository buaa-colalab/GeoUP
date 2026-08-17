from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from torch import Tensor

from mmdet.models.builder import NECKS
from .fpn_base import FPNBase

@NECKS.register_module()
class VGGTOccNeck(FPNBase):
    """Build occupancy FPN features from retained VGGT tokens."""

    def __init__(self,
                 backbone_channel: int,
                 num_register_tokens=4,
                 patch_size=14,
                 in_indices = [4, 11, 17, 23],
                 use_proj=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone_channel = backbone_channel
        self.in_indices = in_indices
        self.patch_size = patch_size
        self.patch_start_idx = 1 + num_register_tokens
        self.use_proj = use_proj

        assert len(self.in_channels) == 4, "in_channels must have two elements."

        self.norm = nn.LayerNorm(backbone_channel)

        if self.use_proj and isinstance(self.in_indices, list):
            self.proj = nn.Linear(backbone_channel * len(self.in_indices), backbone_channel)

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

        if self.patch_size == 14:
            self.resize_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Upsample(scale_factor=3.5, mode='bilinear', align_corners=False),
                        nn.Conv2d(self.in_channels[0], self.in_channels[0], kernel_size=3, padding=1)
                    ),
                    nn.Sequential(
                        nn.Upsample(scale_factor=1.75, mode='bilinear', align_corners=False),
                        nn.Conv2d(self.in_channels[1], self.in_channels[1], kernel_size=3, padding=1)
                    ),
                    nn.Sequential(
                        nn.Upsample(scale_factor=0.875, mode='bilinear', align_corners=False),
                        nn.Conv2d(self.in_channels[2], self.in_channels[2], kernel_size=3, padding=1)
                    ),
                    nn.Sequential(
                        nn.Upsample(scale_factor=0.4375, mode='bilinear', align_corners=False),
                        nn.Conv2d(self.in_channels[3], self.in_channels[3], kernel_size=3, padding=1)
                    ),
                ]
            )
        elif self.patch_size == 16:
            self.resize_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Upsample(scale_factor=4.0, mode='bilinear', align_corners=False),
                        nn.Conv2d(self.in_channels[0], self.in_channels[0], kernel_size=3, padding=1)
                    ),
                    nn.Sequential(
                        nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
                        nn.Conv2d(self.in_channels[1], self.in_channels[1], kernel_size=3, padding=1)
                    ),
                    nn.Sequential(
                        nn.Upsample(scale_factor=1.0, mode='bilinear', align_corners=False),
                        nn.Conv2d(self.in_channels[2], self.in_channels[2], kernel_size=3, padding=1)
                    ),
                    nn.Sequential(
                        nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                        nn.Conv2d(self.in_channels[3], self.in_channels[3], kernel_size=3, padding=1)
                    ),
                ]
            )
        else:
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

    def init_weights(self):
        if hasattr(self, 'proj'):
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.normal_(self.proj.bias, std=1e-3)
        super().init_weights()

    def forward(self, input: List, img: Tensor) -> Tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # assert len(input) == 1, "Input Shape Must be one"
        # build FPN
        inputs = []
        if len(img.shape) == 6:
            _, _, _, _, H, W = img.shape
        else:
            _, _, _, H, W = img.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        if self.use_proj and hasattr(self, 'proj'):
            selected_feats = []
            for feat_index in self.in_indices:
                feat = input[feat_index]
                if feat is None:
                    raise ValueError(
                        f'VGGTOccNeck requires VGGT feature at layer {feat_index}, '
                        'but the backbone did not retain it.')
                selected_feats.append(feat)
            in_x = torch.cat(selected_feats, dim=-1)
            in_x = self.proj(in_x)
            
            in_x = in_x[:, :, self.patch_start_idx:]
            # Do not force reshape batch dimension, trust input
            in_x = in_x.flatten(0, 1)
            in_x = self.norm(in_x)
            in_x = in_x.permute(0, 2, 1).reshape((in_x.shape[0], in_x.shape[-1], patch_h, patch_w))

            for idx in range(len(self.in_channels)):
                x = self.projects[idx](in_x)
                x = self.resize_layers[idx](x)
                inputs.append(x)
        else:
            out_idx = 0
            for idx in self.in_indices:
                feat = input[idx]
                if feat is None:
                    raise ValueError(
                        f'VGGTOccNeck requires VGGT feature at layer {idx}, '
                        'but the backbone did not retain it.')
                x = feat[:, :, self.patch_start_idx:]
                # Do not force reshape batch dimension, trust input
                x = x.flatten(0, 1)
                x = self.norm(x)
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
                x = self.projects[out_idx](x)
                x = self.resize_layers[out_idx](x)
                inputs.append(x)
                out_idx += 1

        return super().forward(inputs)
