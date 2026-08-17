# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


from typing import List, Tuple, Union
import warnings

from mmcv.runner import BaseModule
from projects.mmdet3d_plugin.core.utils import force_fp32
from mmdet.models import HEADS
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.vggt.head_act import activate_head
from ..layers.vggt.utils import create_uv_grid, position_grid_to_embed
from projects.mmdet3d_plugin.models.utils.misc import MLN

@HEADS.register_module()
class DPTHeadPseudo(BaseModule):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
        with_pseudo_depth (bool, optional): Whether to output pseudo depth for detection task. Default is False.
    """

    def __init__(
        self,
        dim_in,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] =[4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        use_intrinsics: bool = False,
        gradient_loss_fn = None,
        main_loss_type: str = "l2",       # "l2" 鎴?"smooth_l1"
        smooth_l1_beta: float = 1.0,
        valid_range = -1,
        loss_scale = 1,
        use_full_loss = False,
        frozen = False,
        init_cfg=None,
        with_pseudo_depth: bool = True,
    ) -> None:
        super(DPTHeadPseudo, self).__init__(init_cfg=init_cfg)
        self.bf16_enabled = False
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.gradient_loss_fn = gradient_loss_fn
        self.valid_range = valid_range
        self.loss_scale = loss_scale
        self.use_full_loss = use_full_loss
        self.main_loss_type = main_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.with_pseudo_depth = with_pseudo_depth
        self._pos_embed_cache = {}

        if isinstance(dim_in, int):
            self.norm = nn.LayerNorm(dim_in)
            # Projection layers for each output channel from tokens.
            self.projects = nn.ModuleList([nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
            )
            # Resize layers for upsampling feature maps.
            self.resize_layers = nn.ModuleList([
                    nn.ConvTranspose2d(
                        in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                    ),
                    nn.ConvTranspose2d(
                        in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                    ),
                    nn.Identity(),
                    nn.Conv2d(
                        in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                    ),
                ]
            )
        else:
            self.norm = nn.Identity()
            self.projects = nn.ModuleList([nn.Identity() for oc in out_channels]
            )
            self.resize_layers = nn.ModuleList(
                [nn.Identity() for oc in out_channels]
            )

        self.use_intrinsics = use_intrinsics
        
        if self.use_intrinsics:
            self.cam_embeding = MLN(4, dim_in)

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )
            
            # Additional branch for Pseudo Depth
            if self.with_pseudo_depth:
                self.scratch.pseudo_output_conv2 = nn.Sequential(
                    nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
                )
        
        if frozen == True:
            for param in self.parameters():
                param.requires_grad = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if getattr(self, 'with_pseudo_depth', False):
            pseudo_prefix = prefix + 'scratch.pseudo_output_conv2.'
            output_prefix = prefix + 'scratch.output_conv2.'
            
            has_pseudo = any(
                k.startswith(pseudo_prefix) for k in state_dict.keys())
            
            if not has_pseudo:
                warnings.warn(
                    'Pseudo depth branch weights not found in checkpoint. '
                    'Cloning main branch weights to pseudo branch.')
                for k in list(state_dict.keys()):
                    if k.startswith(output_prefix):
                        pseudo_k = k.replace(output_prefix, pseudo_prefix)
                        state_dict[pseudo_k] = state_dict[k].clone()
                        
        super(DPTHeadPseudo, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    @force_fp32(apply_to=("pred_depth", "pred_depth_conf", "gt_depth", "gt_depth_mask"))
    def loss(
        self,
        pred_depth,
        pred_depth_conf,
        gt_depth,
        gt_depth_mask,
        gamma=1.0, 
        alpha=0.2,
        use_full_loss=True,
    ):

        gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
        gt_depth = gt_depth[..., None]
        valid_weight = (
            gt_depth_mask.sum(dtype=torch.int64) >= 100
        ).to(dtype=pred_depth.dtype)
        loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=self.gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=self.valid_range,
                                             loss_type=self.main_loss_type, smooth_l1_beta=self.smooth_l1_beta)
        loss_conf = loss_conf * valid_weight
        loss_reg = loss_reg * valid_weight
        loss_grad = loss_grad * valid_weight
        if self.use_full_loss and use_full_loss:
            loss_dict = {
                f"loss_conf_depth": loss_conf * self.loss_scale,
                f"loss_reg_depth": loss_reg * self.loss_scale,
                f"loss_grad_depth": loss_grad * self.loss_scale,
            }
        else:
            loss_dict = {
                f"loss_conf_depth": loss_conf * 0.01,
                f"loss_reg_depth": loss_reg * self.loss_scale,
                # f"loss_grad_depth": loss_grad,
            }
        return loss_dict

    @force_fp32(apply_to=("aggregated_tokens_list", "images", "intrinsics"))
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, ...]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - If with_pseudo_depth=True: Tuple of (predictions, confidence, pseudo_predictions, pseudo_confidence)
                - Otherwise: Tuple of (predictions, confidence) both with shape[B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, intrinsics, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []
        all_conf =[]
        
        if self.with_pseudo_depth:
            all_pseudo_preds = []
            all_pseudo_conf =[]

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, intrinsics, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_outputs = self._forward_impl(
                    aggregated_tokens_list, images, intrinsics, patch_start_idx, frames_start_idx, frames_end_idx
                )
                if self.with_pseudo_depth:
                    chunk_preds, chunk_conf, chunk_pseudo_preds, chunk_pseudo_conf = chunk_outputs
                    all_preds.append(chunk_preds)
                    all_conf.append(chunk_conf)
                    all_pseudo_preds.append(chunk_pseudo_preds)
                    all_pseudo_conf.append(chunk_pseudo_conf)
                else:
                    chunk_preds, chunk_conf = chunk_outputs
                    all_preds.append(chunk_preds)
                    all_conf.append(chunk_conf)

        # Concatenate results along the sequence dimension
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            if self.with_pseudo_depth:
                return (
                    torch.cat(all_preds, dim=1), 
                    torch.cat(all_conf, dim=1),
                    torch.cat(all_pseudo_preds, dim=1),
                    torch.cat(all_pseudo_conf, dim=1)
                )
            else:
                return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)
    
    def normalize_intrinsics(self, K, w, h):
        fx = K[:, :, 0, 0] / max(w, h)  
        fy = K[:, :, 1, 1] / max(w, h)
        cx = K[:, :, 0, 2] / w
        cy = K[:, :, 1, 2] / h
        return torch.stack([fx, fy, cx, cy], dim=-1)   # (B,4)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, ...]: Feature maps or (predictions, confidence). 
                                          Or (predictions, confidence, pseudo_predictions, pseudo_confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
            intrinsics = intrinsics[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out =[]
        dpt_idx = 0
        norm_intrinsics = None
        if self.use_intrinsics:
            norm_intrinsics = self.normalize_intrinsics(intrinsics, W, H)
            norm_intrinsics = norm_intrinsics.reshape(B * S, 1, -1)

        for layer_idx in self.intermediate_layer_idx:
            layer_tokens = aggregated_tokens_list[layer_idx]
            if layer_tokens is None:
                raise ValueError(
                    f"DPTHeadPseudo requires VGGT feature at layer {layer_idx}, "
                    "but the backbone did not retain it."
                )
            x = layer_tokens[:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])

            if self.use_intrinsics:
                x = self.cam_embeding(
                    x, norm_intrinsics.expand(-1, x.shape[1], -1))

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        # Save fused features for pseudo branch if needed
        out_features = out
        
        out = self.scratch.output_conv2(out_features)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        
        if self.with_pseudo_depth:
            pseudo_out = self.scratch.pseudo_output_conv2(out_features)
            pseudo_preds, pseudo_conf = activate_head(pseudo_out, activation=self.activation, conf_activation=self.conf_activation)
            pseudo_preds = pseudo_preds.view(B, S, *pseudo_preds.shape[1:])
            pseudo_conf = pseudo_conf.view(B, S, *pseudo_conf.shape[1:])
            return preds, conf, pseudo_preds, pseudo_conf
            
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        cache_key = (
            patch_h,
            patch_w,
            x.shape[1],
            float(W) / float(H),
            x.dtype,
            str(x.device),
            ratio,
        )
        pos_embed = self._pos_embed_cache.get(cache_key)
        if pos_embed is None:
            pos_embed = create_uv_grid(
                patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
            pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
            pos_embed = (pos_embed * ratio).permute(2, 0, 1)[None]
            self._pos_embed_cache[cache_key] = pos_embed
        pos_embed = pos_embed.expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks =[
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1, loss_type="l2", smooth_l1_beta=1.0):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # ======================
    # 1) 鏍规嵁 loss_type 绠楀熀纭€鍥炲綊 loss锛坧er-pixel scalar锛?    # ======================
    diff = gt - pred

    if loss_type.lower() == "l2":
        loss_reg = torch.norm(diff, dim=-1)
    elif loss_type.lower() in["smooth_l1", "huber"]:
        # SmoothL1 / Huber: compute dense per-pixel losses and defer the valid
        # mask to the reduction step to avoid boolean indexing.
        base_loss = F.smooth_l1_loss(
            pred,
            gt,
            reduction="none",
            beta=smooth_l1_beta,
        )
        loss_reg = base_loss.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Use 'l2' or 'smooth_l1'.")

    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    valid_mask = mask.to(dtype=torch.bool)
    safe_conf = torch.where(valid_mask, conf, conf.new_ones(()))
    loss_conf = gamma * loss_reg * safe_conf - alpha * torch.log(safe_conf)
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = pred.new_zeros(())

    # Prepare confidence for gradient loss if needed
    if gradient_loss_fn is not None and "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if gradient_loss_fn is not None and "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif gradient_loss_fn is not None and "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )
    loss_grad = check_and_fix_inf_nan(loss_grad, "loss_grad")

    # Process confidence-weighted loss
    if valid_range > 0:
        loss_conf = mean_by_quantile(
            loss_conf, valid_range, loss_name="loss_conf_depth",
            valid_mask=valid_mask)
        loss_reg = mean_by_quantile(
            loss_reg, valid_range, loss_name="loss_reg_depth",
            valid_mask=valid_mask)
    else:
        loss_conf = masked_mean(
            check_and_fix_inf_nan(loss_conf, f"loss_conf_depth"), valid_mask)
        loss_reg = masked_mean(
            check_and_fix_inf_nan(loss_reg, f"loss_reg_depth"), valid_mask)

    return loss_conf.float(), loss_grad.float(), loss_reg.float()


def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside[-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor

    # Avoid Python-side bool reductions on CUDA tensors. Expressions like
    # `torch.isnan(x).any() or torch.isinf(x).any()` force device syncs and
    # become disproportionately expensive on large loss tensors.
    if input_tensor.requires_grad:
        input_tensor = torch.nan_to_num(
            input_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if hard_max is not None:
            input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)
        return input_tensor

    input_tensor = input_tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    if hard_max is not None:
        input_tensor = input_tensor.clamp_(min=-hard_max, max=hard_max)
    return input_tensor

def masked_mean(loss_tensor, valid_mask):
    valid_mask_f = valid_mask.to(dtype=loss_tensor.dtype)
    masked_sum = (loss_tensor * valid_mask_f).sum()
    masked_count = valid_mask_f.sum().clamp_min(1)
    return masked_sum / masked_count


def mean_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100, loss_name="default", valid_mask=None):
    """
    Compute a quantile-filtered mean without Tensor-to-bool Python branches.

    If filtering would leave too few elements, this falls back to the mean over
    the full tensor, matching the previous behavior while avoiding
    `aten::is_nonzero` / `.item()` sync points on CUDA tensors.
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(loss_tensor, dtype=torch.bool)

    if loss_tensor.numel() <= min_elements:
        loss_tensor = check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=hard_max)
        return masked_mean(loss_tensor, valid_mask)

    if loss_tensor.numel() > 100000000:
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]
        valid_mask = valid_mask.view(-1)[indices]

    loss_tensor = check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=hard_max)

    flat_loss = loss_tensor.reshape(-1)
    flat_valid = valid_mask.reshape(-1)
    valid_count = flat_valid.sum(dtype=torch.int64)

    sort_input = torch.where(
        flat_valid, flat_loss, flat_loss.new_full((), float("inf")))
    sorted_loss, _ = torch.sort(sort_input)
    quantile_index = torch.round(
        (valid_count.to(torch.float32) - 1).clamp_min(0) * valid_range
    ).to(torch.int64)
    quantile_index = quantile_index.clamp(max=sorted_loss.numel() - 1)
    quantile_thresh = sorted_loss.gather(0, quantile_index.view(1)).squeeze(0)
    quantile_thresh = torch.minimum(
        quantile_thresh, flat_loss.new_tensor(hard_max))

    quantile_mask = flat_valid & (flat_loss < quantile_thresh)
    filtered_mean = masked_mean(flat_loss, quantile_mask)
    original_mean = masked_mean(flat_loss, flat_valid)
    use_filtered = (
        quantile_mask.sum(dtype=torch.int64) > min_elements
    ).to(dtype=flat_loss.dtype)
    return filtered_mean * use_filtered + original_mean * (1 - use_filtered)

def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total

def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    # Extract valid normals. Using the indexed tensor shape avoids a Tensor-to-
    # bool sync from checks like `torch.sum(all_valid) < 10`.
    pred_normals = pred_normals[all_valid]
    gt_normals = gt_normals[all_valid]
    if pred_normals.shape[0] < 10:
        return prediction.new_zeros(())

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    loss = check_and_fix_inf_nan(loss, "normal_loss")

    if conf is not None:
        # Apply confidence weighting
        conf = conf[None, ...].expand(4, -1, -1, -1)
        conf = conf[all_valid]

        loss = gamma * loss * conf - alpha * torch.log(conf)
        return loss.mean()
    return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)
    grad_x = check_and_fix_inf_nan(grad_x, "grad_x")
    grad_y = check_and_fix_inf_nan(grad_y, "grad_y")

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M).clamp_min(1)
    grad_loss = torch.sum(grad_loss) / divisor
    return check_and_fix_inf_nan(grad_loss, "grad_loss")

def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape[4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids
