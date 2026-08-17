# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from ..layers.vggt.block import Block
from ..layers.vggt.rope import RotaryPositionEmbedding2D, PositionGetter
from ..layers.vggt.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

from projects.mmdet3d_plugin.core.utils import force_fp32

from mmcv.runner.base_module import BaseModule
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class AggregatorVGGT(BaseModule):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each DINOv2 patch.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention.
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """
    CAMERA_META_KEYS = ('intrinsics', 'cam_extrinsics_global')
    DEFAULT_SEQ_INFO = dict(seq_length=4, batch_size=16)

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=('frame', 'global', 'view'),
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        out_indices=None,
        frozen=False,
        with_cp=False,
        init_cfg=None,
        dino_embed_dim=1024,
        seq_info=None,
        img_hw=(224, 672),
    ):
        super().__init__(init_cfg)

        self.bf16_enabled = False

        default_seq_info = dict(self.DEFAULT_SEQ_INFO)
        if seq_info is None:
            seq_info = default_seq_info
        else:
            seq_info = {**default_seq_info, **seq_info}
        self.seq_mode = True
        self.seq_info = dict(seq_info)
        self.scene_token_list = [None for _ in range(self.seq_info['batch_size'])]
        self.patch_embed_memory = [None for _ in range(self.seq_info['batch_size'])]
        self.intrinsics_memory = [None for _ in range(self.seq_info['batch_size'])]
        self.cam_extrinsics_global_memory = [None for _ in range(self.seq_info['batch_size'])]
        self.memory_head_list = [0 for _ in range(self.seq_info['batch_size'])]

        self.depth = depth
        vit_models = {
            "dinov2_vitl14_reg": vit_large,
            "dinov2_vitb14_reg": vit_base,
            "dinov2_vits14_reg": vit_small,
            "dinov2_vitg2_reg": vit_giant2,
        }
        self.patch_embed = vit_models[patch_embed](
            img_size=img_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            block_chunks=0,
            init_values=1.0,
            with_cp=with_cp
        )
        if hasattr(self.patch_embed, "mask_token"):
            self.patch_embed.mask_token.requires_grad_(False)

        self.rope = RotaryPositionEmbedding2D(freq=rope_freq)
        self.position_getter = PositionGetter()

        block_kwargs = dict(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
            rope=self.rope,
        )

        self.frame_blocks = nn.ModuleList([
            block_fn(
                **block_kwargs,
                with_cp=with_cp)
            for i in range(depth)
        ])
        self.global_blocks = nn.ModuleList([
            block_fn(
                **block_kwargs,
                with_cp=with_cp)
            for i in range(depth)
        ])
        self.view_blocks = nn.ModuleList([
            block_fn(
                **block_kwargs,
                with_cp=with_cp)
            for i in range(depth)
        ])

        self.aa_order = list(aa_order)
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest.
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        self.out_indices = out_indices

        self.position_dim = 6
        self.position_encoder = nn.Sequential(
            nn.Linear(self.position_dim, embed_dim // 16),
            nn.ReLU(),
            nn.Linear(embed_dim // 16, embed_dim),
        )

        if frozen:
            for param in self.parameters():
                param.requires_grad_(False)

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

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device=None, coords_h=None, coords_w=None):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
            coords_h (torch.tensor): sampled pixel rows in the source image
            coords_w (torch.tensor): sampled pixel cols in the source image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)
        device = c2w.device if device is None else device

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        fx = fxfycxcy[:, 0:1]
        fy = fxfycxcy[:, 1:2]
        cx = fxfycxcy[:, 2:3]
        cy = fxfycxcy[:, 3:4]

        if coords_h is not None or coords_w is not None:
            coords_h = coords_h.to(device=device, dtype=fx.dtype)
            coords_w = coords_w.to(device=device, dtype=fx.dtype)
            h, w = coords_h.numel(), coords_w.numel()
            y, x = torch.meshgrid(coords_h, coords_w, indexing="ij")
        else:
            h_orig = max(int(2 * cy.max().item()), 1)
            w_orig = max(int(2 * cx.max().item()), 1)
            h = h_orig if h is None else h
            w = w_orig if w is None else w

            if h_orig != h or w_orig != w:
                scale = fxfycxcy.new_tensor([w / w_orig, h / h_orig, w / w_orig, h / h_orig])
                fxfycxcy = fxfycxcy * scale
                fx = fxfycxcy[:, 0:1]
                fy = fxfycxcy[:, 1:2]
                cx = fxfycxcy[:, 2:3]
                cy = fxfycxcy[:, 3:4]

            y, x = torch.meshgrid(
                torch.arange(h, device=device, dtype=fx.dtype),
                torch.arange(w, device=device, dtype=fx.dtype),
                indexing="ij",
            )

        x = x.reshape(1, -1).expand(b * v, -1)
        y = y.reshape(1, -1).expand(b * v, -1)
        x = (x + 0.5 - cx) / fx
        y = (y + 0.5 - cy) / fy
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d

    @force_fp32(apply_to=('img', 'camera_metas'))
    def get_plucker_raymap(self, img, camera_metas):
        pad_h, pad_w = img.shape[2:] # 476, 518
        BN, C, H, W = img.shape[0], img.shape[1], img.shape[2] // self.patch_size, img.shape[3] // self.patch_size # 20, 3, 34, 37
        T = camera_metas['intrinsics'].shape[1]
        BNT = BN * T
        coords_h = torch.div(torch.arange(H, device=img.device) * pad_h, H, rounding_mode='floor')
        coords_w = torch.div(torch.arange(W, device=img.device) * pad_w, W, rounding_mode='floor')

        intrinsics = camera_metas['intrinsics'].to(img) # BN, T, 3, 3
        fx = intrinsics[:, :, 0, 0]  # BN, T
        fy = intrinsics[:, :, 1, 1]  # BN, T
        cx = intrinsics[:, :, 0, 2]  # BN, T
        cy = intrinsics[:, :, 1, 2]  # BN, T
        fxfycxcy = torch.stack([fx, fy, cx, cy], dim=2)  # BN, T, 4

        extrinsics = camera_metas['cam_extrinsics_global'][:, :, :3, :4].to(img) # BN, T, 3, 4
        global2cam = extrinsics.new_zeros((*extrinsics.shape[:2], 4, 4))
        global2cam[:, :, :3, :4] = extrinsics
        global2cam[:, :, 3, 3] = 1
        cam2global = self.invert_rigid_transform(global2cam) # BN, T, 4, 4

        global2cam_ref = global2cam[:, 0:1]
        c2w_rel = global2cam_ref @ cam2global

        ray_o, ray_d = self.compute_rays(c2w_rel, fxfycxcy, device=img.device, coords_h=coords_h, coords_w=coords_w)
        pose_cond = torch.cat([torch.cross(ray_o, ray_d, dim=2), ray_d], dim=2) # BN, T, 6, H, W

        pose_cond = pose_cond.view(BNT, 6, H * W).permute(0, 2, 1) # BNT, H * W, 6

        return pose_cond.to(img.dtype)

    def position_embedding_plucker(self, img, camera_metas):
        pose_cond = self.get_plucker_raymap(img, camera_metas)
        pose_cond = self.position_encoder(pose_cond) # BNT, H * W, embed_dims

        return pose_cond # BNT, H * W, embed_dims

    def update_memory(self, patch_token, intrinsics, cam_extrinsics_global, scene_token):
        B, P, C = patch_token.shape
        history_length = self.seq_info['seq_length'] - 1
        return_patch_token = []
        return_intrinsics = []
        return_cam_extrinsics_global = []
        offsets = torch.arange(history_length, device=patch_token.device)
        for i in range(B):
            patch_token_b = patch_token[i].detach()
            intrinsics_b = intrinsics[i].detach()
            cam_extrinsics_global_b = cam_extrinsics_global[i].detach()
            is_new_scene = self.scene_token_list[i] != scene_token[i]
            if is_new_scene:
                self.scene_token_list[i] = scene_token[i]
                self.memory_head_list[i] = 0
                self.patch_embed_memory[i] = patch_token_b.unsqueeze(0).expand(history_length, -1, -1).clone()
                self.intrinsics_memory[i] = intrinsics_b.unsqueeze(0).expand(history_length, -1, -1).clone()
                self.cam_extrinsics_global_memory[i] = cam_extrinsics_global_b.unsqueeze(0).expand(history_length, -1, -1).clone()

            head = self.memory_head_list[i]
            indices = (head + offsets) % history_length
            return_patch_token.append(self.patch_embed_memory[i].index_select(0, indices))
            return_intrinsics.append(self.intrinsics_memory[i].index_select(0, indices))
            return_cam_extrinsics_global.append(
                self.cam_extrinsics_global_memory[i].index_select(0, indices))

            new_head = (head - 1) % history_length
            self.memory_head_list[i] = new_head
            self.patch_embed_memory[i][new_head].copy_(patch_token_b)
            self.intrinsics_memory[i][new_head].copy_(intrinsics_b)
            self.cam_extrinsics_global_memory[i][new_head].copy_(cam_extrinsics_global_b)
        return_patch_token = torch.stack(return_patch_token)
        return_intrinsics = torch.stack(return_intrinsics)
        return_cam_extrinsics_global = torch.stack(return_cam_extrinsics_global)
        return return_patch_token, return_intrinsics, return_cam_extrinsics_global

    def forward(self, images: torch.Tensor, **data) -> List[torch.Tensor]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, N, T, 3, H, W].
                B: batch size, N: number of views, T: sequence length.

        Returns:
            List[torch.Tensor]: The outputs from the attention blocks.
        """
        B, num_view, T, C, H, W = images.shape

        current_images = images.reshape(B * num_view, C, H, W)
        patch_tokens = self.patch_embed(current_images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens['x_norm_patchtokens']

        current_camera_metas = {}
        camera_metas = {}
        for key in self.CAMERA_META_KEYS:
            value = data[key]
            value = value.reshape(B * num_view, *value.shape[3:])
            current_camera_metas[key] = value
            camera_metas[key] = value.unsqueeze(1)

        scene_tokens = [meta['scene_token'] for meta in data['img_metas']]
        scene_token_list = [scene_token for scene_token in scene_tokens for _ in range(num_view)]

        _, patch_count, embed_dim = patch_tokens.shape
        memory_seq_length = self.seq_info['seq_length']
        save_patch_tokens, save_intrinsics, save_cam_extrinsics_global = self.update_memory(
            patch_tokens,
            current_camera_metas['intrinsics'],
            current_camera_metas['cam_extrinsics_global'],
            scene_token_list
        )

        patch_tokens = torch.cat([patch_tokens.unsqueeze(1), save_patch_tokens], dim=1)
        S = memory_seq_length
        self.curr_seq_length = S
        patch_tokens = patch_tokens.reshape(B * num_view, S, patch_count, embed_dim)
        camera_metas['intrinsics'] = torch.cat([camera_metas['intrinsics'], save_intrinsics], dim=1)
        camera_metas['cam_extrinsics_global'] = torch.cat([camera_metas['cam_extrinsics_global'], save_cam_extrinsics_global], dim=1)
        patch_tokens = patch_tokens.view(-1, patch_count, embed_dim)

        pos_3d = self.position_embedding_plucker(current_images, camera_metas)
        pos_3d = torch.nan_to_num(pos_3d, nan=0.0, posinf=0.0, neginf=0.0)
        patch_tokens = patch_tokens + pos_3d

        token_batch = B * num_view
        camera_token = slice_expand_and_flatten(self.camera_token, token_batch, S)
        register_token = slice_expand_and_flatten(self.register_token, token_batch, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, token_count, embed_dim = tokens.shape

        pos = self.position_getter(token_batch * S, H // self.patch_size, W // self.patch_size, device=patch_tokens.device)
        pos = pos + 1
        pos_special = torch.zeros(token_batch * S, self.patch_start_idx, 2, device=patch_tokens.device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)

        block_indices = {'frame': 0, 'global': 0, 'view': 0}
        output_list = [None] * self.depth
        for _ in range(self.aa_block_num):
            frame_intermediates = None
            view_intermediates = None
            for attn_type in self.aa_order:
                if attn_type == 'frame':
                    tokens, block_indices['frame'], frame_intermediates = self.process_frame_attention(
                        tokens, token_batch, S, token_count, embed_dim, block_indices['frame'], pos=pos)
                elif attn_type == 'global':
                    tokens, block_indices['global'], _ = self.process_global_attention(
                        tokens, token_batch, S, token_count, embed_dim, block_indices['global'], pos=pos)
                elif attn_type == 'view':
                    tokens, block_indices['view'], view_intermediates = self.process_view_attention(
                        tokens, B, S, num_view, token_count, embed_dim, block_indices['view'], pos=pos)

            start_depth = block_indices['frame'] - len(frame_intermediates)
            for offset, (frame_tokens, view_tokens) in enumerate(zip(frame_intermediates, view_intermediates)):
                output_depth = start_depth + offset
                if self.out_indices is None or output_depth in self.out_indices:
                    output_list[output_depth] = torch.cat([frame_tokens, view_tokens], dim=-1)

        return output_list

    def process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def process_view_attention(self, tokens, B, S, num_view, P, C, view_idx, pos=None):
        token_batch = B * num_view
        if tokens.shape != (token_batch, S * P, C):
            tokens = tokens.view(token_batch, S, P, C).view(token_batch, S * P, C)
        tokens = tokens.view(B, num_view, S, P, C).permute(0, 2, 1, 3, 4).contiguous().view(
            B * S, num_view * P, C)
        if pos is not None and pos.shape != (token_batch, S * P, 2):
            pos = pos.view(token_batch, S, P, 2).view(token_batch, S * P, 2)
        pos = pos.view(B, num_view, S, P, 2).permute(0, 2, 1, 3, 4).contiguous().view(B * S, num_view * P, 2)
        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.view_blocks[view_idx](tokens, pos=pos)
            tokens = tokens.view(B, S, num_view, P, C).permute(0, 2, 1, 3, 4).contiguous().view(
                token_batch, S * P, C)
            pos = pos.view(B, S, num_view, P, 2).permute(0, 2, 1, 3, 4).contiguous().view(token_batch, S * P, 2)
            view_idx += 1
            intermediates.append(tokens.view(token_batch, S, P, C))

        return tokens, view_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
