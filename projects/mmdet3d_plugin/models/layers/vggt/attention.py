# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX_ATTN = True
except ImportError:
    HAS_FLEX_ATTN = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        window_size: int = 0,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        # Ensure consistent dtype after LayerNorm (may promote bf16 to fp32)
        q, k = q.to(v.dtype), k.to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Standard Attention
        
        if self.window_size > 0:
            dropout_p = self.attn_drop.p if self.training else 0.0
            w = int(self.window_size - 1)

            # Use Flash Attention if available (recommended for large sequence lengths)
            if HAS_FLASH_ATTN:
                q_f = q.transpose(1, 2)
                k_f = k.transpose(1, 2)
                v_f = v.transpose(1, 2)
                window_tuple = (w, w)

                need_cast = (q_f.dtype == torch.float32)
                if need_cast:
                    q_f, k_f, v_f = q_f.to(torch.bfloat16), k_f.to(torch.bfloat16), v_f.to(torch.bfloat16)

                x = flash_attn_func(
                    q_f, k_f, v_f,
                    dropout_p=dropout_p,
                    causal=False,
                    window_size=window_tuple
                )

                if need_cast:
                    x = x.to(torch.float32)
                x = x.transpose(1, 2)

            # Fallback 1: Use PyTorch 2.x flex_attention if available natively
            elif HAS_FLEX_ATTN:
                def sliding_window_mask(b, h, q_idx, kv_idx):
                    return torch.abs(q_idx - kv_idx) <= w

                block_mask = create_block_mask(
                    sliding_window_mask, 
                    B=q.shape[0], H=q.shape[1], 
                    Q_LEN=q.shape[2], KV_LEN=q.shape[2]
                )
                x = flex_attention(q, k, v, block_mask=block_mask)
            
            # Fallback 2: Basic F.scaled_dot_product_attention with boolean mask (may OOM on large lengths)
            else:
                L = q.shape[2]
                idx = torch.arange(L, device=q.device)
                dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
                mask = dist <= w
                
                x = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=mask.unsqueeze(0).unsqueeze(0),
                    dropout_p=dropout_p
                )
        else:
            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
