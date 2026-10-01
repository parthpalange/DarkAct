import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ModalEmbedding(nn.Module):
    """
    Modal Embedding Branch.
    Extracts initial spatiotemporal features G in R^{C x T x H x W}
    from raw video inputs (RGB or Thermal) using MViTv2 patch embedding stride.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3)),
            nn.BatchNorm3d(out_channels),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W) -> G: (B, C, T', H', W')
        return self.conv(x)


class EfficientSpatioTemporalAttention(nn.Module):
    """
    Divided Space-Time Attention using PyTorch's native FlashAttention
    (F.scaled_dot_product_attention) for O(N) memory scaling.
    Separately attends to spatial (H*W) and temporal (T) dimensions,
    preventing 1000+ GB memory spikes on high-resolution video tokens.
    """
    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True, drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.drop = drop
        
        # Spatial attention
        self.s_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.s_proj = nn.Linear(dim, dim)
        self.s_norm = nn.LayerNorm(dim)
        
        # Temporal attention
        self.t_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.t_proj = nn.Linear(dim, dim)
        self.t_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        # x: (B, T*H*W, C)
        B, N, C = x.shape
        
        # 1. Spatial Attention across (H*W) tokens for each frame
        x_norm = self.s_norm(x)
        x_s = x_norm.view(B, T, H * W, C).view(B * T, H * W, C)
        qkv_s = self.s_qkv(x_s).reshape(B * T, H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out_s = F.scaled_dot_product_attention(
            qkv_s[0], qkv_s[1], qkv_s[2],
            dropout_p=self.drop if self.training else 0.0
        )
        out_s = out_s.transpose(1, 2).reshape(B * T, H * W, C)
        out_s = self.s_proj(out_s)
        x = x + out_s.view(B, T * H * W, C)
        
        # 2. Temporal Attention across T tokens for each spatial position
        x_norm_t = self.t_norm(x)
        x_t = x_norm_t.view(B, T, H * W, C).permute(0, 2, 1, 3).reshape(B * H * W, T, C)
        qkv_t = self.t_qkv(x_t).reshape(B * H * W, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out_t = F.scaled_dot_product_attention(
            qkv_t[0], qkv_t[1], qkv_t[2],
            dropout_p=self.drop if self.training else 0.0
        )
        out_t = out_t.transpose(1, 2).reshape(B * H * W, T, C)
        out_t = self.t_proj(out_t)
        out_t = out_t.view(B, H * W, T, C).permute(0, 2, 1, 3).reshape(B, T * H * W, C)
        x = x + out_t
        
        return x


class TransformerBlock(nn.Module):
    """
    Vision Transformer Block with Divided Space-Time Attention and MLP.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.attn = EfficientSpatioTemporalAttention(dim, num_heads=num_heads, drop=drop)
        self.norm = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        x = self.attn(x, T, H, W)
        x = x + self.mlp(self.norm(x))
        return x


class TransformerStage(nn.Module):
    """
    A single stage of the multi-stage video Transformer.
    Downsamples spatial resolution and expands channels across stages.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        downsample: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample = downsample
        
        if downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(out_dim)
            )
        elif in_dim != out_dim:
            self.downsample_layer = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1),
                nn.BatchNorm3d(out_dim)
            )
        else:
            self.downsample_layer = nn.Identity()
            
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=out_dim, num_heads=num_heads)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        Returns: (B, C_out, T, H', W')
        """
        x = self.downsample_layer(x)
        B, C, T, H, W = x.shape
        # Flatten spatiotemporal dimensions to token sequence: (B, T*H*W, C)
        tokens = x.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H * W, C)
        
        for block in self.blocks:
            tokens = block(tokens, T, H, W)
            
        # Reshape back to 5D feature tensor
        out = tokens.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return out


class MultiStageBackbone(nn.Module):
    """
    Hierarchical Multi-Stage Video Transformer Backbone (MViTv2 compatible).
    Stage 1: dim 96
    Stage 2: dim 192
    Stage 3: dim 384
    Stage 4: dim 768
    """
    def __init__(
        self,
        variant: str = "mvitv2_b",
        in_channels: int = 96
    ):
        super().__init__()
        self.variant = variant
        
        if variant == "mvitv2_b":
            # 4 stages: depths [2, 3, 16, 3], channels [96, 192, 384, 768]
            # To allow fast and scalable training on various environments, default depths can be customized
            depths = [2, 3, 6, 2]
            dims = [96, 192, 384, 768]
            heads = [2, 4, 8, 16]
        elif variant == "mvitv2_s":
            depths = [1, 2, 4, 2]
            dims = [96, 192, 384, 768]
            heads = [2, 4, 8, 16]
        else: # compact / lightweight
            depths = [1, 1, 2, 1]
            dims = [64, 128, 256, 512]
            heads = [2, 4, 8, 8]
            
        self.stage_dims = dims
        self.stages = nn.ModuleList()
        current_dim = in_channels
        
        for i, (dim, depth, num_h) in enumerate(zip(dims, depths, heads)):
            downsample = (i > 0)
            stage = TransformerStage(
                in_dim=current_dim,
                out_dim=dim,
                depth=depth,
                num_heads=num_h,
                downsample=downsample
            )
            self.stages.append(stage)
            current_dim = dim

    def num_stages(self) -> int:
        return len(self.stages)
