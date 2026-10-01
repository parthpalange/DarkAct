import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionAwareAttention(nn.Module):
    """
    Motion-Aware Attention (MAA) Module (Section 4.1).
    Highlights human-centric, motion-salient regions through temporal differences
    and spatial-tolerant query attention along the channel dimension.
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        
        # Temporal Saliency MLP
        self.temporal_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.PReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.BatchNorm1d(in_channels)
        )
        
        # Spatial-Tolerant Query MLP: takes concatenated MaxPool & AvgPool features [2*C] -> C
        self.query_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.PReLU(),
            nn.Linear(in_channels, in_channels)
        )
        
        # Learnable mapping matrix W_M^A (Eq. 2)
        self.w_m_a = nn.Parameter(torch.Tensor(in_channels, in_channels))
        nn.init.kaiming_uniform_(self.w_m_a, a=math.sqrt(5))
        
        # Output fusion gate / modulation projection
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def _frame_norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Frame-wise spatial normalization N(.)"""
        # x: (B, C, H, W)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + eps
        return (x - mean) / std

    def compute_temporal_difference(self, G: torch.Tensor) -> torch.Tensor:
        """
        Computes temporal differences between consecutive frames (Eq. 1):
        Delta_t = N(|sin(G_t - G_{t-1})|)
        Boundary consistency: difference between first and last frames appended.
        Input G: (B, C, T, H, W)
        Returns: (B, C, T, H, W)
        """
        B, C, T, H, W = G.shape
        diffs = []
        for t in range(1, T):
            diff = torch.abs(torch.sin(G[:, :, t] - G[:, :, t - 1]))
            diff = self._frame_norm(diff)
            diffs.append(diff)
            
        # Boundary consistency: difference between first and last frame
        boundary_diff = torch.abs(torch.sin(G[:, :, 0] - G[:, :, -1]))
        boundary_diff = self._frame_norm(boundary_diff)
        diffs.append(boundary_diff)
        
        # Stack back along temporal dimension T -> (B, C, T, H, W)
        Delta = torch.stack(diffs, dim=2)
        return Delta

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        """
        Args:
            G: Tensor of shape (B, C, T, H, W)
        Returns:
            Y: Motion-enhanced feature tensor of shape (B, C, T, H, W)
        """
        B, C, T, H, W = G.shape
        N = H * W
        
        # 1. Temporal Motion Saliency E^S (Eq. 1)
        # Compute dynamic motion map Delta
        Delta = self.compute_temporal_difference(G)  # (B, C, T, H, W)
        
        # Aggregate temporal motion saliency over time T -> (B, C, H, W)
        delta_pool = Delta.mean(dim=2)  # (B, C, H, W)
        delta_flat = delta_pool.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        
        # Pass through temporal MLP
        delta_reshaped = delta_flat.reshape(-1, C)  # (B*N, C)
        es_reshaped = self.temporal_mlp(delta_reshaped)  # (B*N, C)
        E_S = es_reshaped.view(B, N, C).permute(0, 2, 1)  # (B, C, N)
        
        # 2. Spatial-Tolerant Query E^Q (Eq. 2)
        # Pool across temporal dimension to preserve spatial distribution
        max_g = G.max(dim=2)[0]  # (B, C, H, W)
        avg_g = G.mean(dim=2)    # (B, C, H, W)
        concat_pool = torch.cat([max_g, avg_g], dim=1)  # (B, 2C, H, W)
        concat_flat = concat_pool.view(B, 2 * C, N).permute(0, 2, 1)  # (B, N, 2C)
        
        # Dimensionality reduction via MLP
        concat_flat_reshaped = concat_flat.reshape(-1, 2 * C)
        E_bar = self.query_mlp(concat_flat_reshaped).view(B, N, C).permute(0, 2, 1)  # (B, C, N)
        
        # Learnable mapping: E^Q = E_bar @ W_M^A (Eq. 2)
        # E_bar is (B, C, N); with W_M^A (C, C), (W_M^A.T @ E_bar) gives (B, C, N)
        E_Q = torch.matmul(self.w_m_a, E_bar)  # (B, C, N)
        
        # 3. Channel Attention (Eq. 3): Softmax(E^Q (E^S)^T / sqrt(C)) E^S
        # (B, C, N) @ (B, N, C) -> (B, C, C)
        attn_scores = torch.bmm(E_Q, E_S.transpose(1, 2)) / math.sqrt(C)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, C, C)
        
        # Saliency-attended feature: (B, C, C) @ (B, C, N) -> (B, C, N)
        salient_att = torch.bmm(attn_weights, E_S)  # (B, C, N)
        salient_map = salient_att.view(B, C, H, W)   # (B, C, H, W)
        
        # Modulation gate applied to feature G
        gate = self.gate_conv(salient_map).unsqueeze(2)  # (B, C, 1, H, W)
        Y = G * gate + G  # Residual motion-enhanced feature (B, C, T, H, W)
        
        return Y
