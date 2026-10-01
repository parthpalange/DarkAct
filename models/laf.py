import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Standard scaled dot-product cross attention:
    Att(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Shapes: (B, N, C)
        B, N_q, C = q.shape
        _, N_k, _ = k.shape
        _, N_v, _ = v.shape
        
        Q = self.q_proj(q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, C)
        return self.out_proj(out)


class LightAdaptiveFusion(nn.Module):
    """
    Light-Adaptive Fusion (LAF) Module (Section 4.2).
    Adaptively fuses RGB and thermal representations according to scene illumination
    and modality reliability using dynamic query and cross-attention mechanisms.
    """
    def __init__(self, channels: int, num_heads: int = 4, y_channels: int | None = None):
        super().__init__()
        self.channels = channels
        # Created here, not lazily in forward: under DataParallel a lazy layer is rebuilt
        # with random weights on every step and never reaches the optimizer.
        y_channels = y_channels or channels
        self.proj_y = nn.Conv3d(y_channels, channels, kernel_size=1) if y_channels != channels else None

        # 1. Query mapping subnetwork phi(.) (Eq. 4)
        self.query_phi = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        # 2. Modality-specific key shallow conv block (Eq. 5): 3x3 dilated conv + Sigmoid
        self.key_conv_r = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.key_conv_t = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. Cross-modal value attention: F_A^r = Att(F^r, F^r, F^t), F_A^t = Att(F^t, F^t, F^r)
        self.cross_attn_r = CrossAttention(channels, num_heads=num_heads)
        self.cross_attn_t = CrossAttention(channels, num_heads=num_heads)
        
        # Hierarchical pooling MLP for value representation F^V: takes [MaxPool(FA); AvgPool(FA)] -> C
        self.value_mlp = nn.Sequential(
            nn.Linear(4 * channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels)
        )
        
        # 4. Refinement MLPs for F_d^r and F_d^t (Eq. 8)
        self.mlp_refine_r = nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels)
        )
        self.mlp_refine_t = nn.Sequential(
            nn.Linear(channels, channels),
            nn.PReLU(),
            nn.Linear(channels, channels)
        )
        
        # Residual output norm
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        F_r: torch.Tensor,
        F_t: torch.Tensor,
        Y_r: torch.Tensor,
        Y_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            F_r: Stage feature for RGB of shape (B, C, T, H, W)
            F_t: Stage feature for Thermal of shape (B, C, T, H, W)
            Y_r: Motion-enhanced RGB feature of shape (B, C_in, T_in, H_in, W_in) or matching F_r
            Y_t: Motion-enhanced Thermal feature of shape (B, C_in, T_in, H_in, W_in) or matching F_t
        Returns:
            F_r_out: F_r + F_LAF
            F_t_out: F_t + F_LAF
            F_LAF: Fused feature (B, C, T, H, W)
        """
        B, C, T, H, W = F_r.shape
        N = H * W
        
        # Resize Y_r and Y_t to match F_r spatial-temporal resolution if different
        if Y_r.shape != F_r.shape:
            # Interpolate spatial-temporal dimensions
            B_y, C_y, T_y, H_y, W_y = Y_r.shape
            Y_r_flat = Y_r.view(B_y * C_y, 1, T_y, H_y, W_y)
            Y_t_flat = Y_t.view(B_y * C_y, 1, T_y, H_y, W_y)
            Y_r = F.interpolate(Y_r_flat, size=(T, H, W), mode='trilinear', align_corners=False).view(B, C_y, T, H, W)
            Y_t = F.interpolate(Y_t_flat, size=(T, H, W), mode='trilinear', align_corners=False).view(B, C_y, T, H, W)
            if self.proj_y is not None:
                Y_r = self.proj_y(Y_r)
                Y_t = self.proj_y(Y_t)

        # Average over temporal dimension for spatial 2D operations, or process frame-wise
        # We process aggregated spatial tokens: (B, C, H, W)
        F_r_2d = F_r.mean(dim=2)  # (B, C, H, W)
        F_t_2d = F_t.mean(dim=2)  # (B, C, H, W)
        Y_r_2d = Y_r.mean(dim=2)  # (B, C, H, W)
        Y_t_2d = Y_t.mean(dim=2)  # (B, C, H, W)

        # 1. Light-Adaptive Query (Eq. 4): F^Q = phi([Y^r * F^r; Y^t * F^t])
        query_input = torch.cat([Y_r_2d * F_r_2d, Y_t_2d * F_t_2d], dim=1)  # (B, 2C, H, W)
        F_Q_spatial = self.query_phi(query_input)  # (B, C, H, W)
        F_Q = F_Q_spatial.view(B, C, N)  # (B, C, N)

        # 2. Modality-Specific Keys (Eq. 5): 3x3 dilated conv + Sigmoid
        F_K_r = self.key_conv_r(F_r_2d).view(B, C, N)  # (B, C, N)
        F_K_t = self.key_conv_t(F_t_2d).view(B, C, N)  # (B, C, N)

        # 3. Cross-Modal Value (Eq. 6)
        # Prepare token sequences: (B, N, C)
        F_r_tokens = F_r_2d.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        F_t_tokens = F_t_2d.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        
        # Cross attention: F_A^r = Att(F^r, F^r, F^t), F_A^t = Att(F^t, F^t, F^r)
        F_A_r = self.cross_attn_r(q=F_r_tokens, k=F_r_tokens, v=F_t_tokens)  # (B, N, C)
        F_A_t = self.cross_attn_t(q=F_t_tokens, k=F_t_tokens, v=F_r_tokens)  # (B, N, C)
        
        # Concat along channel: F_A = [F_A^r; F_A^t] -> shape (B, N, 2C)
        F_A = torch.cat([F_A_r, F_A_t], dim=-1)  # (B, N, 2C)
        
        # Hierarchical pooling: MaxPool and AvgPool along token dimension N
        # We compute local poolings or global + local to retain spatial diversity
        F_A_2d = F_A.permute(0, 2, 1).view(B, 2 * C, H, W)
        # MaxPool(F_A) and AvgPool(F_A) using kernel 3x3 with padding 1
        max_FA = F.max_pool2d(F_A_2d, kernel_size=3, stride=1, padding=1)  # (B, 2C, H, W)
        avg_FA = F.avg_pool2d(F_A_2d, kernel_size=3, stride=1, padding=1)  # (B, 2C, H, W)
        concat_FA = torch.cat([max_FA, avg_FA], dim=1).view(B, 4 * C, N).permute(0, 2, 1)  # (B, N, 4C)
        
        # Pass through value MLP -> (B, N, C) -> (B, C, N)
        F_V = self.value_mlp(concat_FA).permute(0, 2, 1)  # (B, C, N)

        # 4. Final Light-Adaptive Attention (Eq. 7):
        # F_d^{r/t} = Softmax(F^Q (F_K^{r/t})^T / sqrt(C)) F^V
        scale = 1.0 / math.sqrt(C)
        # (B, C, N) @ (B, N, C) -> (B, C, C)
        attn_r = F.softmax(torch.bmm(F_Q, F_K_r.transpose(1, 2)) * scale, dim=-1)
        attn_t = F.softmax(torch.bmm(F_Q, F_K_t.transpose(1, 2)) * scale, dim=-1)
        
        # Attention on value F^V: (B, C, C) @ (B, C, N) -> (B, C, N)
        F_d_r = torch.bmm(attn_r, F_V).permute(0, 2, 1)  # (B, N, C)
        F_d_t = torch.bmm(attn_t, F_V).permute(0, 2, 1)  # (B, N, C)

        # 5. Refinement and Sum (Eq. 8):
        # F_LAF = MLP(F_d^r) + MLP(F_d^t)
        F_LAF_tokens = self.mlp_refine_r(F_d_r) + self.mlp_refine_t(F_d_t)  # (B, N, C)
        F_LAF_tokens = self.norm(F_LAF_tokens)
        
        # Reshape to (B, C, 1, H, W) and broadcast across temporal dimension T
        F_LAF_2d = F_LAF_tokens.permute(0, 2, 1).view(B, C, H, W)
        F_LAF = F_LAF_2d.unsqueeze(2).expand(-1, -1, T, -1, -1)  # (B, C, T, H, W)

        # Residual connection to stage outputs
        F_r_out = F_r + F_LAF
        F_t_out = F_t + F_LAF

        return F_r_out, F_t_out, F_LAF
