import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """
    Sobel Feature Consistency Loss (L_Sobel-fcc).
    L_Sobel-fcc = 1 - (1/(H*W)) sum_{i,j} cos(Omega(Y^r)_{i,j}, Omega(Y^t)_{i,j})
    Calculates gradient orientations using Sobel kernels and maximizes cosine similarity
    between edge feature orientations of RGB and Thermal streams.
    """
    def __init__(self):
        super().__init__()
        # 3x3 Sobel kernels for horizontal and vertical edge gradients
        kx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=torch.float32)
        ky = torch.tensor([[-1.0, -2.0, -1.0],
                           [ 0.0,  0.0,  0.0],
                           [ 1.0,  2.0,  1.0]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _edge_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        gx = F.conv2d(x_reshaped, self.kx, padding=1)
        gy = F.conv2d(x_reshaped, self.ky, padding=1)
        # Safe normalization to avoid division by zero or NaN gradients
        norm = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        gx_norm = (gx / norm).view(B, C, H, W)
        gy_norm = (gy / norm).view(B, C, H, W)
        return gx_norm, gy_norm

    def forward(self, Y_r: torch.Tensor, Y_t: torch.Tensor) -> torch.Tensor:
        # Y_r, Y_t: (B, C, T, H, W) or (B, C, H, W)
        if Y_r.dim() == 5:
            Y_r = Y_r.mean(dim=2)  # Average across temporal dimension
            Y_t = Y_t.mean(dim=2)
            
        gx_r, gy_r = self._edge_gradient(Y_r)
        gx_t, gy_t = self._edge_gradient(Y_t)
        
        # Dot product of normalized gradient orientations = cosine similarity
        cos_sim = gx_r * gx_t + gy_r * gy_t
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        loss = 1.0 - cos_sim.mean()
        return loss


class ModalityContrastiveLoss(nn.Module):
    """
    Modality Contrastive Loss (L_MC):
    L_MC = log(1 + sum_{i != j} exp( (Y_i^r . Y_j^t) / tau - (Y_i^r . Y_i^t) / tau ))
    Numerically stabilized with clamping to prevent exp() overflow.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, Y_r: torch.Tensor, Y_t: torch.Tensor) -> torch.Tensor:
        B = Y_r.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=Y_r.device, requires_grad=True)
            
        feat_r = F.adaptive_avg_pool3d(Y_r, (1, 1, 1)).view(B, -1)
        feat_t = F.adaptive_avg_pool3d(Y_t, (1, 1, 1)).view(B, -1)
        
        # Normalize
        feat_r = F.normalize(feat_r, p=2, dim=1, eps=1e-6)
        feat_t = F.normalize(feat_t, p=2, dim=1, eps=1e-6)
        
        # Similarity matrix: (B, B)
        sim_matrix = torch.matmul(feat_r, feat_t.t()) / self.temperature
        
        # Positive pairs: diagonal (B,)
        pos = torch.diag(sim_matrix)
        
        # Difference with positive: (B, B) clamped to prevent exp overflow
        diff = (sim_matrix - pos.unsqueeze(1)).clamp(max=20.0)
        
        # Mask out diagonal (i == j)
        mask = ~torch.eye(B, dtype=torch.bool, device=Y_r.device)
        exp_diff = torch.exp(diff) * mask.float()
        
        # L_MC = log(1 + sum_{i != j} exp(...))
        loss = torch.log(1.0 + exp_diff.sum(dim=1).clamp(min=0.0)).mean()
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss.
    """
    def __init__(self, window_size: int = 7):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x.mean(dim=2)
            y = y.mean(dim=2)
            
        mu_x = F.avg_pool2d(x, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)
        mu_y = F.avg_pool2d(y, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = (F.avg_pool2d(x * x, kernel_size=self.window_size, stride=1, padding=self.window_size // 2) - mu_x_sq).clamp(min=0.0)
        sigma_y_sq = (F.avg_pool2d(y * y, kernel_size=self.window_size, stride=1, padding=self.window_size // 2) - mu_y_sq).clamp(min=0.0)
        sigma_xy = F.avg_pool2d(x * y, kernel_size=self.window_size, stride=1, padding=self.window_size // 2) - mu_xy
        
        num = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2) + 1e-5
        ssim_map = (num / den).clamp(-1.0, 1.0)
                   
        return (1.0 - ssim_map.mean()).clamp(min=0.0)


class DarkActLoss(nn.Module):
    """
    Composite Multi-Dimensional Loss for DarkAct-Net:
    1. CrossEntropy classification loss (L_CE)
    2. Sobel edge consistency loss (L_Sobel-fcc)
    3. Modality-specific perception loss (L_MSP)
    4. Modality contrastive loss (L_MC)
    5. Structural & perceptual consistency loss (L_SSIM)
    """
    def __init__(
        self,
        weight_ce: float = 1.0,
        weight_sobel: float = 0.1,
        weight_msp: float = 0.1,
        weight_mc: float = 0.05,
        weight_ssim: float = 0.05
    ):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_sobel = weight_sobel
        self.weight_msp = weight_msp
        self.weight_mc = weight_mc
        self.weight_ssim = weight_ssim
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.sobel_loss = SobelEdgeLoss()
        self.mc_loss = ModalityContrastiveLoss()
        self.ssim_loss = SSIMLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        Y_r: torch.Tensor,
        Y_t: torch.Tensor,
        F_LAF: torch.Tensor,
        msp_proj_r: nn.Module = None,
        msp_proj_t: nn.Module = None,
        msp_classifier: nn.Module = None
    ) -> dict[str, torch.Tensor]:
        
        # 1. Main Classification Loss
        l_ce = self.ce_loss(logits, targets)
        
        # 2. Sobel FCC Edge Loss
        l_sobel = self.sobel_loss(Y_r, Y_t)
        
        # 3. Modality Contrastive Loss
        l_mc = self.mc_loss(Y_r, Y_t)
        
        # 4. Modality-Specific Perception Loss (MSP)
        l_msp = torch.tensor(0.0, device=logits.device)
        if msp_proj_r is not None and msp_proj_t is not None:
            recon_r = msp_proj_r(F_LAF)
            recon_t = msp_proj_t(F_LAF)
            if recon_r.shape[2:] != Y_r.shape[2:]:
                recon_r = F.interpolate(recon_r, size=Y_r.shape[2:], mode='trilinear', align_corners=False)
                recon_t = F.interpolate(recon_t, size=Y_t.shape[2:], mode='trilinear', align_corners=False)
            l_msp = F.mse_loss(recon_r, Y_r) + F.mse_loss(recon_t, Y_t)
            if msp_classifier is not None:
                aux_logits = msp_classifier(F.adaptive_avg_pool3d(F_LAF, (1, 1, 1)).view(logits.shape[0], -1))
                l_msp = l_msp + self.ce_loss(aux_logits, targets)
        
        # 5. Structural & Perceptual Loss (SSIM)
        # Target representation is (Y^r + Y^t) / 2
        Y_mean = (Y_r + Y_t) / 2.0
        F_LAF_aligned = F_LAF
        if F_LAF_aligned.shape[2:] != Y_mean.shape[2:]:
            F_LAF_aligned = F.interpolate(F_LAF_aligned, size=Y_mean.shape[2:], mode='trilinear', align_corners=False)
            
        if F_LAF_aligned.shape[1] != Y_mean.shape[1]:
            c_min = min(F_LAF_aligned.shape[1], Y_mean.shape[1])
            l_ssim = self.ssim_loss(F_LAF_aligned[:, :c_min], Y_mean[:, :c_min])
        else:
            l_ssim = self.ssim_loss(F_LAF_aligned, Y_mean)
        
        # Total Weighted Loss
        total_loss = (
            self.weight_ce * l_ce
            + self.weight_sobel * l_sobel
            + self.weight_msp * l_msp
            + self.weight_mc * l_mc
            + self.weight_ssim * l_ssim
        )
        
        return {
            "loss": total_loss,
            "loss_ce": l_ce,
            "loss_sobel": l_sobel,
            "loss_mc": l_mc,
            "loss_msp": l_msp,
            "loss_ssim": l_ssim
        }
