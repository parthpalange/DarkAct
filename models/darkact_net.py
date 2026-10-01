import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .maa import MotionAwareAttention
from .laf import LightAdaptiveFusion
from .backbone import ModalEmbedding, TransformerStage, MultiStageBackbone


class DarkActNet(nn.Module):
    """
    DarkAct-Net: An RGB-Thermal Fusion Framework for Multimodal Low-Light Action Recognition
    (CVPR 2026).
    
    Architecture:
    1. Modal Embedding Branches for RGB and Thermal streams.
    2. Motion-Aware Attention (MAA) modules to highlight motion-salient regions.
    3. Multi-Stage Video Transformer Backbone (MViTv2-based).
    4. Light-Adaptive Fusion (LAF) modules inserted after each Transformer stage.
    5. Final Concatenation & Linear Classification Head for 27 action categories.
    """
    def __init__(
        self,
        num_classes: int = 27,
        backbone: str = "mvitv2_b",
        embed_dim: int = 96,
        use_maa: bool = True,
        use_laf: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_maa = use_maa
        self.use_laf = use_laf
        self.embed_dim = embed_dim
        
        # 1. Modal Embedding branches for RGB and Thermal
        self.rgb_embed = ModalEmbedding(in_channels=3, out_channels=embed_dim)
        self.thermal_embed = ModalEmbedding(in_channels=3, out_channels=embed_dim)
        
        # 2. Motion-Aware Attention (MAA) modules
        if self.use_maa:
            self.maa_rgb = MotionAwareAttention(in_channels=embed_dim)
            self.maa_thermal = MotionAwareAttention(in_channels=embed_dim)
            
        # 3. Multi-Stage Transformer Backbones (RGB and Thermal branches)
        self.rgb_backbone = MultiStageBackbone(variant=backbone, in_channels=embed_dim)
        self.thermal_backbone = MultiStageBackbone(variant=backbone, in_channels=embed_dim)
        
        stage_dims = self.rgb_backbone.stage_dims
        self.num_stages = len(stage_dims)
        
        # 4. Light-Adaptive Fusion (LAF) modules after each stage
        if self.use_laf:
            self.laf_modules = nn.ModuleList([
                LightAdaptiveFusion(channels=dim, num_heads=4, y_channels=embed_dim)
                for dim in stage_dims
            ])
            
        # 5. Modality-Specific Perception (MSP) projection heads for auxiliary loss
        final_dim = stage_dims[-1]
        self.msp_proj_r = nn.Sequential(
            nn.Conv3d(final_dim, embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim),
            nn.PReLU()
        )
        self.msp_proj_t = nn.Sequential(
            nn.Conv3d(final_dim, embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim),
            nn.PReLU()
        )
        self.msp_classifier = nn.Linear(final_dim, num_classes)
        
        # 6. Final Classifier Head
        # Output features of each modality are pooled and concatenated: [F_r; F_t] -> 2 * final_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * final_dim, num_classes)
        )

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Args:
            rgb: RGB video tensor (B, 3, T, H, W)
            thermal: Thermal video tensor (B, 3, T, H, W) (or (B, 1, T, H, W) expanded)
            return_features: If True, returns dict of intermediate representations for losses
        Returns:
            logits or dict containing logits, Y_r, Y_t, F_LAF
        """
        if thermal.shape[1] == 1:
            thermal = thermal.repeat(1, 3, 1, 1, 1)
            
        # Step 1: Modal Embedding
        G_r = self.rgb_embed(rgb)        # (B, C, T, H', W')
        G_t = self.thermal_embed(thermal)  # (B, C, T, H', W')
        
        # Step 2: Motion-Aware Attention (MAA)
        if self.use_maa:
            Y_r = self.maa_rgb(G_r)
            Y_t = self.maa_thermal(G_t)
        else:
            Y_r = G_r
            Y_t = G_t
            
        # Step 3: Multi-Stage Transformer with Light-Adaptive Fusion (LAF)
        F_r = Y_r
        F_t = Y_t
        last_F_LAF = None
        
        for i in range(self.num_stages):
            # Pass through stage Transformer
            F_r = self.rgb_backbone.stages[i](F_r)
            F_t = self.thermal_backbone.stages[i](F_t)
            
            # Apply Light-Adaptive Fusion (LAF)
            if self.use_laf:
                F_r, F_t, F_LAF = self.laf_modules[i](F_r, F_t, Y_r, Y_t)
                last_F_LAF = F_LAF
            else:
                last_F_LAF = (F_r + F_t) / 2.0
                
        # Step 4: Final Classification
        # Spatiotemporal global pooling: (B, C, T, H, W) -> (B, C)
        feat_r = F.adaptive_avg_pool3d(F_r, (1, 1, 1)).view(F_r.shape[0], -1)
        feat_t = F.adaptive_avg_pool3d(F_t, (1, 1, 1)).view(F_t.shape[0], -1)
        
        # Concatenate RGB and Thermal representations
        fused_rep = torch.cat([feat_r, feat_t], dim=1)  # (B, 2 * C_final)
        logits = self.classifier(fused_rep)              # (B, num_classes)
        
        if return_features:
            return {
                "logits": logits,
                "Y_r": Y_r,
                "Y_t": Y_t,
                "F_r": F_r,
                "F_t": F_t,
                "F_LAF": last_F_LAF
            }
            
        return logits
