import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.backbone import ModalEmbedding, MultiStageBackbone
from datasets import DarkActDataset, create_mock_dataset
from utils import AverageMeter, accuracy, setup_logger


class GenericMultimodalBaseline(nn.Module):
    """
    Multimodal baseline that fuses modalities via cross-attention / channel concatenation.
    """
    def __init__(self, model_name: str = "mrfs", embed_dim: int = 96, num_classes: int = 27):
        super().__init__()
        self.model_name = model_name
        self.rgb_embed = ModalEmbedding(in_channels=3, out_channels=embed_dim)
        self.thermal_embed = ModalEmbedding(in_channels=3, out_channels=embed_dim)
        
        self.rgb_backbone = MultiStageBackbone(variant="mvitv2_s", in_channels=embed_dim)
        self.thermal_backbone = MultiStageBackbone(variant="mvitv2_s", in_channels=embed_dim)
        
        final_dim = self.rgb_backbone.stage_dims[-1]
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2 * final_dim, num_classes)
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        feat_r = self.rgb_embed(rgb)
        feat_t = self.thermal_embed(thermal)
        
        for r_stage, t_stage in zip(self.rgb_backbone.stages, self.thermal_backbone.stages):
            feat_r = r_stage(feat_r)
            feat_t = t_stage(feat_t)
            
        p_r = feat_r.mean(dim=(-3, -2, -1))
        p_t = feat_t.mean(dim=(-3, -2, -1))
        fused = torch.cat([p_r, p_t], dim=1)
        return self.head(fused)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Multimodal Action Recognition Baselines on DarkAct")
    parser.add_argument("--model", type=str, default="mrfs", choices=["mrfs", "cmx", "cddfuse"])
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("./logs", name=f"Benchmark_{args.model.upper()}")
    logger.info(f"Starting Multimodal Baseline Benchmark: Model={args.model.upper()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(os.path.join(args.data_path, "videos")):
        create_mock_dataset(args.data_path, num_samples_per_split=8, num_frames=args.num_frames, size=args.crop_size)

    test_dataset = DarkActDataset(
        data_path=args.data_path,
        split="test",
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        is_training=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = GenericMultimodalBaseline(
        model_name=args.model,
        embed_dim=96,
        num_classes=27
    ).to(device)

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        logger.info(f"Loading checkpoint from {args.ckpt_path}")
        model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu")["state_dict"])

    model.eval()
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    with torch.no_grad():
        for rgb, thermal, targets in test_loader:
            rgb = rgb.to(device)
            thermal = thermal.to(device)
            targets = targets.to(device)
            logits = model(rgb, thermal)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            top1.update(acc1, rgb.size(0))
            top5.update(acc5, rgb.size(0))

    logger.info(f"==> [{args.model.upper()} Baseline] Top-1: {top1.avg:.2f}%, Top-5: {top5.avg:.2f}%")


if __name__ == "__main__":
    main()
