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


class UnimodalClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 96, backbone: str = "mvitv2_b", num_classes: int = 27):
        super().__init__()
        self.embed = ModalEmbedding(in_channels=in_channels, out_channels=embed_dim)
        self.backbone = MultiStageBackbone(variant=backbone, in_channels=embed_dim)
        final_dim = self.backbone.stage_dims[-1]
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(final_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.embed(x)
        for stage in self.backbone.stages:
            feat = stage(feat)
        pooled = feat.mean(dim=(-3, -2, -1))
        return self.head(pooled)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Unimodal Action Recognition on DarkAct")
    parser.add_argument("--modality", type=str, default="rgb", choices=["rgb", "thermal"])
    parser.add_argument("--backbone", type=str, default="mvitv2_b")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("./logs", name=f"Benchmark_{args.modality.upper()}")
    logger.info(f"Starting Unimodal Benchmark: Modality={args.modality}, Backbone={args.backbone}")
    
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

    model = UnimodalClassifier(
        in_channels=3,
        embed_dim=96,
        backbone=args.backbone,
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
            x = rgb if args.modality == "rgb" else thermal
            x = x.to(device)
            targets = targets.to(device)
            logits = model(x)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            top1.update(acc1, x.size(0))
            top5.update(acc5, x.size(0))

    logger.info(f"==> [{args.modality.upper()}-{args.backbone}] Top-1: {top1.avg:.2f}%, Top-5: {top5.avg:.2f}%")


if __name__ == "__main__":
    main()
