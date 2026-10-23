import os
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import DarkActNet, DarkActLoss
from datasets import DarkActDataset, create_mock_dataset
from utils import AverageMeter, accuracy, set_seed, setup_logger, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="DarkAct-Net: Multimodal Low-Light Action Recognition")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "demo"],
                        help="Operating mode: train, test, or demo")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to DarkAct dataset directory")
    parser.add_argument("--backbone", type=str, default="mvitv2_b",
                        choices=["mvitv2_b", "mvitv2_s", "compact"],
                        help="Transformer backbone variant")
    parser.add_argument("--embed_dim", type=int, default=96,
                        help="Modal embedding feature channel dimension")
    parser.add_argument("--num_classes", type=int, default=27,
                        help="Number of action classes")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of sampled frames per video")
    parser.add_argument("--crop_size", type=int, default=224,
                        help="Spatial frame crop/resize size")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=120,
                        help="Total training epochs (20 warmup + 100 cosine annealing)")
    parser.add_argument("--warmup_epochs", type=int, default=20,
                        help="Warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-4,
                        help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--weight_decay", type=float, default=5e-3,
                        help="AdamW weight decay")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader worker processes")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/best_model.pth",
                        help="Checkpoint path for evaluation")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save logs and checkpoints")
    parser.add_argument("--no_maa", action="store_true",
                        help="Ablation: Disable Motion-Aware Attention (MAA)")
    parser.add_argument("--no_laf", action="store_true",
                        help="Ablation: Disable Light-Adaptive Fusion (LAF)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--create_mock_data", action="store_true",
                        help="Generate a mock synthetic dataset if real data is missing")
    return parser.parse_args()


def get_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from min_lr / base_lr to 1.0
            return (epoch + 1) / max(1, warmup_epochs)
        else:
            # Cosine decay from base_lr to min_lr
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed = (min_lr + (base_lr - min_lr) * cosine_decay) / base_lr
            return decayed
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: DarkActLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger
):
    model.train()
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    end = time.time()
    for step, (rgb, thermal, targets) in enumerate(dataloader):
        rgb = rgb.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(rgb, thermal, return_features=True)
        raw_model = model.module if hasattr(model, "module") else model

        loss_dict = criterion(
            logits=out["logits"],
            targets=targets,
            Y_r=out["Y_r"],
            Y_t=out["Y_t"],
            F_LAF=out["F_LAF"],
            msp_proj_r=raw_model.msp_proj_r,
            msp_proj_t=raw_model.msp_proj_t,
            msp_classifier=raw_model.msp_classifier
        )
        total_loss = loss_dict["loss"]
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"Warning: NaN/Inf detected in loss at step {step + 1}. Skipping step.")
            optimizer.zero_grad()
            continue
            
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc1, acc5 = accuracy(out["logits"], targets, topk=(1, 5))
        loss_meter.update(total_loss.item(), rgb.size(0))
        top1.update(acc1, rgb.size(0))
        top5.update(acc5, rgb.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % max(1, len(dataloader) // 5) == 0 or step == len(dataloader) - 1:
            logger.info(
                f"Epoch: [{epoch + 1}][{step + 1}/{len(dataloader)}] "
                f"Loss: {loss_meter.avg:.4f} (CE: {loss_dict['loss_ce'].item():.3f}, "
                f"Sobel: {loss_dict['loss_sobel'].item():.3f}, "
                f"MC: {loss_dict['loss_mc'].item():.3f}) "
                f"Top-1: {top1.avg:.2f}% Top-5: {top5.avg:.2f}%"
            )

    return loss_meter.avg, top1.avg, top5.avg


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    logger
):
    # Single-GPU eval: DataParallel fails on a final batch smaller than the GPU count.
    model = model.module if isinstance(model, nn.DataParallel) else model
    model.eval()
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    with torch.no_grad():
        for rgb, thermal, targets in dataloader:
            rgb = rgb.to(device, non_blocking=True)
            thermal = thermal.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(rgb, thermal, return_features=False)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

            top1.update(acc1, rgb.size(0))
            top5.update(acc5, rgb.size(0))

    logger.info(f"==> Test Results: Top-1 Acc: {top1.avg:.2f}%, Top-5 Acc: {top5.avg:.2f}%")
    return top1.avg, top5.avg


def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, name="DarkActNet")
    logger.info("Initializing DarkAct-Net Action Recognition Framework...")
    logger.info(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Operating on device: {device}")

    # Verify or generate dataset
    if args.create_mock_data or not os.path.exists(os.path.join(args.data_path, "videos")):
        logger.warning(f"Data not found at {args.data_path}. Generating mock dataset for testing...")
        create_mock_dataset(args.data_path, num_samples_per_split=16, num_frames=args.num_frames, size=args.crop_size)

    # Instantiate datasets
    train_dataset = DarkActDataset(
        data_path=args.data_path,
        split="train",
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        is_training=True
    )
    test_dataset = DarkActDataset(
        data_path=args.data_path,
        split="test",
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        is_training=False
    )

    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    # Initialize DarkAct-Net Model
    model = DarkActNet(
        num_classes=args.num_classes,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        use_maa=not args.no_maa,
        use_laf=not args.no_laf
    ).to(device)

    # Multi-GPU support if available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel execution.")
        model = nn.DataParallel(model)

    if args.mode == "test":
        if os.path.exists(args.ckpt_path):
            logger.info(f"Loading checkpoint from {args.ckpt_path}")
            load_checkpoint(args.ckpt_path, model)
        else:
            logger.warning(f"Checkpoint {args.ckpt_path} not found! Testing with randomly initialized model.")
        evaluate(model, test_loader, device, logger)
        return

    # Loss and Optimizer
    criterion = DarkActLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr
    )

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    best_acc = 0.0
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        raw_model = model.module if hasattr(model, "module") else model
        start_epoch = load_checkpoint(args.resume, raw_model, optimizer)
        logger.info(f"Resuming from epoch {start_epoch}")

    logger.info("Starting training loop...")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger
        )
        scheduler.step()

        # Evaluate periodically
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_acc1, val_acc5 = evaluate(model, test_loader, device, logger)
            is_best = val_acc1 > best_acc
            if is_best:
                best_acc = val_acc1
            
            save_checkpoint(
                state={
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                checkpoint_dir=checkpoint_dir,
                filename=f"epoch_{epoch + 1}.pth"
            )

    logger.info(f"Training completed. Best Top-1 Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
