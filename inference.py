#!/usr/bin/env python3
"""inference.py
~~~~~~~~~~~~~~~~
Utility script to run inference with a trained DarkAct‑Net model.

Usage
-----
```bash
python inference.py \
    --rgb_path /path/to/rgb_video.mp4 \
    --thermal_path /path/to/thermal_video.mp4 \
    --ckpt_path /path/to/model_epoch_45.pth \
    [--num_frames 16] [--crop_size 224]
```
The script prints the predicted class index and the human‑readable class name.

It mirrors the preprocessing pipeline used during training:
* Uniform (or random) frame sampling – here we always use *uniform* for deterministic inference.
* Resize to ``crop_size``.
* Normalise with the same mean/std as the training code.

The model architecture is defined in ``models/darkact_net.py`` and expects a pair of tensors
``rgb`` and ``thermal`` of shape ``(3, T, H, W)`` where ``T`` is ``num_frames``.
"""

import argparse
import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helper functions – these are identical to the ones in ``datasets/darkact_dataset.py``
# ---------------------------------------------------------------------------

def read_video_frames(
    video_path: str,
    num_frames: int = 16,
    sample_mode: str = "uniform",
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Read ``num_frames`` frames from ``video_path``.

    The implementation is copied from ``datasets/darkact_dataset.read_video_frames``
    so that inference uses exactly the same sampling logic as training.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        # Return a black clip if the video cannot be read – this mirrors the fallback in training.
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)

    if sample_mode == "random":
        if total_frames > num_frames:
            intervals = np.linspace(0, total_frames - 1, num_frames + 1, dtype=int)
            indices = [
                np.random.randint(intervals[i], intervals[i + 1] - 1)
                for i in range(num_frames)
            ]
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:  # uniform sampling (default for inference)
        indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)

    frames = []
    current_idx = 0
    selected_set = set(indices)
    frame_dict = {}

    while cap.isOpened() and len(frame_dict) < len(selected_set):
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in selected_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame.shape[0], frame.shape[1]) != target_size:
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
            frame_dict[current_idx] = frame
        current_idx += 1
    cap.release()

    # Build the final clip – repeat the last valid frame when a video is shorter.
    last_frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    for idx in indices:
        if idx in frame_dict:
            last_frame = frame_dict[idx]
        frames.append(last_frame)

    return np.array(frames, dtype=np.uint8)


def transform_pair(
    rgb_frames: np.ndarray, thermal_frames: np.ndarray, is_training: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert raw ``uint8`` frame arrays to normalized ``torch`` tensors.

    The code is deliberately identical to ``DarkActDataset._transform`` so that the
    model sees exactly the same distribution during inference.
    """
    # Convert to (C, T, H, W) float tensors in [0, 1]
    rgb = torch.from_numpy(rgb_frames).permute(3, 0, 1, 2).float() / 255.0
    thermal = torch.from_numpy(thermal_frames).permute(3, 0, 1, 2).float() / 255.0

    # No random augmentations during inference – keep deterministic.
    # Normalisation (same mean/std as training)
    rgb_mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    rgb_std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    thermal_mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    thermal_std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

    rgb = (rgb - rgb_mean) / rgb_std
    thermal = (thermal - thermal_mean) / thermal_std

    return rgb, thermal

# ---------------------------------------------------------------------------
# Model loading – the DarkActNet class is defined in ``models/darkact_net.py``
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Instantiate ``DarkActNet`` and load the checkpoint.

    The checkpoint is expected to contain a ``state_dict`` saved by ``utils.misc.save_checkpoint``.
    """
    # Import here to avoid heavy imports when the script is just inspected.
    from models.darkact_net import DarkActNet

    # The backbone configuration used during training is the default ``mvitv2_b``
    # with an input channel count of 96 (output of ModalEmbedding).
    model = DarkActNet(num_classes=len(DARKACT_CLASSES))  # type: ignore[name-defined]
    model.to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    # ``ckpt`` may contain additional fields (epoch, optimizer, etc.) – we only need the weights.
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    return model

# ---------------------------------------------------------------------------
# Mapping from class index to human readable name – reuse the constant from the dataset.
# ---------------------------------------------------------------------------

DARKACT_CLASSES = [
    "call",
    "climb_stairs",
    "crouch",
    "door_close",
    "door_open",
    "drink",
    "holding_umbrella",
    "jump",
    "lift",
    "photographing",
    "pick",
    "pour",
    "push",
    "put_off_coat",
    "put_on_coat",
    "run",
    "searching",
    "sit",
    "slap",
    "sleeping",
    "squat",
    "stand",
    "turn",
    "typing",
    "using_smartphone",
    "walk",
    "wave",
]

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run DarkAct‑Net inference on a pair of RGB / Thermal videos.")
    parser.add_argument("--rgb_path", type=str, required=True, help="Path to the RGB video file.")
    parser.add_argument("--thermal_path", type=str, required=True, help="Path to the Thermal video file.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the .pth checkpoint produced by training.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample (must match training configuration).")
    parser.add_argument("--crop_size", type=int, default=224, help="Spatial size after resizing (must match training).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on.")

    args = parser.parse_args()

    if not os.path.isfile(args.rgb_path):
        sys.exit(f"[ERROR] RGB video not found: {args.rgb_path}")
    if not os.path.isfile(args.thermal_path):
        sys.exit(f"[ERROR] Thermal video not found: {args.thermal_path}")
    if not os.path.isfile(args.ckpt_path):
        sys.exit(f"[ERROR] Checkpoint not found: {args.ckpt_path}")

    device = torch.device(args.device)

    # -------------------------------------------------------------------
    # 1. Load and preprocess the two videos
    # -------------------------------------------------------------------
    rgb_frames = read_video_frames(
        args.rgb_path,
        num_frames=args.num_frames,
        sample_mode="uniform",
        target_size=(args.crop_size, args.crop_size),
    )
    thermal_frames = read_video_frames(
        args.thermal_path,
        num_frames=args.num_frames,
        sample_mode="uniform",
        target_size=(args.crop_size, args.crop_size),
    )

    rgb_tensor, thermal_tensor = transform_pair(rgb_frames, thermal_frames)
    # Add batch dimension expected by the model: (B, C, T, H, W)
    rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
    thermal_tensor = thermal_tensor.unsqueeze(0).to(device)

    # -------------------------------------------------------------------
    # 2. Load the model checkpoint
    # -------------------------------------------------------------------
    model = load_model(args.ckpt_path, device)

    # -------------------------------------------------------------------
    # 3. Forward pass – we only need the class logits
    # -------------------------------------------------------------------
    with torch.no_grad():
        logits = model(rgb_tensor, thermal_tensor)  # shape (B, num_classes)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_idx].item()

    pred_name = DARKACT_CLASSES[pred_idx] if pred_idx < len(DARKACT_CLASSES) else "UNKNOWN"

    print("--- Inference Result -----------------------------------")
    print(f"Predicted class index : {pred_idx}")
    print(f"Predicted class name  : {pred_name}")
    print(f"Confidence (softmax)  : {pred_prob:.4f}")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
