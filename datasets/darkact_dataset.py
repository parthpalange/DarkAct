import os
import random
import glob
from typing import List, Tuple, Dict, Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


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
    "wave"
]

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(DARKACT_CLASSES)}


def read_video_frames(
    video_path: str,
    num_frames: int = 16,
    sample_mode: str = "uniform",
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Reads video frames using OpenCV.
    Returns:
        frames: np.ndarray of shape (T, H, W, 3), dtype uint8.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        # Fallback for empty or unreadable video
        cap.release()
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)
        
    if sample_mode == "random":
        if total_frames > num_frames:
            intervals = np.linspace(0, total_frames - 1, num_frames + 1, dtype=int)
            indices = [random.randint(intervals[i], intervals[i+1] - 1) for i in range(num_frames)]
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else: # uniform
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
    
    # Construct sequence
    last_frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    for idx in indices:
        if idx in frame_dict:
            last_frame = frame_dict[idx]
        frames.append(last_frame)
        
    return np.array(frames, dtype=np.uint8)


class DarkActDataset(Dataset):
    """
    Dataset loader for paired RGB-Thermal videos in DarkAct.
    Directory structure expected:
    data_path/
      annotations/
        train.txt (or test.txt) with lines: <rgb_rel_path> <thermal_rel_path> <class_idx>
      videos/
        RGB/<class_name>/<video_id>.mp4
        Thermal/<class_name>/<video_id>.mp4
    """
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        num_frames: int = 16,
        crop_size: int = 224,
        is_training: bool = True
    ):
        self.data_path = data_path
        self.split = split
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.is_training = is_training
        
        self.samples = self._load_samples()

    def _normalize_name(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    def _load_samples(self) -> List[Dict]:
        samples = []
        ann_file = os.path.join(self.data_path, "annotations", f"{self.split}.txt")
        
        if os.path.exists(ann_file):
            with open(ann_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Handle both paths relative to data_path and paths relative to data_path/videos
                        p0 = parts[0].lstrip("/").lstrip("\\")
                        p1 = parts[1].lstrip("/").lstrip("\\")
                        if p0.startswith("videos/") or p0.startswith("videos\\"):
                            rgb_path = os.path.join(self.data_path, p0)
                            thermal_path = os.path.join(self.data_path, p1)
                        else:
                            rgb_path = os.path.join(self.data_path, "videos", p0)
                            thermal_path = os.path.join(self.data_path, "videos", p1)

                        if not os.path.isfile(rgb_path) or not os.path.isfile(thermal_path):
                            # Try direct relative to data_path
                            direct_rgb = os.path.join(self.data_path, p0)
                            direct_th = os.path.join(self.data_path, p1)
                            if os.path.isfile(direct_rgb) and os.path.isfile(direct_th):
                                rgb_path = direct_rgb
                                thermal_path = direct_th
                            else:
                                raise FileNotFoundError(f"Video listed in {ann_file} not found: {rgb_path} or {thermal_path}")
                        label = int(parts[2])
                        samples.append({"rgb": rgb_path, "thermal": thermal_path, "label": label})
        else:
            # Automatic discovery by walking videos/RGB/ and matching videos/Thermal/
            rgb_base = os.path.join(self.data_path, "videos", "RGB")
            thermal_base = os.path.join(self.data_path, "videos", "Thermal")
            
            if os.path.exists(rgb_base):
                for cls_dir in sorted(os.listdir(rgb_base)):
                    normalized_cls = self._normalize_name(cls_dir)
                    label = CLASS_TO_IDX.get(normalized_cls, 0)
                    
                    rgb_cls_path = os.path.join(rgb_base, cls_dir)
                    if not os.path.isdir(rgb_cls_path):
                        continue
                        
                    for video_name in sorted(os.listdir(rgb_cls_path)):
                        if video_name.endswith(('.mp4', '.avi', '.mkv')):
                            rgb_video = os.path.join(rgb_cls_path, video_name)
                            # Look for matching thermal video
                            thermal_video = os.path.join(thermal_base, cls_dir, video_name)
                            if not os.path.exists(thermal_video):
                                # Check alternative casing/extension
                                candidates = glob.glob(os.path.join(thermal_base, "*", video_name))
                                thermal_video = candidates[0] if candidates else rgb_video
                            
                            samples.append({
                                "rgb": rgb_video,
                                "thermal": thermal_video,
                                "label": label
                            })
                            
                # Split pseudo train/test if using folder discovery
                random.seed(42)
                random.shuffle(samples)
                split_idx = int(0.8 * len(samples))
                if self.split == "train":
                    samples = samples[:split_idx]
                else:
                    samples = samples[split_idx:]
                    
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _transform(self, rgb_frames: np.ndarray, thermal_frames: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synchronized spatial-temporal augmentations.
        """
        # Convert uint8 (T, H, W, 3) to float tensor (3, T, H, W) in [0, 1]
        rgb = torch.from_numpy(rgb_frames).permute(3, 0, 1, 2).float() / 255.0
        thermal = torch.from_numpy(thermal_frames).permute(3, 0, 1, 2).float() / 255.0
        
        # Synchronized random horizontal flip during training
        if self.is_training and random.random() > 0.5:
            rgb = torch.flip(rgb, dims=[-1])
            thermal = torch.flip(thermal, dims=[-1])
            
        # Standard Normalization: (x - mean) / std
        rgb_mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        rgb_std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        thermal_mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        thermal_std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        
        rgb = (rgb - rgb_mean) / rgb_std
        thermal = (thermal - thermal_mean) / thermal_std
        
        return rgb, thermal

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        item = self.samples[idx]
        sample_mode = "random" if self.is_training else "uniform"
        
        rgb_frames = read_video_frames(
            item["rgb"],
            num_frames=self.num_frames,
            sample_mode=sample_mode,
            target_size=(self.crop_size, self.crop_size)
        )
        thermal_frames = read_video_frames(
            item["thermal"],
            num_frames=self.num_frames,
            sample_mode=sample_mode,
            target_size=(self.crop_size, self.crop_size)
        )
        
        rgb_tensor, thermal_tensor = self._transform(rgb_frames, thermal_frames)
        return rgb_tensor, thermal_tensor, item["label"]


def create_mock_dataset(
    root_dir: str,
    num_samples_per_split: int = 16,
    num_frames: int = 16,
    size: int = 112
) -> None:
    """
    Creates a synthetic mini DarkAct dataset directory with valid video files
    for local smoke testing, unit verification, and CI pipelines.
    """
    os.makedirs(os.path.join(root_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "videos", "RGB"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "videos", "Thermal"), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for split in ["train", "test"]:
        ann_lines = []
        for i in range(num_samples_per_split):
            cls_name = DARKACT_CLASSES[i % len(DARKACT_CLASSES)]
            cls_idx = CLASS_TO_IDX[cls_name]
            
            rgb_cls_dir = os.path.join(root_dir, "videos", "RGB", cls_name)
            thermal_cls_dir = os.path.join(root_dir, "videos", "Thermal", cls_name)
            os.makedirs(rgb_cls_dir, exist_ok=True)
            os.makedirs(thermal_cls_dir, exist_ok=True)
            
            v_name = f"sample_{split}_{i:03d}.mp4"
            rgb_vpath = os.path.join(rgb_cls_dir, v_name)
            thermal_vpath = os.path.join(thermal_cls_dir, v_name)
            
            # Write synthetic video
            out_rgb = cv2.VideoWriter(rgb_vpath, fourcc, 15, (size, size))
            out_thermal = cv2.VideoWriter(thermal_vpath, fourcc, 15, (size, size))
            
            for f in range(num_frames):
                # Low-light RGB frame (dark with small noisy motion)
                rgb_img = np.random.randint(0, 50, (size, size, 3), dtype=np.uint8)
                cv2.circle(rgb_img, (20 + (f * 4) % 60, 40), 10, (120, 110, 100), -1)
                
                # Thermal frame (bright human signature in dark background)
                thermal_img = np.random.randint(10, 40, (size, size, 3), dtype=np.uint8)
                cv2.circle(thermal_img, (20 + (f * 4) % 60, 40), 12, (220, 220, 240), -1)
                
                out_rgb.write(rgb_img)
                out_thermal.write(thermal_img)
                
            out_rgb.release()
            out_thermal.release()
            
            rel_rgb = os.path.relpath(rgb_vpath, root_dir)
            rel_thermal = os.path.relpath(thermal_vpath, root_dir)
            ann_lines.append(f"{rel_rgb} {rel_thermal} {cls_idx}\n")
            
        with open(os.path.join(root_dir, "annotations", f"{split}.txt"), "w") as f:
            f.writelines(ann_lines)
            
    print(f"Mock dataset generated at {root_dir} successfully.")
