import os
import sys
import shutil
import tempfile
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import (
    MotionAwareAttention,
    LightAdaptiveFusion,
    ModalEmbedding,
    MultiStageBackbone,
    DarkActNet,
    DarkActLoss,
    SobelEdgeLoss,
    ModalityContrastiveLoss,
    SSIMLoss
)
from datasets import DarkActDataset, create_mock_dataset, DARKACT_CLASSES
from utils import AverageMeter, accuracy, set_seed


def test_maa():
    print("Testing MotionAwareAttention (MAA)...")
    maa = MotionAwareAttention(in_channels=48)
    G = torch.randn(2, 48, 8, 32, 32)
    Y = maa(G)
    assert Y.shape == G.shape, f"MAA shape mismatch: {Y.shape} vs {G.shape}"
    loss = Y.sum()
    loss.backward()
    print("  [PASSED] MAA forward and backward.")


def test_laf():
    print("Testing LightAdaptiveFusion (LAF)...")
    laf = LightAdaptiveFusion(channels=64, num_heads=4)
    F_r = torch.randn(2, 64, 4, 16, 16)
    F_t = torch.randn(2, 64, 4, 16, 16)
    Y_r = torch.randn(2, 64, 4, 16, 16)
    Y_t = torch.randn(2, 64, 4, 16, 16)
    
    out_r, out_t, F_LAF = laf(F_r, F_t, Y_r, Y_t)
    assert out_r.shape == F_r.shape, f"LAF out_r shape mismatch: {out_r.shape}"
    assert out_t.shape == F_t.shape, f"LAF out_t shape mismatch: {out_t.shape}"
    assert F_LAF.shape == F_r.shape, f"LAF F_LAF shape mismatch: {F_LAF.shape}"
    loss = (out_r + out_t + F_LAF).sum()
    loss.backward()
    print("  [PASSED] LAF forward and backward.")


def test_losses():
    print("Testing DarkAct Multi-dimensional Losses...")
    criterion = DarkActLoss()
    logits = torch.randn(4, 27, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    Y_r = torch.randn(4, 32, 4, 16, 16, requires_grad=True)
    Y_t = torch.randn(4, 32, 4, 16, 16, requires_grad=True)
    F_LAF = torch.randn(4, 64, 4, 16, 16, requires_grad=True)
    
    losses = criterion(logits, targets, Y_r, Y_t, F_LAF)
    assert "loss" in losses and losses["loss"].item() > 0
    losses["loss"].backward()
    print("  [PASSED] DarkActLoss computed and backpropagated.")


def test_ablations():
    print("Testing DarkAct-Net Ablations (Table 4 in paper)...")
    B, T, H, W = 2, 4, 64, 64
    rgb = torch.randn(B, 3, T, H, W)
    thermal = torch.randn(B, 3, T, H, W)
    
    configs = [
        ("Full DarkAct-Net", True, True),
        ("w/o MAA", False, True),
        ("w/o LAF", True, False),
        ("w/o MAA + LAF", False, False),
    ]
    
    for name, use_maa, use_laf in configs:
        model = DarkActNet(
            num_classes=27,
            backbone="compact",
            embed_dim=32,
            use_maa=use_maa,
            use_laf=use_laf
        )
        out = model(rgb, thermal)
        assert out.shape == (B, 27), f"{name} output shape mismatch: {out.shape}"
        print(f"  [PASSED] {name} passed.")


def test_dataset_and_training():
    print("Testing Mock Dataset Creation and Mini Training Loop...")
    tmp_dir = os.path.abspath("./tests_tmp_data")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        create_mock_dataset(tmp_dir, num_samples_per_split=4, num_frames=4, size=64)
        dataset = DarkActDataset(tmp_dir, split="train", num_frames=4, crop_size=64)
        assert len(dataset) == 4, f"Dataset length mismatch: {len(dataset)}"
        
        rgb, thermal, label = dataset[0]
        assert rgb.shape == (3, 4, 64, 64), f"RGB shape mismatch: {rgb.shape}"
        assert thermal.shape == (3, 4, 64, 64), f"Thermal shape mismatch: {thermal.shape}"
        
        # Test 1 mini training step
        model = DarkActNet(num_classes=27, backbone="compact", embed_dim=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = DarkActLoss()
        
        rgb_b = rgb.unsqueeze(0)
        thermal_b = thermal.unsqueeze(0)
        target_b = torch.tensor([label])
        
        out = model(rgb_b, thermal_b, return_features=True)
        loss_dict = criterion(
            logits=out["logits"],
            targets=target_b,
            Y_r=out["Y_r"],
            Y_t=out["Y_t"],
            F_LAF=out["F_LAF"],
            msp_proj_r=model.msp_proj_r,
            msp_proj_t=model.msp_proj_t,
            msp_classifier=model.msp_classifier
        )
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()
        
        print("  [PASSED] Mini training step completed successfully.")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    set_seed(42)
    test_maa()
    test_laf()
    test_losses()
    test_ablations()
    test_dataset_and_training()
    print("\nAll DarkAct-Net pipeline tests passed successfully!")
