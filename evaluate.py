#!/usr/bin/env python3
"""Evaluate a pretrained DarkActNet checkpoint on a labeled test list.

Each non-empty line in the list contains: RGB path, thermal path, class label.
Paths may be absolute or relative to --data_root (or --data_root/videos).
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.darkact_dataset import DARKACT_CLASSES, read_video_frames
from inference import transform_pair
from models.darkact_net import DarkActNet


def resolve_video(path: str, data_root: str) -> str:
    path = Path(path).expanduser()
    candidates = [path] if path.is_absolute() else [Path(data_root) / path, Path(data_root) / "videos" / path]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Video not found: {path} (tried {tried})")


def read_test_list(test_file: str, data_root: str):
    samples = []
    with open(test_file, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{test_file}:{line_no}: expected RGB_PATH THERMAL_PATH LABEL")
            rgb, thermal, label_text = parts
            try:
                label = int(label_text)
            except ValueError:
                try:
                    label = DARKACT_CLASSES.index(label_text.lower().replace(" ", "_").replace("-", "_"))
                except ValueError as exc:
                    raise ValueError(f"{test_file}:{line_no}: unknown class label {label_text!r}") from exc
            if not 0 <= label < len(DARKACT_CLASSES):
                raise ValueError(f"{test_file}:{line_no}: label {label} outside [0, {len(DARKACT_CLASSES)-1}]")
            samples.append((resolve_video(rgb, data_root), resolve_video(thermal, data_root), label))
    if not samples:
        raise ValueError(f"No test samples found in {test_file}")
    return samples


def load_model(checkpoint_path, device, backbone, embed_dim, num_classes, no_maa, no_laf):
    model = DarkActNet(num_classes=num_classes, backbone=backbone, embed_dim=embed_dim,
                       use_maa=not no_maa, use_laf=not no_laf).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    # Accommodate checkpoints saved from DataParallel.
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def classification_metrics(y_true, y_pred, scores, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        cm[actual, predicted] += 1
    support = cm.sum(axis=1)
    predicted_count = cm.sum(axis=0)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) != 0)
    total = int(cm.sum())
    accuracy = float(tp.sum() / total)
    present = support > 0
    weighted = lambda values: float(np.average(values, weights=support))
    observed = accuracy
    expected = float(np.dot(support, predicted_count) / max(total * total, 1))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 0.0
    c = total * int(tp.sum()) - int(np.dot(support, predicted_count))
    denom = np.sqrt(max(0, (total**2 - int(np.dot(predicted_count, predicted_count))) *
                            (total**2 - int(np.dot(support, support)))))
    mcc = float(c / denom) if denom else 0.0
    top5 = float(np.mean([actual in np.argsort(row)[-min(5, n_classes):] for actual, row in zip(y_true, scores)]))
    per_class = {}
    for i, name in enumerate(DARKACT_CLASSES[:n_classes]):
        per_class[name] = {"precision": float(precision[i]), "recall": float(recall[i]),
                           "f1_score": float(f1[i]), "support": int(support[i])}
    return {
        "num_samples": total,
        "accuracy": accuracy,
        "top_5_accuracy": top5,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_precision": weighted(precision),
        "weighted_recall": weighted(recall),
        "weighted_f1": weighted(f1),
        "matthews_correlation_coefficient": mcc,
        "cohen_kappa": float(kappa),
        "per_class": per_class,
        "confusion_matrix": {"labels": DARKACT_CLASSES[:n_classes], "values": cm.tolist()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True, help="Pretrained .pth checkpoint")
    parser.add_argument("--test_file", required=True, help="Text file with RGB path, thermal path, and class index/name")
    parser.add_argument("--data_root", default=".", help="Base directory for paths in test_file")
    parser.add_argument("--output_dir", default="./evaluation", help="Directory for metrics and predictions")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", choices=["mvitv2_b", "mvitv2_s", "compact"], default="mvitv2_b")
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--num_classes", type=int, default=27)
    parser.add_argument("--no_maa", action="store_true")
    parser.add_argument("--no_laf", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.test_file):
        parser.error(f"test list not found: {args.test_file}")
    if args.num_classes > len(DARKACT_CLASSES):
        parser.error(f"num_classes cannot exceed {len(DARKACT_CLASSES)} for this class list")
    device = torch.device(args.device)
    samples = read_test_list(args.test_file, args.data_root)
    model = load_model(args.checkpoint, device, args.backbone, args.embed_dim, args.num_classes,
                       args.no_maa, args.no_laf)

    y_true, y_pred, all_scores, rows = [], [], [], []
    for start in tqdm(range(0, len(samples), args.batch_size), desc="Evaluating"):
        batch = samples[start:start + args.batch_size]
        rgb_batch, thermal_batch = [], []
        for rgb_path, thermal_path, _ in batch:
            rgb_frames = read_video_frames(rgb_path, args.num_frames, "uniform", (args.crop_size, args.crop_size))
            thermal_frames = read_video_frames(thermal_path, args.num_frames, "uniform", (args.crop_size, args.crop_size))
            rgb, thermal = transform_pair(rgb_frames, thermal_frames)
            rgb_batch.append(rgb)
            thermal_batch.append(thermal)
        with torch.inference_mode():
            logits = model(torch.stack(rgb_batch).to(device), torch.stack(thermal_batch).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for sample, prob, pred in zip(batch, probs, preds):
            rgb_path, thermal_path, label = sample
            y_true.append(label)
            y_pred.append(int(pred))
            all_scores.append(prob)
            rows.append({"rgb_path": rgb_path, "thermal_path": thermal_path,
                         "true_label": DARKACT_CLASSES[label], "true_index": label,
                         "predicted_label": DARKACT_CLASSES[int(pred)], "predicted_index": int(pred),
                         "confidence": float(prob[pred]), "correct": bool(pred == label)})

    metrics = classification_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(all_scores), args.num_classes)
    output_dir = args.output_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        fallback_dir = os.path.join(os.getcwd(), "evaluation")
        print(f"Cannot write to {output_dir!r}; saving results to {fallback_dir!r} instead.")
        output_dir = fallback_dir
        os.makedirs(output_dir, exist_ok=True)
    metrics["checkpoint"] = os.path.abspath(args.checkpoint)
    metrics["test_file"] = os.path.abspath(args.test_file)
    metrics["class_names"] = DARKACT_CLASSES[:args.num_classes]
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}, indent=2))
    print(f"\nFull report: {os.path.join(output_dir, 'metrics.json')}\nPredictions: {os.path.join(output_dir, 'predictions.csv')}")


if __name__ == "__main__":
    main()
