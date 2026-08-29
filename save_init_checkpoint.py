#!/usr/bin/env python3
"""Save initialized model checkpoint for web smoke tests when GPU training is unavailable."""

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.architecture import build_model
from model.dataset import NUM_SEGMENTS, RESIZE_SIZE

CONFIG = {
    "name": "resnet50+BBox",
    "backbone": "resnet50",
    "cbam": True,
    "tcn": True,
    "trans": True,
    "bbox": True,
}


def main():
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    model = build_model(CONFIG, pretrained=False)
    path = os.path.join(ROOT, "checkpoints", "best_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": CONFIG,
        "num_segments": NUM_SEGMENTS,
        "resize_size": RESIZE_SIZE,
        "epoch": 0,
        "val_metrics": {"note": "init checkpoint — run train_best.py on GPU for trained weights"},
    }, path)
    print(f"Saved init checkpoint to {path}")

    metrics = {
        "config": CONFIG,
        "note": "Placeholder metrics — run train_best.py to populate test_metrics",
        "test_metrics": {},
    }
    metrics_path = os.path.join(ROOT, "results", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved placeholder metrics to {metrics_path}")


if __name__ == "__main__":
    main()
