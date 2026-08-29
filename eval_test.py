#!/usr/bin/env python3
"""Evaluate best_model.pth on new_test set (Table I metrics format)."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.architecture import build_model
from model.dataset import load_labels, prepare_dataloaders

CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "best_model.pth")
RESULTS_PATH = os.path.join(ROOT, "results", "metrics.json")


def compute_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    pre = tp / max(1, tp + fp)
    sen = tp / max(1, tp + fn)
    spe = tn / max(1, tn + fp)
    f1 = 2 * pre * sen / max(1e-8, pre + sen)
    auc_score = float(auc(*roc_curve(y_true, y_score)[:2]))
    return {
        "Acc": round(acc * 100, 2),
        "F1": round(f1 * 100, 2),
        "Pre": round(pre * 100, 2),
        "Sen": round(sen * 100, 2),
        "Spe": round(spe * 100, 2),
        "AUC": round(auc_score * 100, 2),
    }


@torch.no_grad()
def evaluate(checkpoint_path: str = CHECKPOINT_PATH):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(config, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    train_labels, test_labels = load_labels()
    _, _, test_loader, _ = prepare_dataloaders(
        train_labels, test_labels,
        return_bbox=config.get("bbox", True),
        batch_size=16 if torch.cuda.is_available() else 1,
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    y_true, y_scores = [], []
    for batch in tqdm(test_loader, desc="Evaluating"):
        x, y, b = batch
        x, b = x.to(device), b.to(device)
        out = model(x, b)
        probs = torch.softmax(out, 1)[:, 1].cpu().numpy()
        y_scores.extend(probs.tolist())
        y_true.extend(y.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = (np.array(y_scores) >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, np.array(y_scores))

    results = {
        "config": config,
        "checkpoint": checkpoint_path,
        "test_metrics": metrics,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("=== new_test metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}%")
    print(f"Saved to {RESULTS_PATH}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    evaluate(parser.parse_args().checkpoint)
