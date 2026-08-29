#!/usr/bin/env python3
"""Train single resnet50+BBox TSMM model."""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")
EXAMPLE_CONFIG = os.path.join(ROOT, "configs", "default.yaml.example")
PRIVATE_CONFIG = os.path.join(ROOT, "configs", "train.yaml")

CONFIG = {
    "name": "resnet50+BBox",
    "backbone": "resnet50",
    "cbam": True,
    "tcn": True,
    "trans": True,
    "bbox": True,
}


def load_hparams(config_path: str | None = None) -> dict:
    if config_path:
        path = config_path
        using_example = os.path.abspath(path) == os.path.abspath(EXAMPLE_CONFIG)
    elif os.path.isfile(PRIVATE_CONFIG):
        path = PRIVATE_CONFIG
        using_example = False
    else:
        path = EXAMPLE_CONFIG
        using_example = True

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training config not found: {path}")

    with open(path, encoding="utf-8") as f:
        hp = yaml.safe_load(f) or {}

    required = (
        "batch_size",
        "num_epochs",
        "learning_rate",
        "weight_decay",
        "label_smooth",
        "early_stop_patience",
        "mixup_alpha",
        "use_amp",
        "ema_decay",
        "grad_clip_norm",
        "eta_min",
    )
    missing = [k for k in required if k not in hp]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    if using_example:
        print(
            "[WARN] Using placeholder configs/default.yaml.example. "
            "Copy it to configs/train.yaml and fill in your own values. "
            "This placeholder will not reproduce paper metrics."
        )
    else:
        print(f"Loaded training config from {path}")
    return hp


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def compute_metrics(y_true, y_pred, y_score):
    y_score = np.nan_to_num(np.asarray(y_score, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    y_score = np.clip(y_score, 0.0, 1.0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    pre = tp / max(1, tp + fp)
    sen = tp / max(1, tp + fn)
    spe = tn / max(1, tn + fp)
    f1 = 2 * pre * sen / max(1e-8, pre + sen)
    try:
        auc_score = float(auc(*roc_curve(y_true, y_score)[:2]))
    except ValueError:
        auc_score = float("nan")
    return {
        "Acc": round(acc * 100, 2),
        "F1": round(f1 * 100, 2),
        "Pre": round(pre * 100, 2),
        "Sen": round(sen * 100, 2),
        "Spe": round(spe * 100, 2),
        "AUC": round(auc_score * 100, 2) if not np.isnan(auc_score) else None,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, ema=None, use_bbox=True):
    model.eval()
    if ema is not None:
        ema.ema.eval()

    total_loss = 0.0
    total = 0
    y_true, y_scores = [], []

    for batch in loader:
        if use_bbox:
            x, y, b = batch
            x, y, b = x.to(device), y.to(device), b.to(device)
            net = ema.ema if ema is not None else model
            out = net(x, b)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            net = ema.ema if ema is not None else model
            out = net(x)

        loss = criterion(out.float(), y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        probs = torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy()
        y_scores.extend(probs.tolist())
        y_true.extend(y.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = (np.array(y_scores) >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, np.array(y_scores))
    metrics["loss"] = total_loss / max(1, total)
    return metrics, y_true, y_scores


def train(args):
    from model.architecture import ModelEMA, build_model
    from model.dataset import NUM_SEGMENTS, RESIZE_SIZE, load_labels, prepare_dataloaders

    hp = load_hparams(args.config)
    num_epochs = int(hp["num_epochs"])
    batch_size = int(hp["batch_size"])
    learning_rate = float(hp["learning_rate"])
    weight_decay = float(hp["weight_decay"])
    label_smooth = float(hp["label_smooth"])
    early_stop_patience = int(hp["early_stop_patience"])
    mixup_alpha = float(hp["mixup_alpha"])
    use_amp = bool(hp["use_amp"])
    ema_decay = float(hp["ema_decay"])
    grad_clip_norm = float(hp["grad_clip_norm"])
    eta_min = float(hp["eta_min"])

    debug = args.debug
    if debug:
        num_epochs = 1
        batch_size = 2
        early_stop_patience = 1

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    set_seed(42)
    cudnn.benchmark = not debug

    if not torch.cuda.is_available():
        print("[WARN] CUDA unavailable — training on CPU will be very slow.")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_labels, test_labels = load_labels()

    effective_batch = batch_size if torch.cuda.is_available() else 1
    train_loader, val_loader, test_loader, class_w = prepare_dataloaders(
        train_labels,
        test_labels,
        return_bbox=CONFIG["bbox"],
        batch_size=effective_batch,
        num_workers=4 if torch.cuda.is_available() else 0,
        debug=debug,
    )

    model = build_model(CONFIG, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth, weight=class_w)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    ema = ModelEMA(model, create_fn=lambda: build_model(CONFIG, pretrained=False), decay=ema_decay)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for batch in pbar:
            try:
                if CONFIG["bbox"]:
                    x, y, b = batch
                    x, y, b = x.to(device, non_blocking=True), y.to(device), b.to(device, non_blocking=True)
                else:
                    x, y = batch
                    x, y = x.to(device, non_blocking=True), y.to(device)
                    b = None

                optimizer.zero_grad(set_to_none=True)

                # BBox 分支不做 Mixup；loss 用 fp32 计算避免 AMP 数值问题
                with autocast(enabled=scaler.is_enabled()):
                    if CONFIG["bbox"]:
                        out = model(x, b)
                    else:
                        x_mix, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
                        out = model(x_mix)

                with autocast(enabled=False):
                    logits = out.float()
                    if CONFIG["bbox"]:
                        loss = criterion(logits, y)
                    else:
                        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

                if not torch.isfinite(loss):
                    print("[WARN] non-finite loss, skip batch")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

                bs = x.size(0)
                train_loss += loss.item() * bs
                train_correct += (logits.argmax(1) == y).sum().item()
                train_total += bs
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[WARN] OOM, skip batch")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise

        scheduler.step()

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, ema=ema, use_bbox=CONFIG["bbox"])
        val_loss = val_metrics["loss"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_total),
            "train_acc": train_correct / max(1, train_total),
            "val_loss": val_loss,
            "val_acc": val_metrics["Acc"] / 100.0,
        })

        print(
            f"Epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_Acc={val_metrics['Acc']:.2f}%"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()}
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "model_state_dict": best_state,
                "config": CONFIG,
                "num_segments": NUM_SEGMENTS,
                "resize_size": RESIZE_SIZE,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  -> saved best checkpoint (val_loss={best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        ema.ema.load_state_dict(best_state)

    test_metrics, y_true, y_scores = evaluate(
        model, test_loader, criterion, device, ema=ema, use_bbox=CONFIG["bbox"]
    )

    results = {
        "config": CONFIG,
        "num_epochs_run": len(history),
        "test_metrics": test_metrics,
        "history": history,
    }
    results_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        if k != "loss":
            print(f"  {k}: {v}%")
    print(f"Saved metrics to {results_path}")
    print(f"Best checkpoint: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")
    return test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train TSMM resnet50+BBox")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML training config. Defaults to configs/train.yaml if present, else the placeholder example.",
    )
    parser.add_argument("--debug", action="store_true", help="Quick smoke test (1 epoch, tiny data)")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet pretrained weights")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
