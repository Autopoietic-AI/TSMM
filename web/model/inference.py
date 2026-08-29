import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision.io import read_video

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.architecture import build_model
from model.dataset import get_transform
from web.config import CHECKPOINT_PATH, CONFIG, NUM_SEGMENTS, RESIZE_SIZE

_device = None
_model = None
_transform = None
_ckpt_meta = {}


def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _device


def load_model(checkpoint_path: str | None = None):
    global _model, _transform, _ckpt_meta
    path = checkpoint_path or CHECKPOINT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", CONFIG)
    _ckpt_meta = {
        "num_segments": ckpt.get("num_segments", NUM_SEGMENTS),
        "resize_size": ckpt.get("resize_size", RESIZE_SIZE),
    }

    device = _get_device()
    model = build_model(config, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    _model = model
    _transform = get_transform(train=False, size=_ckpt_meta["resize_size"])
    return model, _ckpt_meta["num_segments"], _ckpt_meta["resize_size"]


def _prepare_clip(video_path: str, num_segments: int):
    video, _, _ = read_video(video_path, pts_unit="sec")
    t, h, w = video.shape[0], video.shape[1], video.shape[2]
    if t >= num_segments:
        idxs = np.linspace(0, t - 1, num_segments, dtype=int)
        frames = video[idxs]
    else:
        pad = num_segments - t
        last = video[-1:].expand(pad, -1, -1, -1)
        frames = torch.cat([video, last], dim=0)

    xs = []
    for frame in frames:
        img = Image.fromarray(frame.numpy().astype("uint8"))
        xs.append(_transform(img))
    return torch.stack(xs, 0).unsqueeze(0)


def predict_video(video_path: str, checkpoint_path: str | None = None) -> dict:
    global _model, _transform, _ckpt_meta
    if _model is None:
        load_model(checkpoint_path)

    num_segments = _ckpt_meta.get("num_segments", NUM_SEGMENTS)

    device = _get_device()
    t0 = time.time()

    clip = _prepare_clip(video_path, num_segments).to(device)
    bbox = torch.zeros(1, 4, device=device)

    with torch.no_grad():
        logits = _model(clip, bbox)
        probs = torch.softmax(logits, dim=1)[0]
        fake_prob = float(probs[1].item())
        real_prob = float(probs[0].item())

    elapsed = time.time() - t0
    is_fake = fake_prob >= 0.5
    confidence = fake_prob if is_fake else real_prob

    return {
        "is_fake": is_fake,
        "label": "伪造" if is_fake else "真实",
        "fake_prob": round(fake_prob * 100, 2),
        "real_prob": round(real_prob * 100, 2),
        "confidence": round(confidence * 100, 2),
        "num_segments": num_segments,
        "inference_time_sec": round(elapsed, 3),
    }
