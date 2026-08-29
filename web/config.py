import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "best_model.pth")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
MAX_UPLOAD_MB = 50
ALLOWED_EXT = {".mp4"}

NUM_SEGMENTS = 8
RESIZE_SIZE = 224

CONFIG = {
    "name": "resnet50+BBox",
    "backbone": "resnet50",
    "cbam": True,
    "tcn": True,
    "trans": True,
    "bbox": True,
}
