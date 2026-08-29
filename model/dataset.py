import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.io import read_video
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TSMM_DATA_DIR", os.path.join(_REPO_ROOT, "data"))
TRAIN_LABEL_FILE = os.path.join(DATA_DIR, "new_train_anno_file_fixed.json")
TEST_LABEL_FILE = os.path.join(DATA_DIR, "new_test_anno_file_fixed.json")
TRAIN_DIR = os.path.join(DATA_DIR, "train_fixed")
TEST_DIR = os.path.join(DATA_DIR, "test_fixed")
CACHE_DIR = os.path.join(DATA_DIR, "frame_cache_clips")

RESIZE_SIZE = 224
NUM_SEGMENTS = 8


def load_labels():
    with open(TRAIN_LABEL_FILE) as f:
        train_labels = json.load(f)
    with open(TEST_LABEL_FILE) as f:
        test_labels = json.load(f)
    return train_labels, test_labels


def get_transform(train: bool = True, size: int | None = None):
    s = size or RESIZE_SIZE
    if train:
        return Compose([
            RandomResizedCrop(s, scale=(0.8, 1.0)),
            RandomHorizontalFlip(0.5),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return Compose([
        Resize((s, s)),
        CenterCrop(s),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _filter_paths(directory: str, allowed_ids: set[str]) -> list[str]:
    paths = sorted(glob.glob(os.path.join(directory, "*.mp4")))
    return [p for p in paths if os.path.splitext(os.path.basename(p))[0] in allowed_ids]


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        paths,
        labels,
        return_bbox: bool = False,
        train_transform=None,
        val_transform=None,
        num_segments: int = NUM_SEGMENTS,
        cache_dir: str | None = CACHE_DIR,
        use_cache: bool = True,
    ):
        self.paths = paths
        self.labels = labels
        self.return_bbox = return_bbox
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.num_segments = num_segments
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def _load_clip(self, path: str):
        vid = os.path.splitext(os.path.basename(path))[0]
        cache_file = None
        if self.use_cache and self.cache_dir:
            cache_file = os.path.join(
                self.cache_dir, f"{vid}_T{self.num_segments}_S{RESIZE_SIZE}.pt"
            )
            if os.path.exists(cache_file):
                x = torch.load(cache_file, map_location="cpu", weights_only=True)
                t = x.shape[0]
                idxs = np.arange(t, dtype=int)
                h, w = x.shape[2], x.shape[3]
                return x, idxs, h, w

        video, _, _ = read_video(path, pts_unit="sec")
        t, h, w = video.shape[0], video.shape[1], video.shape[2]
        if t >= self.num_segments:
            idxs = np.linspace(0, t - 1, self.num_segments, dtype=int)
            frames = video[idxs]
        else:
            pad = self.num_segments - t
            last = video[-1:].expand(pad, -1, -1, -1)
            frames = torch.cat([video, last], dim=0)
            idxs = np.array(list(range(t)) + [t - 1] * pad, dtype=int)

        transform = self.train_transform or self.val_transform
        xs = []
        for frame in frames:
            img = Image.fromarray(frame.numpy().astype("uint8"))
            xs.append(transform(img))
        x = torch.stack(xs, 0)

        if cache_file:
            torch.save(x, cache_file)
        return x, idxs, h, w

    def __getitem__(self, idx):
        path = self.paths[idx]
        x, idxs, h, w = self._load_clip(path)
        vid = os.path.splitext(os.path.basename(path))[0]
        y = int(self.labels[vid]["is_fake"])

        if not self.return_bbox:
            return x, y

        det = self.labels[vid].get("detail", [])
        by_frame = {}
        for item in det:
            if not isinstance(item, dict):
                continue
            try:
                fi = int(item.get("frame_idx", -1))
            except (TypeError, ValueError):
                continue
            by_frame.setdefault(fi, []).extend(item.get("bbox", []))

        vals = []
        for i in idxs:
            for b in by_frame.get(int(i), []):
                if not (isinstance(b, list) and len(b) == 4):
                    continue
                x1, y1, x2, y2 = b
                if x2 <= x1 or y2 <= y1:
                    continue
                vals.append([
                    ((x1 + x2) / 2.0) / w,
                    ((y1 + y2) / 2.0) / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h,
                ])

        bbox_feats = (
            np.mean(np.asarray(vals, dtype=np.float32), axis=0)
            if vals
            else np.zeros(4, dtype=np.float32)
        )
        return x, y, torch.tensor(bbox_feats, dtype=torch.float32)


def prepare_dataloaders(
    train_labels,
    test_labels,
    return_bbox: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    debug: bool = False,
):
    train_ids = set(train_labels.keys())
    test_ids = set(test_labels.keys())
    train_paths = _filter_paths(TRAIN_DIR, train_ids)
    test_paths = _filter_paths(TEST_DIR, test_ids)

    if not train_paths:
        raise RuntimeError("No training videos matched new_train labels.")
    if not test_paths:
        raise RuntimeError("No test videos matched new_test labels.")

    if debug:
        train_paths = train_paths[:8]
        test_paths = test_paths[:4]

    split = max(1, int(0.8 * len(train_paths)))
    train_split, val_split = train_paths[:split], train_paths[split:]
    if not val_split:
        val_split = train_split[-1:]
        train_split = train_split[:-1]

    labels_all = {**train_labels, **test_labels}
    y_train = [int(train_labels[os.path.splitext(os.path.basename(p))[0]]["is_fake"]) for p in train_split]
    counts = np.bincount(y_train, minlength=2)
    cw = [1.0 / max(1, int(counts[i])) for i in range(2)]
    sw = [cw[y] for y in y_train]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_w = torch.tensor(cw, dtype=torch.float32, device=device)

    train_tf = get_transform(train=True)
    val_tf = get_transform(train=False)

    train_ds = VideoFrameDataset(train_split, labels_all, return_bbox, train_tf, None)
    val_ds = VideoFrameDataset(val_split, labels_all, return_bbox, None, val_tf)
    test_ds = VideoFrameDataset(test_paths, labels_all, return_bbox, None, val_tf)

    nw = 0 if debug else num_workers
    common = dict(batch_size=batch_size, pin_memory=torch.cuda.is_available(), num_workers=nw)
    if nw > 0:
        common.update(persistent_workers=True, prefetch_factor=2)

    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, class_w
