"""
=============================================================================
 Phase 2 — Data Preprocessing Pipeline
 Project  : Hardware-Aware NAS for Edge Devices
 Dataset  : Tiny-ImageNet-200
=============================================================================

2.1  Dataset Normalization
      Computes (or loads from eda.py's results) dataset mean / std.

2.2  Data Augmentation Strategy
      Training  : RandomCrop(64, padding=8) | RandomHorizontalFlip |
                  ColorJitter | RandomErasing | Normalize
      Validation: CenterCrop(56) → Resize(64) | Normalize

2.3  Efficient DataLoader
      - multi-worker loading  (num_workers = os.cpu_count() - 1)
      - pin_memory = True     (faster CPU→GPU transfer)
      - persistent_workers    (avoid fork overhead across epochs)
      - prefetch_factor = 2   (double-buffered prefetch)

Outputs (saved to  results/)
  - augmentation_preview.png  : grid of original vs augmented samples
  - dataloader_benchmark.json : loader throughput (imgs/sec)

Usage
------
  python data-processing.py

Or import in other modules:
  from data_processing import get_dataloaders, MEAN, STD
=============================================================================
"""

import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent   # project root (one level above scripts/)
DATASET_DIR = BASE_DIR / "tiny-imagenet-200"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STATS_FILE  = RESULTS_DIR / "dataset_stats.json"
TRAIN_DIR   = DATASET_DIR / "train"
VAL_DIR     = DATASET_DIR / "val"
VAL_ANNOT   = VAL_DIR / "val_annotations.txt"
WNIDS_FILE  = DATASET_DIR / "wnids.txt"

# ─────────────────────────────────────────────────────────────────────────────
# Load mean / std  (from Phase 1 EDA – fallback to published values)
# ─────────────────────────────────────────────────────────────────────────────
if STATS_FILE.exists():
    with open(STATS_FILE) as f:
        _s = json.load(f)
    MEAN = tuple(_s["rgb_mean"])
    STD  = tuple(_s["rgb_std"])
else:
    MEAN = (0.4802, 0.4481, 0.3975)
    STD  = (0.2770, 0.2691, 0.2821)
    warnings.warn("dataset_stats.json not found — using published reference values. Run eda.py first!")

# ─────────────────────────────────────────────────────────────────────────────
# Load class index mapping  {wnid: class_index}
# ─────────────────────────────────────────────────────────────────────────────
with open(WNIDS_FILE) as f:
    _wnids = [l.strip() for l in f if l.strip()]
CLASS_MAP   = {wnid: idx for idx, wnid in enumerate(sorted(_wnids))}
NUM_CLASSES = len(CLASS_MAP)

# ─────────────────────────────────────────────────────────────────────────────
# 2.2  Transforms
# ─────────────────────────────────────────────────────────────────────────────

# Training augmentation pipeline:
#   1. RandomCrop 64×64 (pad 8px on each side)
#   2. RandomHorizontalFlip (p=0.5)
#   3. ColorJitter (brightness / contrast / sat / hue)
#   4. RandomGrayscale (p=0.05) – rare colour drop
#   5. ToTensor
#   6. RandomErasing (p=0.2, scale 2%-10%)  — acts on tensor
#   7. Normalize
TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(64, padding=8, padding_mode="reflect"),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
    T.Normalize(mean=MEAN, std=STD),
])

# Validation / test transform:
#   1. CenterCrop 56×56 (mild crop to remove border artefacts)
#   2. Resize back to 64×64
#   3. ToTensor
#   4. Normalize
VAL_TRANSFORM = T.Compose([
    T.CenterCrop(56),
    T.Resize(64, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class TinyImageNetTrain(Dataset):
    """
    Loads Tiny-ImageNet training split.

    Directory layout expected:
      tiny-imagenet-200/train/{wnid}/images/{wnid}_{idx}.JPEG
    """

    def __init__(self, root: Path = TRAIN_DIR, transform=None):
        self.root      = root
        self.transform = transform or TRAIN_TRANSFORM
        self.samples: list[tuple[Path, int]] = []

        for wnid, label in CLASS_MAP.items():
            img_dir = root / wnid / "images"
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob("*.JPEG"):
                self.samples.append((img_path, label))

        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class TinyImageNetVal(Dataset):
    """
    Loads Tiny-ImageNet validation split.

    Reads labels from  val/val_annotations.txt.
    """

    def __init__(self, root: Path = VAL_DIR, transform=None):
        self.root      = root
        self.transform = transform or VAL_TRANSFORM
        self.samples: list[tuple[Path, int]] = []

        img_dir = root / "images"
        with open(VAL_ANNOT) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                fname, wnid = parts[0], parts[1]
                if wnid in CLASS_MAP:
                    self.samples.append((img_dir / fname, CLASS_MAP[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Efficient DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    batch_size:  int  = 128,
    num_workers: int  = None,
    pin_memory:  bool = True,
    train_transform=None,
    val_transform=None,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) ready for GPU training.

    Args
    ----
    batch_size   : images per mini-batch (128 fits comfortably in 8 GB VRAM)
    num_workers  : parallel data-loading workers (default = cpu_count - 1)
    pin_memory   : lock CPU tensors in page-locked memory → faster GPU upload
    train/val_transform : override default augmentation

    Returns
    -------
    train_loader, val_loader
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    train_ds = TinyImageNetTrain(transform=train_transform)
    val_ds   = TinyImageNetVal(transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size       = batch_size,
        shuffle          = True,
        num_workers      = num_workers,
        pin_memory       = pin_memory,
        persistent_workers = (num_workers > 0),
        prefetch_factor  = 2 if num_workers > 0 else None,
        drop_last        = True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size       = batch_size,
        shuffle          = False,
        num_workers      = num_workers,
        pin_memory       = pin_memory,
        persistent_workers = (num_workers > 0),
        prefetch_factor  = 2 if num_workers > 0 else None,
        drop_last        = False,
    )

    return train_loader, val_loader





# ─────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")


# Augmentation preview
train_ds_raw = TinyImageNetTrain(transform=T.Compose([T.ToTensor()]))
aug_tf       = TRAIN_TRANSFORM
n_samples    = 6
indices      = random.sample(range(len(train_ds_raw)), n_samples)
n_cols       = 4

fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols*3, n_samples*3),
                          gridspec_kw={"hspace": 0.1, "wspace": 0.05})
for col_idx, title in enumerate(["Original", "Aug ①", "Aug ②", "Aug ③"]):
    axes[0, col_idx].set_title(title, fontsize=12, pad=8, fontweight="bold")

mean_t = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
std_t  = torch.tensor(STD,  dtype=torch.float32).view(3, 1, 1)

for row, idx in enumerate(indices):
    path, _ = train_ds_raw.samples[idx]
    pil_img = Image.open(path).convert("RGB")
    raw_tensor = T.ToTensor()(pil_img)
    axes[row, 0].imshow((raw_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    axes[row, 0].axis("off")
    for col_idx in range(1, 4):
        aug_tensor = aug_tf(pil_img)
        img_denorm = (aug_tensor * std_t + mean_t).clamp(0, 1)
        img_final = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        axes[row, col_idx].imshow(img_final)
        axes[row, col_idx].axis("off")

fig.suptitle("Data Augmentation Preview — Training Pipeline",
             fontsize=14, fontweight="bold", y=1.01)
out = RESULTS_DIR / "augmentation_preview.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓  Augmentation preview → {out}")

# DataLoader benchmark
num_workers = max(1, (os.cpu_count() or 4) - 1)
batch_size  = 128
n_batches   = 20
train_loader, _ = get_dataloaders(batch_size=batch_size)
loader_iter     = iter(train_loader)
_ = next(loader_iter)   # warm-up

t0 = time.perf_counter()
n_imgs = 0
for _ in range(min(n_batches, len(train_loader))):
    imgs, _ = next(loader_iter)
    n_imgs += imgs.shape[0]
elapsed    = time.perf_counter() - t0
throughput = n_imgs / elapsed

result = {
    "batch_size":        batch_size,
    "num_workers":       num_workers,
    "n_batches":         n_batches,
    "total_images":      n_imgs,
    "elapsed_sec":       round(elapsed, 3),
    "throughput_img_s":  round(throughput, 1),
}
out = RESULTS_DIR / "dataloader_benchmark.json"
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"✓  Loader throughput : {throughput:,.0f} imgs/sec  → {out}")
