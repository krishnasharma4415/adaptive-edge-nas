"""
=============================================================================
 nas.py — NAS Utilities (standalone model + fine-tune)
 Project  : Hardware-Aware NAS for Edge Devices
 Dataset  : Tiny-ImageNet-200

 Provides:
   - StandaloneNASModel: extract & instantiate a found architecture
   - Fine-tune loop: train the best arch from scratch
=============================================================================
"""

import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent   # project root (one level above scripts/)
DATASET_DIR = BASE_DIR / "tiny-imagenet-200"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_DIR  = DATASET_DIR / "train"
VAL_DIR    = DATASET_DIR / "val"
VAL_ANNOT  = VAL_DIR / "val_annotations.txt"
WNIDS_FILE = DATASET_DIR / "wnids.txt"
STATS_FILE = RESULTS_DIR / "dataset_stats.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Normalization stats
# ─────────────────────────────────────────────────────────────────────────────
if STATS_FILE.exists():
    with open(STATS_FILE) as _f:
        _s = json.load(_f)
    MEAN = tuple(_s["rgb_mean"])
    STD  = tuple(_s["rgb_std"])
else:
    MEAN = (0.4802, 0.4481, 0.3975)
    STD  = (0.2770, 0.2691, 0.2821)
    warnings.warn("dataset_stats.json not found — using published reference values.")

# ─────────────────────────────────────────────────────────────────────────────
# Class map
# ─────────────────────────────────────────────────────────────────────────────
with open(WNIDS_FILE) as _f:
    _wnids = [l.strip() for l in _f if l.strip()]
CLASS_MAP   = {wnid: idx for idx, wnid in enumerate(sorted(_wnids))}
NUM_CLASSES = len(CLASS_MAP)

# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(64, padding=8, padding_mode="reflect"),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
    T.Normalize(mean=MEAN, std=STD),
])

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
    def __init__(self):
        self.samples: list[tuple[Path, int]] = []
        for wnid, label in CLASS_MAP.items():
            img_dir = TRAIN_DIR / wnid / "images"
            if img_dir.exists():
                for p in img_dir.glob("*.JPEG"):
                    self.samples.append((p, label))
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return TRAIN_TRANSFORM(Image.open(path).convert("RGB")), label


class TinyImageNetVal(Dataset):
    def __init__(self):
        self.samples: list[tuple[Path, int]] = []
        img_dir = VAL_DIR / "images"
        with open(VAL_ANNOT) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[1] in CLASS_MAP:
                    self.samples.append((img_dir / parts[0], CLASS_MAP[parts[1]]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return VAL_TRANSFORM(Image.open(path).convert("RGB")), label


def get_dataloaders(batch_size=128, num_workers=None, pin_memory=True):
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)
    train_loader = DataLoader(
        TinyImageNetTrain(), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None, drop_last=True,
    )
    val_loader = DataLoader(
        TinyImageNetVal(), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Search-space primitives (duplicated from hardware-aware.py)
# ─────────────────────────────────────────────────────────────────────────────

OP_NAMES = [
    "identity", "dwconv3x3", "dwconv5x5",
    "mbconv3x3", "mbconv5x5", "shuffle_block", "se_block",
]
NUM_OPS = len(OP_NAMES)

CELL_CONFIG = [
    (32,  1), (32,  1), (64,  2),
    (64,  1), (64,  1), (128, 2),
    (128, 1), (128, 1), (128, 1),
    (128, 1), (192, 2), (192, 1),
    (192, 1), (192, 1), (256, 2),
    (256, 1), (256, 1), (256, 1),
    (256, 1), (256, 1),
]
NUM_CELLS = len(CELL_CONFIG)
STEM_CH   = 32


def _make_divisible(v, divisor=8):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class DepthwiseSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
                      padding=pad, groups=C_in, bias=False),
            nn.BatchNorm2d(C_in), nn.ReLU6(inplace=True),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out), nn.ReLU6(inplace=True),
        )

    def forward(self, x): return self.op(x)


class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, expand_ratio=3):
        super().__init__()
        mid = _make_divisible(C_in * expand_ratio)
        pad = kernel_size // 2
        self.use_res = (stride == 1) and (C_in == C_out)
        self.op = nn.Sequential(
            nn.Conv2d(C_in, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, kernel_size, stride=stride,
                      padding=pad, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, C_out, 1, bias=False), nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        out = self.op(x)
        return x + out if self.use_res else out


class ShuffleBlock(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        branch = C_out // 2
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(C_in, C_in, 3, stride=stride, padding=1, groups=C_in, bias=False),
                nn.BatchNorm2d(C_in),
                nn.Conv2d(C_in, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(C_in, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
                nn.Conv2d(branch, branch, 3, stride=stride, padding=1, groups=branch, bias=False),
                nn.BatchNorm2d(branch),
                nn.Conv2d(branch, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()
            half = C_in // 2
            self.branch2 = nn.Sequential(
                nn.Conv2d(half, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
                nn.Conv2d(branch, branch, 3, padding=1, groups=branch, bias=False),
                nn.BatchNorm2d(branch),
                nn.Conv2d(branch, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True),
            )
        self.stride = stride

    def channel_shuffle(self, x, groups=2):
        B, C, H, W = x.shape
        return x.view(B, groups, C // groups, H, W).transpose(1, 2).contiguous().view(B, C, H, W)

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        return self.channel_shuffle(out)


class SEBlock(nn.Module):
    def __init__(self, C_in, C_out, stride=1, reduction=4):
        super().__init__()
        mid = max(8, C_in // reduction)
        self.use_res = (stride == 1) and (C_in == C_out)
        self.bn      = nn.BatchNorm2d(C_out)
        self.conv_dw = nn.Conv2d(C_in, C_in, 3, stride=stride, padding=1, groups=C_in, bias=False)
        self.conv_pw = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.se      = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C_out, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, C_out), nn.Sigmoid(),
        )

    def forward(self, x):
        out = F.relu6(self.conv_dw(x))
        out = self.conv_pw(out)
        se  = self.se(out).view(out.size(0), out.size(1), 1, 1)
        out = out * se
        return (out + x) if self.use_res else self.bn(out)


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        self.adapt = nn.Identity() if (C_in == C_out and stride == 1) else \
                     nn.Sequential(nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False),
                                   nn.BatchNorm2d(C_out))

    def forward(self, x): return self.adapt(x)


def build_op(op_name: str, C_in: int, C_out: int, stride: int = 1) -> nn.Module:
    if op_name == "identity":        return Identity(C_in, C_out, stride)
    elif op_name == "dwconv3x3":     return DepthwiseSepConv(C_in, C_out, 3, stride)
    elif op_name == "dwconv5x5":     return DepthwiseSepConv(C_in, C_out, 5, stride)
    elif op_name == "mbconv3x3":     return MBConv(C_in, C_out, 3, stride)
    elif op_name == "mbconv5x5":     return MBConv(C_in, C_out, 5, stride)
    elif op_name == "shuffle_block": return ShuffleBlock(C_in, C_out, stride)
    elif op_name == "se_block":      return SEBlock(C_in, C_out, stride)
    else: raise ValueError(f"Unknown op: {op_name}")


# ═════════════════════════════════════════════════════════════════════════════
# StandaloneNASModel — fixed architecture (no op switching overhead)
# ═════════════════════════════════════════════════════════════════════════════

class StandaloneNASModel(nn.Module):
    """
    Instantiate the best-found architecture as a single static net.
    Much leaner than SuperNet (no parallel ops).

    Args
      arch       : list of op indices (length = NUM_CELLS)
      num_classes: number of output classes
    """

    def __init__(self, arch: list[int], num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, STEM_CH, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(STEM_CH),
            nn.ReLU6(inplace=True),
        )
        self.cells = nn.Sequential()
        C_in = STEM_CH
        for cell_idx, (op_idx, (C_out, stride)) in enumerate(zip(arch, CELL_CONFIG)):
            op = build_op(OP_NAMES[op_idx], C_in, C_out, stride)
            self.cells.add_module(f"cell_{cell_idx:02d}", op)
            C_in = C_out

        C_final = CELL_CONFIG[-1][0]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(C_final, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cells(self.stem(x)))


# ═════════════════════════════════════════════════════════════════════════════
# Fine-tune config
# ═════════════════════════════════════════════════════════════════════════════

FT_CFG = dict(
    epochs       = 10,
    batch_size   = 64,
    lr           = 1e-3,
    weight_decay = 1e-4,
    label_smooth = 0.10,
    mixup_alpha  = 0.20,
    patience     = 12,
    num_workers  = 4,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dry-run mode — set True to test the pipeline with minimal compute,
# then flip to False for full training.
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN         = True   # ← change to False for full training
DRY_EPOCHS      = 2      # fine-tune epochs when dry-running
DRY_BATCHES     = 10     # train batches per epoch when dry-running
DRY_VAL_BATCHES = 5      # val batches per epoch when dry-running


# ═════════════════════════════════════════════════════════════════════════════
# Script entry-point
# ═════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    arch_path = RESULTS_DIR / "best_arch.json"
    if not arch_path.exists():
        print("[nas.py] best_arch.json not found. Run hardware-aware.py first.")
        raise SystemExit(1)

    with open(arch_path) as f:
        best = json.load(f)
    arch = best["arch"]

    print(f"Fine-tuning best NAS architecture from scratch")
    print(f"Arch ops: {[OP_NAMES[i] for i in arch]}")

    model    = StandaloneNASModel(arch).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.2f} M")

    train_loader, val_loader = get_dataloaders(
        batch_size=FT_CFG["batch_size"], num_workers=FT_CFG["num_workers"], pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=FT_CFG["label_smooth"])
    optimizer = optim.AdamW(model.parameters(),
                             lr=FT_CFG["lr"], weight_decay=FT_CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FT_CFG["epochs"], eta_min=1e-6)
    scaler    = GradScaler()

    best_acc1  = 0.
    patience_c = 0

    n_epochs = DRY_EPOCHS if DRY_RUN else FT_CFG["epochs"]
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0, run_loss, total = time.time(), 0., 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if DRY_RUN and batch_idx >= DRY_BATCHES:
                break
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            alpha = FT_CFG["mixup_alpha"]
            lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.
            idx   = torch.randperm(imgs.size(0), device=imgs.device)
            imgs_m = lam * imgs + (1 - lam) * imgs[idx]
            y_a, y_b = labels, labels[idx]

            optimizer.zero_grad()
            with autocast():
                out  = model(imgs_m)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * imgs.size(0)
            total    += imgs.size(0)

        scheduler.step()

        model.eval()
        correct, n_val = 0, 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader):
                if DRY_RUN and batch_idx >= DRY_VAL_BATCHES:
                    break
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    out = model(imgs)
                correct += (out.argmax(1) == labels).sum().item()
                n_val   += labels.size(0)
        val_acc1 = correct / n_val

        print(f"  Epoch [{epoch:03d}/{FT_CFG['epochs']}]  "
              f"Loss={run_loss/total:.4f}  ValAcc={val_acc1*100:.2f}%  "
              f"t={time.time()-t0:.1f}s")

        if val_acc1 > best_acc1:
            best_acc1  = val_acc1
            patience_c = 0
            ckpt_path  = MODELS_DIR / "nas_best_finetuned.pth"
            torch.save({
                "arch":        arch,
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc1":    val_acc1,
            }, ckpt_path)
        else:
            patience_c += 1
            if patience_c >= FT_CFG["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nBest val accuracy : {best_acc1*100:.2f}%")
    print(f"Saved → {MODELS_DIR / 'nas_best_finetuned.pth'}")
