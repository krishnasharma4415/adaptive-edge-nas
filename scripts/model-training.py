"""
=============================================================================
 Phase 3 — Baseline Model Training
 Project  : Hardware-Aware NAS for Edge Devices
 Dataset  : Tiny-ImageNet-200
=============================================================================

Trains three standard lightweight reference models on Tiny-ImageNet-200:
  1. MobileNetV2
  2. ShuffleNetV2 (×1.0)
  3. EfficientNet-B0

Metrics recorded per model:
  - Top-1 / Top-5 accuracy (val)
  - FLOPs (via thop)
  - Parameter count
  - Inference latency  (median of 100 forward passes @ batch=1 on GPU)
  - Peak GPU memory
  - Model size on disk (MB)

Training config  (tuned for RTX 4060 8 GB)
  - batch_size       = 128
  - epochs           = 50  (early stopping patience = 8)
  - optimizer        = AdamW  (lr=1e-3, weight_decay=1e-4)
  - scheduler        = CosineAnnealingLR
  - label smoothing  = 0.1
  - Mixup alpha      = 0.2

Outputs (saved to  models/  and  results/)
  - models/{model_name}_best.pth
  - results/baseline_{name}_metrics.json
  - results/baseline_training_curves.png
  - results/baseline_comparison.png
=============================================================================
"""

import os
import json
import time
import random
import copy
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as tvm

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
# Training hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    batch_size      = 128,
    epochs          = 50,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    label_smoothing = 0.10,
    mixup_alpha     = 0.20,
    patience        = 8,
    num_workers     = 4,
    pin_memory      = True,
)

MODELS_TO_TRAIN = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]

# ─────────────────────────────────────────────────────────────────────────────
# Dry-run mode — set True to test the pipeline with minimal compute,
# then flip to False for full training.
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN        = True   # ← change to False for full training
DRY_EPOCHS     = 2      # epochs when dry-running
DRY_BATCHES    = 10     # train batches per epoch when dry-running
DRY_VAL_BATCHES = 5     # val batches per epoch when dry-running

# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str) -> nn.Module:
    """Instantiate a lightweight model adapted for Tiny-ImageNet-200 (64×64)."""
    if name == "mobilenetv2":
        model = tvm.mobilenet_v2(weights=None, num_classes=NUM_CLASSES)
        model.features[0][0].stride = (1, 1)
    elif name == "shufflenetv2":
        model = tvm.shufflenet_v2_x1_0(weights=None, num_classes=NUM_CLASSES)
        model.conv1[0].stride = (1, 1)
    elif name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=None, num_classes=NUM_CLASSES)
        model.features[0][0].stride = (1, 1)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(DEVICE)



# ─────────────────────────────────────────────────────────────────────────────
# Hardware profiling
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency(model: nn.Module, n_runs: int = 100) -> float:
    """Median inference latency (ms) at batch=1."""
    model.eval()
    dummy = torch.randn(1, 3, 64, 64, device=DEVICE)
    latencies = []
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.median(latencies))


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct1, correct5, total = 0., 0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        if DRY_RUN and batch_idx >= DRY_VAL_BATCHES:
            break
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        correct1   += (out.argmax(dim=1) == labels).sum().item()
        _, top5     = out.topk(5, dim=1)
        correct5   += sum(labels[i].item() in top5[i].tolist() for i in range(labels.size(0)))
    return total_loss / total, correct1 / total, correct5 / total


def train_model(name: str) -> dict:
    """Full training loop for one model. Returns metrics dict."""
    print(f"\n{'='*60}\n  Training : {name.upper()}  |  Device: {DEVICE}\n{'='*60}")

    model    = build_model(name)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    try:
        from thop import profile as thop_profile
        # Profile on a copy — thop registers total_ops/total_params buffers in-place,
        # which would corrupt the model's state_dict and break evaluation later.
        _m_copy = copy.deepcopy(model)
        flops, _ = thop_profile(_m_copy, inputs=(torch.randn(1, 3, 64, 64, device=DEVICE),), verbose=False)
        flops = float(flops)
        del _m_copy
    except ImportError:
        flops = -1.0

    print(f"  Parameters : {n_params/1e6:.2f} M" +
          (f"  |  FLOPs : {flops/1e6:.1f} M" if flops > 0 else ""))

    train_loader, val_loader = get_dataloaders(
        batch_size=CFG["batch_size"], num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"])

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    scaler    = GradScaler()

    best_acc1  = 0.
    patience_c = 0
    history    = {"train_loss": [], "val_loss": [], "val_acc1": [], "val_acc5": [], "lr": []}

    n_epochs = DRY_EPOCHS if DRY_RUN else CFG["epochs"]
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0, running_loss, total = time.time(), 0., 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if DRY_RUN and batch_idx >= DRY_BATCHES:
                break
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            lam    = np.random.beta(CFG["mixup_alpha"], CFG["mixup_alpha"]) if CFG["mixup_alpha"] > 0 else 1.0
            idx_m  = torch.randperm(imgs.size(0), device=imgs.device)
            imgs_m = lam * imgs + (1 - lam) * imgs[idx_m]
            y_a, y_b = labels, labels[idx_m]
            optimizer.zero_grad()
            with autocast():
                out  = model(imgs_m)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            total        += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / total
        val_loss, val_acc1, val_acc5 = eval_epoch(model, val_loader, criterion)
        lr_now = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc1"].append(val_acc1)
        history["val_acc5"].append(val_acc5)
        history["lr"].append(lr_now)

        print(f"  Epoch [{epoch:02d}/{CFG['epochs']}]  "
              f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
              f"Acc@1={val_acc1*100:.2f}%  Acc@5={val_acc5*100:.2f}%  "
              f"lr={lr_now:.5f}  t={time.time()-t0:.1f}s")

        if val_acc1 > best_acc1:
            best_acc1  = val_acc1
            patience_c = 0
            ckpt_path  = MODELS_DIR / f"{name}_best.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc1": val_acc1, "val_acc5": val_acc5}, ckpt_path)
        else:
            patience_c += 1
            if patience_c >= CFG["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    torch.cuda.reset_peak_memory_stats(DEVICE)
    latency_ms = measure_latency(model)
    peak_mem   = torch.cuda.max_memory_allocated(DEVICE) / 1024**2 if DEVICE.type == "cuda" else -1.

    ckpt_path = MODELS_DIR / f"{name}_best.pth"
    model_mb  = ckpt_path.stat().st_size / 1024**2 if ckpt_path.exists() else -1

    metrics = {
        "model":         name,
        "params_M":      round(n_params / 1e6, 3),
        "flops_M":       round(flops / 1e6, 1) if flops > 0 else -1,
        "best_val_acc1": round(best_acc1 * 100, 2),
        "best_val_acc5": round(max(history["val_acc5"]) * 100, 2),
        "latency_ms":    round(latency_ms, 2),
        "peak_mem_MB":   round(peak_mem, 1),
        "model_size_MB": round(model_mb, 1),
        "epochs_run":    len(history["train_loss"]),
        "history":       history,
    }

    out_json = RESULTS_DIR / f"baseline_{name}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓  Metrics → {out_json}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ─────────────────────────────────────────────────────────────────────────────

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU  : {torch.cuda.get_device_name(0)}"
              f"  |  VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    all_metrics = []
    for model_name in MODELS_TO_TRAIN:
        all_metrics.append(train_model(model_name))

    print("\n" + "─"*80)
    print(f"  {'Model':<20} {'Acc@1':>8} {'Acc@5':>8} {'Params(M)':>10} "
          f"{'FLOPs(M)':>10} {'Lat(ms)':>9} {'Size(MB)':>9}")
    print("─"*80)
    for m in all_metrics:
        print(f"  {m['model']:<20} {m['best_val_acc1']:>8.2f} {m['best_val_acc5']:>8.2f} "
              f"{m['params_M']:>10.2f} {m['flops_M']:>10.1f} "
              f"{m['latency_ms']:>9.2f} {m['model_size_MB']:>9.1f}")
    print("─"*80)

    # ── Training curves ──────────────────────────────────────────────────
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title, ax_label in [
        (axes[0], "val_acc1",   "Validation Accuracy (Top-1)", "Accuracy"),
        (axes[1], "val_loss",   "Validation Loss",             "Loss"),
        (axes[2], "train_loss", "Training Loss",               "Loss"),
    ]:
        for m, color in zip(all_metrics, colors):
            h = m["history"][key]
            ax.plot(range(1, len(h)+1), h, color=color, lw=2, label=m["model"])
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ax_label, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "baseline_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓  Training curves → {out}")

    # ── Comparison bar chart ─────────────────────────────────────────────
    names = [m["model"] for m in all_metrics]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, (key, ylabel) in zip(axes, [
        ("best_val_acc1", "Top-1 Accuracy (%)"),
        ("latency_ms",    "Latency (ms, batch=1)"),
        ("params_M",      "# Parameters (M)"),
        ("model_size_MB", "Model Size (MB)"),
    ]):
        vals = [m[key] for m in all_metrics]
        bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.suptitle("Baseline Model Comparison — Tiny-ImageNet-200",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "baseline_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓  Comparison chart → {out}")
