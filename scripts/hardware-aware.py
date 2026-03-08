"""
=============================================================================
 Phases 4–6 — Hardware-Aware NAS (Supernet + Evolution Search)
 Project  : Hardware-Aware NAS for Edge Devices
 Dataset  : Tiny-ImageNet-200
=============================================================================

Phase 4  — Define NAS Search Space
  Cell-based design inspired by NAS-Bench-201 / FBNet.
  Network: Input → Stem → [N cells] → Classifier

  Candidate operations per cell:
    0: Identity (skip)
    1: 3×3 Depthwise Separable Conv
    2: 5×5 Depthwise Separable Conv
    3: 3×3 MBConv (expand_ratio=3)
    4: 5×5 MBConv (expand_ratio=3)
    5: Shuffle Block (like ShuffleNetV2)
    6: SE Block (channel attention)

  Search space size: 7^20 ≈ 79 trillion architectures

Phase 5  — Hardware-Aware NAS Algorithm
  5.1 Latency Lookup Table
  5.2 Multi-Objective Loss: L = CE_loss + λ * max(0, latency - latency_budget)
  5.3 Supernet Training (One-Shot)
  5.4 Two-Stage Search (HURRICANE-inspired)

Phase 6  — Evolutionary Architecture Search & Selection

Outputs
  - results/latency_lut.json
  - models/supernet_final.pth
  - results/search_history.json
  - results/pareto_front.png
  - results/best_arch.json
=============================================================================
"""

import os
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────────────────────
# Paths & seeds
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
# Dry-run mode — set True to test the pipeline with minimal compute,
# then flip to False for full training.
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN             = True   # ← change to False for full training
DRY_SUPERNET_EPOCHS = 2      # supernet epochs when dry-running
DRY_BATCHES         = 10     # train batches per epoch when dry-running
DRY_VAL_BATCHES     = 5      # val batches per epoch when dry-running
DRY_GENERATIONS     = 2      # evo search generations when dry-running
DRY_POPULATION      = 6      # evo population size when dry-running

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


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4 — Search Space Primitive Operations
# ═════════════════════════════════════════════════════════════════════════════

OP_NAMES = [
    "identity",       # 0
    "dwconv3x3",      # 1
    "dwconv5x5",      # 2
    "mbconv3x3",      # 3
    "mbconv5x5",      # 4
    "shuffle_block",  # 5
    "se_block",       # 6
]
NUM_OPS = len(OP_NAMES)


class DepthwiseSepConv(nn.Module):
    """Depthwise + Pointwise convolution (MobileNet-style)."""
    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
                      padding=pad, groups=C_in, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU6(inplace=True),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x): return self.op(x)


class MBConv(nn.Module):
    """Inverted residual block (MobileNetV2 style), expand_ratio=3."""
    def __init__(self, C_in, C_out, kernel_size, stride=1, expand_ratio=3):
        super().__init__()
        # make divisible by 8
        v = C_in * expand_ratio
        mid = max(8, int(v + 8 / 2) // 8 * 8)
        pad  = kernel_size // 2
        self.use_res = (stride == 1) and (C_in == C_out)
        self.op = nn.Sequential(
            nn.Conv2d(C_in, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, kernel_size, stride=stride,
                      padding=pad, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        out = self.op(x)
        return x + out if self.use_res else out


class ShuffleBlock(nn.Module):
    """ShuffleNetV2-style channel split + shuffle block."""
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        branch = C_out // 2
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(C_in, C_in, 3, stride=stride, padding=1,
                          groups=C_in, bias=False),
                nn.BatchNorm2d(C_in),
                nn.Conv2d(C_in, branch, 1, bias=False),
                nn.BatchNorm2d(branch),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(C_in, branch, 1, bias=False),
                nn.BatchNorm2d(branch),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch, branch, 3, stride=stride, padding=1,
                          groups=branch, bias=False),
                nn.BatchNorm2d(branch),
                nn.Conv2d(branch, branch, 1, bias=False),
                nn.BatchNorm2d(branch),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()
            half = C_in // 2
            self.branch2 = nn.Sequential(
                nn.Conv2d(half, branch, 1, bias=False),
                nn.BatchNorm2d(branch),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch, branch, 3, padding=1, groups=branch, bias=False),
                nn.BatchNorm2d(branch),
                nn.Conv2d(branch, branch, 1, bias=False),
                nn.BatchNorm2d(branch),
                nn.ReLU(inplace=True),
            )
        self.stride = stride

    def channel_shuffle(self, x, groups=2):
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W).transpose(1, 2)
        return x.contiguous().view(B, C, H, W)

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        return self.channel_shuffle(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, C_in, C_out, stride=1, reduction=4):
        super().__init__()
        mid = max(8, C_in // reduction)
        self.use_res = (stride == 1) and (C_in == C_out)
        self.proj    = nn.Identity() if self.use_res else \
                       nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False)
        self.bn      = nn.BatchNorm2d(C_out)
        self.conv_dw = nn.Conv2d(C_in, C_in, 3, stride=stride,
                                 padding=1, groups=C_in, bias=False)
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
    if op_name == "identity":      return Identity(C_in, C_out, stride)
    elif op_name == "dwconv3x3":   return DepthwiseSepConv(C_in, C_out, 3, stride)
    elif op_name == "dwconv5x5":   return DepthwiseSepConv(C_in, C_out, 5, stride)
    elif op_name == "mbconv3x3":   return MBConv(C_in, C_out, 3, stride)
    elif op_name == "mbconv5x5":   return MBConv(C_in, C_out, 5, stride)
    elif op_name == "shuffle_block": return ShuffleBlock(C_in, C_out, stride)
    elif op_name == "se_block":    return SEBlock(C_in, C_out, stride)
    else: raise ValueError(f"Unknown op: {op_name}")


# ═════════════════════════════════════════════════════════════════════════════
# Macro architecture: Stem → 20 cells → Classifier
# ═════════════════════════════════════════════════════════════════════════════

CELL_CONFIG = [
    (32,  1), (32,  1), (64,  2),
    (64,  1), (64,  1), (128, 2),
    (128, 1), (128, 1), (128, 1),
    (128, 1), (192, 2), (192, 1),
    (192, 1), (192, 1), (256, 2),
    (256, 1), (256, 1), (256, 1),
    (256, 1), (256, 1),
]
NUM_CELLS = len(CELL_CONFIG)   # 20
STEM_CH   = 32


class SuperNet(nn.Module):
    """
    One-Shot Supernet: each cell has ALL ops in parallel.
    During forward, `arch` (list of op indices, length=NUM_CELLS)
    selects one path per cell.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, STEM_CH, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(STEM_CH),
            nn.ReLU6(inplace=True),
        )
        self.cells = nn.ModuleList()
        C_in = STEM_CH
        for C_out, stride in CELL_CONFIG:
            self.cells.append(nn.ModuleList([
                build_op(name, C_in, C_out, stride) for name in OP_NAMES
            ]))
            C_in = C_out

        C_final = CELL_CONFIG[-1][0]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p=0.2), nn.Linear(C_final, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor, arch: list[int]) -> torch.Tensor:
        x = self.stem(x)
        for cell_ops, op_idx in zip(self.cells, arch):
            x = cell_ops[op_idx](x)
        return self.head(x)

    def random_arch(self) -> list[int]:
        return [random.randint(0, NUM_OPS - 1) for _ in range(NUM_CELLS)]


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5.1 — Latency Lookup Table
# ═════════════════════════════════════════════════════════════════════════════

def build_latency_lut(n_runs: int = 50) -> dict:
    """Measure median latency (ms) for each op at each cell configuration."""
    print("[LUT] Building latency lookup table …")
    lut  = {}
    C_in = STEM_CH

    for cell_idx, (C_out, stride) in enumerate(CELL_CONFIG):
        lut[cell_idx] = {}
        spatial = 64
        for i in range(cell_idx):
            if CELL_CONFIG[i][1] == 2:
                spatial //= 2
        dummy = torch.randn(1, C_in, max(1, spatial), max(1, spatial), device=DEVICE)

        for op_idx, op_name in enumerate(OP_NAMES):
            op = build_op(op_name, C_in, C_out, stride).to(DEVICE).eval()
            latencies = []
            with torch.no_grad():
                for _ in range(10):
                    _ = op(dummy)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    _ = op(dummy)
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - t0) * 1000)
            lut[cell_idx][op_idx] = round(float(np.median(latencies)), 4)
            del op

        C_in = C_out
        if cell_idx % 5 == 0:
            print(f"  Cell {cell_idx:2d}/{NUM_CELLS-1} done")

    out = RESULTS_DIR / "latency_lut.json"
    with open(out, "w") as f:
        json.dump(lut, f, indent=2)
    print(f"✓  LUT saved → {out}")
    return lut


def predict_latency(arch: list[int], lut: dict) -> float:
    """Predict total latency from LUT (ms)."""
    if not lut:
        return 0.0
    return sum(lut.get(str(ci), {}).get(str(oi), 0.5)
               for ci, oi in enumerate(arch))


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5.2 / 5.3 — Supernet Training
# ═════════════════════════════════════════════════════════════════════════════

HW_CFG = dict(
    supernet_epochs = 30,
    batch_size      = 128,
    lr              = 5e-4,
    weight_decay    = 1e-4,
    label_smoothing = 0.10,
    latency_budget  = 5.0,
    lambda_lat      = 0.01,
    num_workers     = 4,
    early_layers    = list(range(0, 10)),
    late_layers     = list(range(10, 20)),
)


def supernet_train(supernet: SuperNet, lut: dict) -> dict:
    """Trains the supernet using one-shot path sampling. Returns history dict."""
    print(f"\n{'='*60}\n  Phase 5.3 — Supernet Training (One-Shot NAS)\n{'='*60}")

    train_loader, val_loader = get_dataloaders(
        batch_size=HW_CFG["batch_size"], num_workers=HW_CFG["num_workers"], pin_memory=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=HW_CFG["label_smoothing"])
    optimizer = optim.AdamW(supernet.parameters(),
                             lr=HW_CFG["lr"], weight_decay=HW_CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HW_CFG["supernet_epochs"])
    scaler    = GradScaler()
    history   = {"train_loss": [], "val_acc1": []}

    n_epochs = DRY_SUPERNET_EPOCHS if DRY_RUN else HW_CFG["supernet_epochs"]
    for epoch in range(1, n_epochs + 1):
        supernet.train()
        t0, run_loss, total = time.time(), 0., 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if DRY_RUN and batch_idx >= DRY_BATCHES:
                break
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if epoch <= HW_CFG["supernet_epochs"] // 2:
                arch = [0] * NUM_CELLS
                for i in HW_CFG["late_layers"]:
                    arch[i] = random.randint(0, NUM_OPS - 1)
            else:
                arch = supernet.random_arch()

            lat_penalty = max(0.0, predict_latency(arch, lut) - HW_CFG["latency_budget"])
            optimizer.zero_grad()
            with autocast():
                out  = supernet(imgs, arch)
                loss = criterion(out, labels) + HW_CFG["lambda_lat"] * lat_penalty

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(supernet.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * imgs.size(0)
            total    += imgs.size(0)

        scheduler.step()

        supernet.eval()
        correct, n_val = 0, 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader):
                if DRY_RUN and batch_idx >= DRY_VAL_BATCHES:
                    break
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with autocast():
                    out = supernet(imgs, supernet.random_arch())
                correct += (out.argmax(1) == labels).sum().item()
                n_val   += labels.size(0)

        val_acc1 = correct / n_val
        history["train_loss"].append(run_loss / total)
        history["val_acc1"].append(val_acc1)
        print(f"  Epoch [{epoch:02d}/{HW_CFG['supernet_epochs']}]  "
              f"Loss={run_loss/total:.4f}  ValAcc@1={val_acc1*100:.2f}%  "
              f"t={time.time()-t0:.1f}s")

    ckpt = MODELS_DIR / "supernet_final.pth"
    torch.save({"state_dict": supernet.state_dict(), "history": history}, ckpt)
    print(f"✓  Supernet saved → {ckpt}")
    return history


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6 — Evolutionary Architecture Search
# ═════════════════════════════════════════════════════════════════════════════

EVO_CFG = dict(
    population     = 50,
    generations    = 20,
    top_k          = 10,
    mutation_p     = 0.15,
    n_eval_batches = 20,
)


def evaluate_arch(supernet: SuperNet, arch: list[int],
                  val_loader, lut: dict) -> tuple[float, float]:
    """Quick accuracy estimate + predicted latency. Returns (accuracy, latency_ms)."""
    supernet.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            if i >= EVO_CFG["n_eval_batches"]:
                break
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with autocast():
                out = supernet(imgs, arch)
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / max(total, 1), predict_latency(arch, lut)



def evolutionary_search(supernet: SuperNet, lut: dict) -> list[dict]:
    """Returns a list of evaluated architectures."""
    print(f"\n{'='*60}\n  Phase 6 — Evolutionary Architecture Search\n{'='*60}")

    _, val_loader = get_dataloaders(batch_size=256, num_workers=HW_CFG["num_workers"], pin_memory=True)
    n_generations = DRY_GENERATIONS if DRY_RUN else EVO_CFG["generations"]
    pop_size      = DRY_POPULATION  if DRY_RUN else EVO_CFG["population"]
    population    = [supernet.random_arch() for _ in range(pop_size)]
    evaluated     = {}

    def score(a): return evaluated.get(tuple(a), (0., 999.))[0]

    all_results = []
    for gen in range(1, n_generations + 1):
        for arch in population:
            key = tuple(arch)
            if key not in evaluated:
                acc, lat = evaluate_arch(supernet, arch, val_loader, lut)
                evaluated[key] = (acc, lat)
                all_results.append({"arch": arch, "acc": acc, "lat_ms": lat})

        top_k = sorted(population, key=score, reverse=True)[:EVO_CFG["top_k"]]
        new_pop = top_k.copy()
        while len(new_pop) < EVO_CFG["population"]:
            p1, p2 = random.sample(top_k, 2)
            # Crossover
            cut = random.randint(1, len(p1) - 1)
            child = p1[:cut] + p2[cut:]
            # Mutate
            for i in range(len(child)):
                if random.random() < EVO_CFG["mutation_p"]:
                    child[i] = random.randint(0, NUM_OPS - 1)
            new_pop.append(child)
        population = new_pop

        best = max(evaluated.values(), key=lambda v: v[0])
        print(f"  Gen [{gen:02d}/{EVO_CFG['generations']}]  "
              f"Best acc={best[0]*100:.2f}%  lat={best[1]:.2f} ms  "
              f"Evaluated={len(evaluated)} archs")

    with open(RESULTS_DIR / "search_history.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓  Search history → {RESULTS_DIR / 'search_history.json'}")
    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Script entry-point
# ═════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore")


print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU  : {torch.cuda.get_device_name(0)}"
          f"  |  VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

lut_path = RESULTS_DIR / "latency_lut.json"
if lut_path.exists():
    with open(lut_path) as f:
        lut = json.load(f)
    print(f"[Phase 5.1] Loaded existing LUT from {lut_path.name}")
else:
    lut = build_latency_lut(n_runs=50)

supernet = SuperNet().to(DEVICE)
n_params = sum(p.numel() for p in supernet.parameters() if p.requires_grad)
print(f"[Phase 5.3] Supernet params : {n_params/1e6:.2f} M")

supernet_ckpt = MODELS_DIR / "supernet_final.pth"
if supernet_ckpt.exists():
    ckpt = torch.load(supernet_ckpt, map_location=DEVICE)
    supernet.load_state_dict(ckpt["state_dict"])
    print(f"[Phase 5.3] Loaded supernet from {supernet_ckpt.name}")
else:
    supernet_train(supernet, lut)

results = evolutionary_search(supernet, lut)

budget = HW_CFG["latency_budget"]
valid  = [r for r in results if r["lat_ms"] <= budget] or results
best   = max(valid, key=lambda r: r["acc"])

best_path = RESULTS_DIR / "best_arch.json"
with open(best_path, "w") as f:
    json.dump(best, f, indent=2)
print(f"✓  Best arch → {best_path}")
print(f"   Accuracy : {best['acc']*100:.2f}%  |  Latency : {best['lat_ms']:.2f} ms")
print(f"   Ops      : {[OP_NAMES[i] for i in best['arch']]}")

# Plot Pareto Front
accs = [r["acc"] * 100 for r in results]
lats = [r["lat_ms"]    for r in results]

pareto_mask = []
for i, (a1, l1) in enumerate(zip(accs, lats)):
    dominated = any(
        (a2 >= a1 and l2 <= l1 and (a2 > a1 or l2 < l1))
        for j, (a2, l2) in enumerate(zip(accs, lats)) if j != i
    )
    pareto_mask.append(not dominated)

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter([l for l, p in zip(lats, pareto_mask) if not p],
           [a for a, p in zip(accs, pareto_mask) if not p],
           alpha=0.4, s=20, c="#95A5A6", label="Searched architectures")
px = [l for l, p in zip(lats, pareto_mask) if p]
py = [a for a, p in zip(accs, pareto_mask) if p]
order = sorted(range(len(px)), key=lambda i: px[i])
ax.plot([px[o] for o in order], [py[o] for o in order],
        "o-", lw=2, color="#E74C3C", ms=8, label="Pareto front", zorder=5)
ax.set_xlabel("Predicted Latency (ms)", fontsize=12)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
ax.set_title("Hardware-Aware NAS — Search Results: Accuracy vs Latency",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
out = RESULTS_DIR / "pareto_front.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓  Pareto front → {out}")
