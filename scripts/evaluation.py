"""
=============================================================================
 Phases 7–9 — Final Optimization & Evaluation
 Project  : Hardware-Aware NAS for Edge Devices
 Dataset  : Tiny-ImageNet-200
=============================================================================

Phase 7 — Final Model Optimization
  7.1 Post-Training Quantization (INT8 via torch.quantization)
  7.2 Magnitude Pruning (unstructured)
  7.3 ONNX export + ONNX Runtime inference benchmark

Phase 8 — Final Evaluation
  Compare all models on:
    Accuracy | Latency | Params | FLOPs | Size

Phase 9 — Visualization
  - Accuracy vs Latency scatter
  - FLOPs vs Latency
  - Model comparison bar chart

Outputs
  - models/{name}_quantized.pth / .onnx
  - results/final_comparison.json / .png
  - results/accuracy_vs_latency.png
  - results/flops_vs_latency.png
=============================================================================
"""

import os
import json
import copy
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
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as tvm

# ─────────────────────────────────────────────────────────────────────────────
# Paths & device
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent   # project root (one level above scripts/)
DATASET_DIR = BASE_DIR / "tiny-imagenet-200"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

VAL_DIR    = DATASET_DIR / "val"
VAL_ANNOT  = VAL_DIR / "val_annotations.txt"
WNIDS_FILE = DATASET_DIR / "wnids.txt"
STATS_FILE = RESULTS_DIR / "dataset_stats.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE_NAMES = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]

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
# Val dataset & loader
# ─────────────────────────────────────────────────────────────────────────────
VAL_TRANSFORM = T.Compose([
    T.CenterCrop(56),
    T.Resize(64, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


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


def get_val_loader(batch_size=256, num_workers=4, pin_memory=True):
    return DataLoader(
        TinyImageNetVal(), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def measure_latency_ms(model: nn.Module, n_runs: int = 100,
                        input_shape=(1, 3, 64, 64)) -> float:
    model.eval()
    dummy = torch.randn(*input_shape, device=DEVICE)
    lats  = []
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _  = model(dummy)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lats))


@torch.no_grad()
def evaluate(model: nn.Module, val_loader) -> dict:
    model.eval()
    correct1, correct5, total, total_loss = 0, 0, 0, 0.
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        correct1   += (out.argmax(1) == labels).sum().item()
        _, top5    = out.topk(5, dim=1)
        correct5   += sum(labels[i].item() in top5[i].tolist()
                          for i in range(labels.size(0)))
    return {
        "loss": round(total_loss / total, 4),
        "acc1": round(correct1 / total * 100, 2),
        "acc5": round(correct5 / total * 100, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7.1 — Post-Training Quantization (PTQ)
# ═════════════════════════════════════════════════════════════════════════════

def quantize_model(model: nn.Module, name: str) -> nn.Module:
    """Dynamic INT8 quantization. Falls back gracefully on failure."""
    model_cpu = copy.deepcopy(model).cpu().eval()
    try:
        q_model = torch.quantization.quantize_dynamic(
            model_cpu, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        out_path = MODELS_DIR / f"{name}_quantized.pth"
        torch.save(q_model.state_dict(), out_path)
        print(f"  [PTQ] {name} → {out_path}  ({out_path.stat().st_size/1024**2:.1f} MB)")
        return q_model
    except Exception as e:
        print(f"  [PTQ] {name} quantization failed: {e}")
        return model_cpu


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7.2 — Pruning
# ═════════════════════════════════════════════════════════════════════════════

def prune_model(model: nn.Module, name: str, amount: float = 0.3) -> nn.Module:
    """Unstructured magnitude pruning on all Conv2d layers."""
    model_pruned = copy.deepcopy(model).cpu()
    parameters_to_prune = [
        (m, "weight") for m in model_pruned.modules() if isinstance(m, nn.Conv2d)
    ]
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for module, param in parameters_to_prune:
        prune.remove(module, param)

    total = sum(p.numel() for p in model_pruned.parameters())
    nzero = sum((p != 0).sum().item() for p in model_pruned.parameters())
    print(f"  [Pruning] {name}  sparsity={1 - nzero/total:.1%}")

    out_path = MODELS_DIR / f"{name}_pruned.pth"
    torch.save(model_pruned.state_dict(), out_path)
    print(f"  [Pruning] Saved → {out_path}")
    return model_pruned.to(DEVICE)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7.3 — ONNX Export & ONNXRuntime Benchmark
# ═════════════════════════════════════════════════════════════════════════════

def export_and_benchmark_onnx(model: nn.Module, name: str, n_runs: int = 100) -> float:
    """Export model to ONNX and benchmark with ONNXRuntime. Returns latency or -1."""
    try:
        import onnxruntime as ort
    except ImportError:
        return -1.0

    out_path  = str(MODELS_DIR / f"{name}.onnx")
    model_cpu = copy.deepcopy(model).cpu().eval()
    dummy     = torch.randn(1, 3, 64, 64, device="cpu")

    try:
        torch.onnx.export(
            model_cpu, dummy, out_path,
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        print(f"  [ONNX] {name} exported ({Path(out_path).stat().st_size/1024**2:.1f} MB) → {out_path}")
    except Exception as e:
        print(f"  [ONNX] Export failed for {name}: {e}")
        return -1.0

    sess = ort.InferenceSession(out_path,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp  = {"input": np.random.randn(1, 3, 64, 64).astype(np.float32)}
    lats = []
    for _ in range(10):
        sess.run(None, inp)
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, inp)
        lats.append((time.perf_counter() - t0) * 1000)
    lat_ms = float(np.median(lats))
    print(f"  [ONNX] {name} runtime latency : {lat_ms:.2f} ms")
    return lat_ms


# ═════════════════════════════════════════════════════════════════════════════
# Phase 9 — Visualizations
# ═════════════════════════════════════════════════════════════════════════════




# ═════════════════════════════════════════════════════════════════════════════
# Script entry-point
# ═════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore")


print(f"Device : {DEVICE}")
val_loader     = get_val_loader(batch_size=256, num_workers=4, pin_memory=True)
all_model_data = []

for name in BASELINE_NAMES:
    ckpt_path = MODELS_DIR / f"{name}_best.pth"
    if not ckpt_path.exists():
        print(f"  ⚠  Checkpoint not found for {name} — skipping")
        continue

    print(f"\n{'='*55}\n  Evaluating : {name.upper()}\n{'='*55}")

    if name == "mobilenetv2":
        model = tvm.mobilenet_v2(weights=None, num_classes=200)
        model.features[0][0].stride = (1, 1)
    elif name == "shufflenetv2":
        model = tvm.shufflenet_v2_x1_0(weights=None, num_classes=200)
        model.conv1[0].stride = (1, 1)
    elif name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=None, num_classes=200)
        model.features[0][0].stride = (1, 1)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)

    val_metrics = evaluate(model, val_loader)
    latency_ms  = measure_latency_ms(model)
    params_m    = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    size_mb     = ckpt_path.stat().st_size / 1024**2

    try:
        import importlib
        thop = importlib.import_module("thop")
        # Profile on a copy — thop registers total_ops/total_params buffers
        # on the model in-place; doing this on the original would corrupt
        # its state_dict and break quantize_model / load_state_dict later.
        _model_copy = copy.deepcopy(model)
        flops_m, _ = thop.profile(_model_copy, inputs=(torch.randn(1, 3, 64, 64, device=DEVICE),), verbose=False)
        flops_m = float(flops_m) / 1e6
        del _model_copy
    except ImportError:
        flops_m = -1.0

    print(f"  Accuracy (Top-1/5) : {val_metrics['acc1']:.2f}% / {val_metrics['acc5']:.2f}%")
    print(f"  Latency : {latency_ms:.2f} ms  |  Params : {params_m:.2f} M  |  Size : {size_mb:.1f} MB")

    quantize_model(model, name)
    prune_model(model, name, amount=0.30)
    onnx_lat = export_and_benchmark_onnx(model, name)

    all_model_data.append({
        "name":          name,
        "acc1":          val_metrics["acc1"],
        "acc5":          val_metrics["acc5"],
        "params_M":      round(params_m, 3),
        "flops_M":       round(flops_m, 1),
        "lat_ms":        round(latency_ms, 2),
        "onnx_lat_ms":   round(onnx_lat, 2),
        "model_size_MB": round(size_mb, 1),
    })

nas_path = RESULTS_DIR / "best_arch.json"
if nas_path.exists():
    with open(nas_path) as f:
        nas_info = json.load(f)
    all_model_data.append({
        "name":          "NAS-Found",
        "acc1":          round(nas_info.get("acc", 0) * 100, 2),
        "acc5":          0.0,
        "params_M":      0.0,
        "flops_M":       0.0,
        "lat_ms":        round(nas_info.get("lat_ms", 0), 2),
        "model_size_MB": 0.0,
    })

print("\n" + "═"*90)
print(f"  {'Model':<22} {'Acc@1':>7} {'Acc@5':>7} {'Params':>8} {'FLOPs':>9} "
      f"{'Lat(ms)':>9} {'Size(MB)':>9}")
print("═"*90)
for m in all_model_data:
    print(f"  {m['name']:<22} {m['acc1']:>7.2f} {m['acc5']:>7.2f} "
          f"{m['params_M']:>8.2f} {m['flops_M']:>9.1f} "
          f"{m['lat_ms']:>9.2f} {m['model_size_MB']:>9.1f}")
print("═"*90)

out_json = RESULTS_DIR / "final_comparison.json"
with open(out_json, "w") as f:
    json.dump(all_model_data, f, indent=2)
print(f"\n✓  Final comparison data → {out_json}")

# ── 1. Final Comparison Bar Chart ──────────────────────────────────────
if all_model_data:
    names  = [m.get("name", "?") for m in all_model_data]
    acc1   = [m.get("acc1", 0) for m in all_model_data]
    lats   = [m.get("lat_ms", 0) for m in all_model_data]
    params = [m.get("params_M", 0) for m in all_model_data]
    flops  = [m.get("flops_M", 0) for m in all_model_data]

    palette = plt.cm.Set2(np.linspace(0, 1, len(names)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for ax, (vals, ylabel, title) in zip(axes, [
        (acc1,   "Top-1 Accuracy (%)",    "Accuracy"),
        (lats,   "Inference Latency (ms)", "Latency (ms)"),
        (params, "# Parameters (M)",       "Params"),
        (flops,  "FLOPs (M)",              "FLOPs (M)"),
    ]):
        bars = ax.bar(names, vals, color=palette, edgecolor="white", linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)

    plt.suptitle("Final Model Comparison — Hardware-Aware NAS Project",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out1 = RESULTS_DIR / "final_comparison.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Final comparison → {out1}")

# ── 2. Accuracy vs Latency Bubble Chart ───────────────────────────────
if all_model_data:
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, len(all_model_data)))

    for m, color in zip(all_model_data, palette):
        name   = m.get("name", "?")
        acc    = m.get("acc1", 0)
        lat    = m.get("lat_ms", 0)
        params_ = m.get("params_M", 1.0)
        ax.scatter(lat, acc, s=params_ * 80, color=color,
                   edgecolors="white", linewidths=1.5, zorder=3, label=name, alpha=0.9)
        ax.annotate(name, (lat, acc), textcoords="offset points",
                    xytext=(8, 4), fontsize=9, color=color)

    ax.set_xlabel("Inference Latency (ms, batch=1)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Latency Trade-off\n(Bubble size ∝ # Parameters)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out2 = RESULTS_DIR / "accuracy_vs_latency.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Accuracy vs Latency → {out2}")

# ── 3. FLOPs vs Latency Scatter ────────────────────────────────────────
f_vals = [m.get("flops_M", 0) for m in all_model_data]
l_vals = [m.get("lat_ms", 0)  for m in all_model_data]
names_ = [m.get("name", "?")   for m in all_model_data]

if f_vals and not all(v <= 0 for v in f_vals):
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = plt.cm.Set1(np.linspace(0, 1, len(names_)))

    for lat, flop, name, color in zip(l_vals, f_vals, names_, palette):
        if flop > 0:
            ax.scatter(flop, lat, s=120, color=color, label=name,
                       edgecolors="white", linewidths=1.5, zorder=3)
            ax.annotate(name, (flop, lat), textcoords="offset points",
                        xytext=(6, 3), fontsize=9, color=color)

    ax.set_xlabel("FLOPs (M)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("FLOPs vs Real Latency\n(FLOPs ≠ Latency — hardware matters!)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out3 = RESULTS_DIR / "flops_vs_latency.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  FLOPs vs Latency → {out3}")
