import os
import json
import copy
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as tvm

warnings.filterwarnings('ignore')
%matplotlib inline 

# ── GPU optimizations (RTX 4060 / Ampere) ─────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    # TF32: ~10% free throughput gain on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'TF32   : matmul={torch.backends.cuda.matmul.allow_tf32}  '
          f'cudnn={torch.backends.cudnn.allow_tf32}')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path.cwd()
PROCESSED_DIR = BASE_DIR / 'processed'
MODELS_DIR    = BASE_DIR / 'models'
RESULTS_DIR   = BASE_DIR / 'results'
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BASELINE_NAMES = ['mobilenetv2', 'shufflenetv2', 'efficientnet_b0']

# ── Load manifest ──────────────────────────────────────────────────────────────
manifest_path = PROCESSED_DIR / 'data_manifest.pkl'
assert manifest_path.exists(), 'Run data-preprocessing.py first!'

with open(manifest_path, 'rb') as f:
    manifest = pickle.load(f)

val_samples = [(Path(p), lbl) for p, lbl in manifest['val']]
MEAN        = tuple(manifest['mean'])
STD         = tuple(manifest['std'])
NUM_CLASSES = manifest['num_classes']

print(f'Val samples : {len(val_samples):,}  |  Classes : {NUM_CLASSES}')

# ── Transform + Dataset ────────────────────────────────────────────────────────
VAL_TRANSFORM = T.Compose([
    T.CenterCrop(56),
    T.Resize(64, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


class TinyImageNetVal(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return VAL_TRANSFORM(Image.open(path).convert('RGB')), label


def get_val_loader(batch_size: int = 256,
                   num_workers: int = 6,   # FIX: was hardcoded to 4; match training config
                   pin_memory: bool = True) -> DataLoader:
    return DataLoader(
        TinyImageNetVal(val_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and (DEVICE.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )


# ── Latency measurement ────────────────────────────────────────────────────────
@torch.no_grad()
def measure_latency_ms(model: nn.Module, n_runs: int = 100,
                        use_channels_last: bool = True) -> float:
    model.eval()
    dummy = torch.randn(1, 3, 64, 64, device=DEVICE)
    if use_channels_last and DEVICE.type == 'cuda':
        dummy = dummy.to(memory_format=torch.channels_last)
    lats = []
    for _ in range(10):
        _ = model(dummy)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(dummy)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    return float(np.median(lats))


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    correct1, correct5, total, total_loss = 0, 0, 0, 0.
    criterion = nn.CrossEntropyLoss()

    for imgs, labels in loader:
        # FIX: non_blocking=True was missing in the original eval function
        # channels_last: consistent with training for maximum GPU efficiency
        imgs   = imgs.to(DEVICE, non_blocking=True,
                         memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)   # FIX: non_blocking added
        with autocast(device_type=DEVICE.type):
            out  = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        correct1   += (out.argmax(1) == labels).sum().item()
        _, top5    = out.topk(5, dim=1)
        correct5   += sum(labels[i].item() in top5[i].tolist()
                          for i in range(labels.size(0)))
    return {
        'loss': round(total_loss / total, 4),
        'acc1': round(correct1 / total * 100, 2),
        'acc5': round(correct5 / total * 100, 2),
    }


# ── Quantization ───────────────────────────────────────────────────────────────
def quantize_model(model: nn.Module, name: str) -> nn.Module:
    """Dynamic INT8 quantization (CPU-side). Falls back gracefully on failure."""
    # quantize_dynamic requires CPU — deepcopy avoids mutating the GPU model
    model_cpu = copy.deepcopy(model).cpu().eval()
    try:
        q_model = torch.quantization.quantize_dynamic(
            model_cpu, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        out_path = MODELS_DIR / f'{name}_quantized.pth'
        torch.save(q_model.state_dict(), out_path)
        print(f'  [PTQ] {name} → {out_path.name}  '
              f'({out_path.stat().st_size / 1024**2:.1f} MB)')
        return q_model
    except Exception as e:
        print(f'  [PTQ] {name} quantization failed: {e}')
        return model_cpu


# ── Pruning ────────────────────────────────────────────────────────────────────
def prune_model(model: nn.Module, name: str, amount: float = 0.30) -> nn.Module:
    """Unstructured L1 magnitude pruning on all Conv2d layers."""
    model_pruned = copy.deepcopy(model).cpu()
    parameters_to_prune = [
        (m, 'weight') for m in model_pruned.modules() if isinstance(m, nn.Conv2d)
    ]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, param in parameters_to_prune:
        prune.remove(module, param)

    total = sum(p.numel() for p in model_pruned.parameters())
    nzero = sum((p != 0).sum().item() for p in model_pruned.parameters())
    sparsity = 1 - nzero / total
    print(f'  [Pruning] {name}  sparsity={sparsity:.1%}')

    out_path = MODELS_DIR / f'{name}_pruned.pth'
    torch.save(model_pruned.state_dict(), out_path)
    print(f'  [Pruning] Saved → {out_path.name}')
    return model_pruned.to(DEVICE)


# ── ONNX export + benchmark ────────────────────────────────────────────────────
def export_and_benchmark_onnx(model: nn.Module, name: str, n_runs: int = 100) -> float:
    """Export to ONNX and benchmark with ONNXRuntime. Returns latency or -1."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(f'  [ONNX] onnxruntime not installed — skipping {name}')
        return -1.0

    out_path  = str(MODELS_DIR / f'{name}.onnx')
    model_cpu = copy.deepcopy(model).cpu().eval()
    dummy     = torch.randn(1, 3, 64, 64, device='cpu')

    try:
        torch.onnx.export(
            model_cpu, dummy, out_path,
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=['input'], output_names=['logits'],
            dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
        )
        size_mb = Path(out_path).stat().st_size / 1024**2
        print(f'  [ONNX] {name} exported ({size_mb:.1f} MB) → {out_path}')
    except Exception as e:
        print(f'  [ONNX] Export failed for {name}: {e}')
        return -1.0

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(out_path, providers=providers)
    inp  = {'input': np.random.randn(1, 3, 64, 64).astype(np.float32)}
    lats = []
    for _ in range(10):
        sess.run(None, inp)
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, inp)
        lats.append((time.perf_counter() - t0) * 1000)
    lat_ms = float(np.median(lats))
    print(f'  [ONNX] {name} runtime latency : {lat_ms:.2f} ms')
    return lat_ms


# ── Helper: rebuild model skeleton from name ───────────────────────────────────
def _build_skeleton(name: str) -> nn.Module:
    if name == 'mobilenetv2':
        model = tvm.mobilenet_v2(weights=None, num_classes=NUM_CLASSES)
        model.features[0][0].stride = (1, 1)
    elif name == 'shufflenetv2':
        model = tvm.shufflenet_v2_x1_0(weights=None, num_classes=NUM_CLASSES)
        model.conv1[0].stride = (1, 1)
    elif name == 'efficientnet_b0':
        model = tvm.efficientnet_b0(weights=None, num_classes=NUM_CLASSES)
        model.features[0][0].stride = (1, 1)
    else:
        raise ValueError(f'Unknown model: {name}')
    return model


# ── Main evaluation loop ───────────────────────────────────────────────────────
val_loader     = get_val_loader(batch_size=256, num_workers=6, pin_memory=True)
all_model_data = []

for name in BASELINE_NAMES:
    ckpt_path = MODELS_DIR / f'{name}_best.pth'
    if not ckpt_path.exists():
        print(f'  ⚠  Checkpoint not found for {name} — skipping')
        continue

    print(f"\n{'='*55}\n  Evaluating : {name.upper()}\n{'='*55}")

    model = _build_skeleton(name)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    # channels_last: matches the format used during training for correct eval behavior
    model = model.to(memory_format=torch.channels_last)
    model = model.to(DEVICE)

    val_metrics = evaluate(model, val_loader)
    latency_ms  = measure_latency_ms(model, use_channels_last=True)
    params_m    = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    size_mb     = ckpt_path.stat().st_size / 1024**2

    try:
        import importlib
        thop = importlib.import_module('thop')
        _model_copy = copy.deepcopy(model)
        # Profile with channels_last dummy input to match training conditions
        dummy_thop = torch.randn(1, 3, 64, 64, device=DEVICE).to(
            memory_format=torch.channels_last)
        flops_m, _ = thop.profile(_model_copy,
                                   inputs=(dummy_thop,), verbose=False)
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
        'name':          name,
        'acc1':          val_metrics['acc1'],
        'acc5':          val_metrics['acc5'],
        'params_M':      round(params_m, 3),
        'flops_M':       round(flops_m, 1),
        'lat_ms':        round(latency_ms, 2),
        'onnx_lat_ms':   round(onnx_lat, 2),
        'model_size_MB': round(size_mb, 1),
    })

    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

# ── Optional: include NAS result if available ──────────────────────────────────
nas_path = RESULTS_DIR / 'best_arch.json'
if nas_path.exists():
    with open(nas_path) as f:
        nas_info = json.load(f)
    all_model_data.append({
        'name':          'NAS-Found',
        'acc1':          round(nas_info.get('acc', 0) * 100, 2),
        'acc5':          0.0,
        'params_M':      0.0,
        'flops_M':       0.0,
        'lat_ms':        round(nas_info.get('lat_ms', 0), 2),
        'onnx_lat_ms':  -1.0,
        'model_size_MB': 0.0,
    })

# ── Summary table ──────────────────────────────────────────────────────────────
print('\n' + '═' * 90)
print(f"  {'Model':<22} {'Acc@1':>7} {'Acc@5':>7} {'Params':>8} {'FLOPs':>9} "
      f"{'Lat(ms)':>9} {'Size(MB)':>9}")
print('═' * 90)
for m in all_model_data:
    print(f"  {m['name']:<22} {m['acc1']:>7.2f} {m['acc5']:>7.2f} "
          f"{m['params_M']:>8.2f} {m['flops_M']:>9.1f} "
          f"{m['lat_ms']:>9.2f} {m['model_size_MB']:>9.1f}")
print('═' * 90)

out_json = RESULTS_DIR / 'final_comparison.json'
with open(out_json, 'w') as f:
    json.dump(all_model_data, f, indent=2)
print(f'\n✓  Final comparison data → {out_json}')

# ── Bar chart comparison ───────────────────────────────────────────────────────
if all_model_data:
    names   = [m['name'] for m in all_model_data]
    palette = plt.cm.Set2(np.linspace(0, 1, len(names)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for ax, (vals_key, ylabel, title) in zip(axes, [
        ('acc1',     'Top-1 Accuracy (%)',     'Accuracy'),
        ('lat_ms',   'Inference Latency (ms)', 'Latency (ms)'),
        ('params_M', '# Parameters (M)',        'Params'),
        ('flops_M',  'FLOPs (M)',               'FLOPs (M)'),
    ]):
        vals = [m[vals_key] for m in all_model_data]
        bars = ax.bar(names, vals, color=palette, edgecolor='white', linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(vals) * 0.01 if max(vals) > 0 else 0.01),
                    f'{v:.1f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)

    plt.suptitle('Final Model Comparison — Hardware-Aware NAS Project',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'final_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved → final_comparison.png')

# ── Accuracy vs latency scatter ────────────────────────────────────────────────
if all_model_data:
    palette2 = plt.cm.tab10(np.linspace(0, 1, len(all_model_data)))
    fig, ax  = plt.subplots(figsize=(10, 7))

    for m, color in zip(all_model_data, palette2):
        params_ = max(m['params_M'], 0.1)
        ax.scatter(m['lat_ms'], m['acc1'],
                   s=params_ * 80, color=color,
                   edgecolors='white', linewidths=1.5, zorder=3,
                   label=m['name'], alpha=0.9)
        ax.annotate(m['name'], (m['lat_ms'], m['acc1']),
                    textcoords='offset points', xytext=(8, 4),
                    fontsize=9, color=color)

    ax.set_xlabel('Inference Latency (ms, batch=1)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Latency Trade-off\n(Bubble size ∝ # Parameters)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'accuracy_vs_latency.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved → accuracy_vs_latency.png')

# ── FLOPs vs latency scatter ───────────────────────────────────────────────────
valid_flops = [m for m in all_model_data if m['flops_M'] > 0]
if valid_flops:
    palette3 = plt.cm.Set1(np.linspace(0, 1, len(valid_flops)))
    fig, ax  = plt.subplots(figsize=(9, 6))

    for m, color in zip(valid_flops, palette3):
        ax.scatter(m['flops_M'], m['lat_ms'], s=120, color=color,
                   label=m['name'], edgecolors='white', linewidths=1.5, zorder=3)
        ax.annotate(m['name'], (m['flops_M'], m['lat_ms']),
                    textcoords='offset points', xytext=(6, 3),
                    fontsize=9, color=color)

    ax.set_xlabel('FLOPs (M)', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('FLOPs vs Real Latency\n(FLOPs ≠ Latency — hardware matters!)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'flops_vs_latency.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved → flops_vs_latency.png')
else:
    print('No FLOPs data available (install thop for FLOPs counting).')