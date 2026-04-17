# 4_nas_finetune.py – Fine‑tune selected NAS architectures

"""
This script loads the best architecture(s) discovered by the hardware‑aware NAS
(step 3) and fine‑tunes them as **stand‑alone** models.

Features:
  • No `def main()` – the script runs on import.
  • Uses the same GPU‑optimised training utilities as the baseline script.
  • Supports fine‑tuning multiple architectures (e.g. best‑accuracy, best‑latency,
    balanced) if they are stored in `results/best_arch.json` as a list; otherwise it
    falls back to the single entry saved by `3_supernet_nas.py`.
  • Saves the final checkpoint to `models/cat_dog_nas_<tag>_best.pth` and a metrics
    JSON to `results/nas_<tag>_metrics.json`.
"""

import pathlib, json, random, warnings, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path.cwd()
PROCESSED_ROOT = BASE_DIR / "experiments" / "cat-dog_voice" / "processed"
RESULTS_ROOT   = BASE_DIR / "experiments" / "cat-dog_voice" / "results"
MODELS_ROOT    = BASE_DIR / "models"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH  = PROCESSED_ROOT / "data_manifest.pkl"
BEST_ARCH_PATH = RESULTS_ROOT / "best_arch.json"

# ── Load manifest (train/val split) ───────────────────────────────────────────────
import pickle
with open(MANIFEST_PATH, "rb") as f:
    manifest = pickle.load(f)
train_samples = [(pathlib.Path(p), lbl) for p, lbl in manifest["train"]]
val_samples   = [(pathlib.Path(p), lbl) for p, lbl in manifest["val"]]
NUM_CLASSES   = manifest["num_classes"]

# Simple Dataset – loads pre‑saved spectrogram tensors
class SpectrogramDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        p, lbl = self.entries[idx]
        tensor = torch.load(p, map_location=DEVICE)  # (3,64,64)
        return tensor.to(DEVICE, non_blocking=True), torch.tensor(lbl, dtype=torch.long, device=DEVICE)

train_ds = SpectrogramDataset(train_samples)
val_ds   = SpectrogramDataset(val_samples)
BATCH_SIZE = 64
NUM_WORKERS = 4
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

# ── Import the StandaloneNASModel definition from the original Tiny‑ImageNet code ──────
TINY_SCRIPT_DIR = BASE_DIR / "experiments" / "tiny-imagenet" / "scripts"
sys.path.append(str(TINY_SCRIPT_DIR))
from nas import StandaloneNASModel, OP_NAMES, CELL_CONFIG, NUM_CELLS

# Helper: evaluate a model on the validation set (accuracy & macro‑F1)
from sklearn.metrics import f1_score

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbl in loader:
            out = model(imgs)
            _, pred = torch.max(out, 1)
            all_preds.append(pred.cpu())
            all_labels.append(lbl.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1

# Training loop (shared for all architectures)
def train_finetune(model, tag, epochs=50, patience=12, lr=1e-3):
    model = model.to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_acc, best_f1 = 0.0, 0.0
    no_improve = 0
    ckpt_path = MODELS_ROOT / f"cat_dog_nas_{tag}_best.pth"
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, samples = 0.0, 0
        for imgs, lbl in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type):
                out = model(imgs)
                loss = criterion(out, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * imgs.size(0)
            samples += imgs.size(0)
        scheduler.step()
        val_acc, val_f1 = evaluate(model, val_loader)
        print(f"NAS {tag} – Epoch {epoch:02d} – loss {epoch_loss/samples:.4f} – Val Acc {val_acc*100:.2f}% – F1 {val_f1:.4f}")
        if val_acc > best_acc:
            best_acc, best_f1 = val_acc, val_f1
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹ Early stopping {tag} at epoch {epoch}")
                break
    # Save final metrics
    metrics = {"accuracy": best_acc, "macro_f1": best_f1, "model": f"nas_{tag}"}
    with open(RESULTS_ROOT / f"nas_{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Fine‑tuned NAS {tag} saved → {ckpt_path}")

# ── Load best architecture(s) ─────────────────────────────────────────────────────
if not BEST_ARCH_PATH.exists():
    raise FileNotFoundError(f"{BEST_ARCH_PATH} not found – run 3_supernet_nas.py first")
with open(BEST_ARCH_PATH, "r") as f:
    best_data = json.load(f)
# The file may contain a single dict or a list of dicts
if isinstance(best_data, list):
    arch_list = best_data
else:
    arch_list = [best_data]

# For each architecture, instantiate a StandaloneNASModel and fine‑tune it
for idx, arch_entry in enumerate(arch_list, start=1):
    arch = arch_entry["arch"] if isinstance(arch_entry, dict) else arch_entry
    # Tag for checkpoint naming – use index and optional label (e.g., "best", "speed")
    tag = f"arch{idx}"
    print(f"\n=== Fine‑tuning NAS architecture {idx}/{len(arch_list)} (tag: {tag}) ===")
    # Build the static model – the constructor expects the architecture list and
    # the number of classes. It re‑uses the same primitive ops as the SuperNet.
    nas_model = StandaloneNASModel(arch, num_classes=NUM_CLASSES)
    train_finetune(nas_model, tag)

# End of script – runs automatically when imported.
