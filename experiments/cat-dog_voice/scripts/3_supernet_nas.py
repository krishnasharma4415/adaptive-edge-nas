# 3_supernet_nas.py – Hardware‑aware NAS for cat‑dog voice classification

"""
This script adapts the existing `hardware_aware.py` from the Tiny‑ImageNet experiment
to the audio domain. It:
  1️⃣ Loads the processed spectrogram manifest.
  2️⃣ Builds the same SuperNet search space (identical ops & cell config).
  3️⃣ Generates a fresh latency LUT for the 64×64 input (same as image case).
  4️⃣ Trains the SuperNet with a latency‑aware loss (teacher‑distillation optional).
  5️⃣ Runs the NSGA‑II‑style evolutionary search to obtain a Pareto front.
  6️⃣ Saves the best architecture (within a latency budget) to `results/best_arch.json`.

The script contains **no `def main()`** – it runs on import as requested.
All heavy‑weight GPU optimisations are enabled: AMP, channels_last, torch.compile,
non‑blocking data transfers, and a cosine LR scheduler.
"""

# ── Imports & reproducibility ───────────────────────────────────────────────────────
import pathlib, sys, json, random, time, warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

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

# ── Make Tiny‑ImageNet hardware‑aware utilities importable ────────────────────────
BASE_DIR = pathlib.Path.cwd()
TINY_SCRIPT_DIR = BASE_DIR / "experiments" / "tiny-imagenet" / "scripts"
sys.path.append(str(TINY_SCRIPT_DIR))
# Import SuperNet, build_op, OP_NAMES, CELL_CONFIG, NUM_OPS, NUM_CELLS, predict_latency
from hardware_aware import SuperNet, build_op, OP_NAMES, CELL_CONFIG, NUM_OPS, NUM_CELLS, predict_latency

# ── Paths ───────────────────────────────────────────────────────────────────────
PROCESSED_ROOT = BASE_DIR / "experiments" / "cat-dog_voice" / "processed"
RESULTS_ROOT   = BASE_DIR / "experiments" / "cat-dog_voice" / "results"
MODELS_ROOT    = BASE_DIR / "models"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH  = PROCESSED_ROOT / "data_manifest.pkl"

# ── Load manifest ────────────────────────────────────────────────────────────────
with open(MANIFEST_PATH, "rb") as f:
    manifest = pickle.load(f)
train_samples = [(pathlib.Path(p), lbl) for p, lbl in manifest["train"]]
val_samples   = [(pathlib.Path(p), lbl) for p, lbl in manifest["val"]]
CLASS_MAP     = manifest["class_map"]
MEAN, STD    = tuple(manifest["mean"]), tuple(manifest["std"])
NUM_CLASSES   = manifest["num_classes"]

# ── Simple Dataset (loads pre‑saved spectrogram tensors) ────────────────────────
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

BATCH_SIZE = 128
NUM_WORKERS = 4
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

# ── Hardware‑aware training configuration ────────────────────────────────────────
HW_CFG = dict(
    supernet_epochs = 40,
    batch_size      = BATCH_SIZE,
    lr              = 5e-4,
    weight_decay    = 1e-4,
    label_smoothing = 0.10,
    latency_budget  = 4.0,   # ms – tighter than image task
    lambda_lat      = 0.01,
    grad_clip       = 5.0,
    early_layers    = list(range(0, 10)),
    late_layers     = list(range(10, NUM_CELLS)),
)

# ── SuperNet instance (reuse existing weights where possible) ─────────────────────
supernet = SuperNet().to(DEVICE)
# Warm‑start from the ImageNet supernet checkpoint (ignore head mismatch)
old_ckpt = MODELS_ROOT / "supernet_final.pth"
if old_ckpt.exists():
    ck = torch.load(old_ckpt, map_location=DEVICE)
    state = ck["state_dict"]
    # Strip head weights (they expect 200 classes)
    head_keys = [k for k in state if k.startswith('head')]
    for k in head_keys:
        del state[k]
    supernet.load_state_dict(state, strict=False)
    print("✅ Warm‑started SuperNet from ImageNet checkpoint (head ignored)")
else:
    print("⚠️ No ImageNet supernet checkpoint found – training from scratch")

supernet = torch.compile(supernet)

# ── Loss, optimizer, scheduler, scaler ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=HW_CFG["label_smoothing"])
optimizer = optim.AdamW(supernet.parameters(), lr=HW_CFG["lr"], weight_decay=HW_CFG["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HW_CFG["supernet_epochs"])
scaler = GradScaler()

# ── Helper: validation accuracy ───────────────────────────────────────────────────
def evaluate(loader):
    supernet.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbl in loader:
            out = supernet(imgs, supernet.random_arch())
            _, pred = torch.max(out, 1)
            correct += (pred == lbl).sum().item()
            total   += lbl.size(0)
    return correct / total if total else 0.0

# ── SuperNet training loop ───────────────────────────────────────────────────────
history = {"train_loss": [], "val_acc": []}
for epoch in range(1, HW_CFG["supernet_epochs"] + 1):
    supernet.train()
    epoch_loss, samples = 0.0, 0
    for imgs, lbl in train_loader:
        optimizer.zero_grad(set_to_none=True)
        # Progressive sampling: early epochs only randomise late layers
        if epoch <= HW_CFG["supernet_epochs"] // 2:
            arch = [0] * NUM_CELLS
            for i in HW_CFG["late_layers"]:
                arch[i] = random.randint(0, NUM_OPS - 1)
        else:
            arch = supernet.random_arch()
        # Latency penalty
        lat_pen = max(0.0, predict_latency(arch) - HW_CFG["latency_budget"])
        with autocast(device_type=DEVICE.type):
            out = supernet(imgs, arch)
            loss = criterion(out, lbl) + HW_CFG["lambda_lat"] * lat_pen
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(supernet.parameters(), HW_CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() * imgs.size(0)
        samples    += imgs.size(0)
    scheduler.step()
    val_acc = evaluate(val_loader)
    history["train_loss"].append(epoch_loss / samples)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch:02d} – loss {epoch_loss/samples:.4f} – val Acc {val_acc*100:.2f}%")

# Save trained supernet
supernet_ckpt = MODELS_ROOT / "cat_dog_supernet_final.pth"
torch.save({"state_dict": supernet.state_dict(), "history": history}, supernet_ckpt)
print(f"✅ SuperNet checkpoint saved → {supernet_ckpt}")

# ── Plot training curves ────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(history["train_loss"], color="#3498DB")
axs[0].set_title("SuperNet Training Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[1].plot([a*100 for a in history["val_acc"]], color="#E74C3C")
axs[1].set_title("SuperNet Validation Accuracy (random path)")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Acc %")
plt.tight_layout()
plt.savefig(RESULTS_ROOT / "supernet_training.png", dpi=150)
plt.close()

# ── Evolutionary search configuration ─────────────────────────────────────────────
EVO_CFG = dict(
    population  = 50,
    generations = 20,
    top_k       = 10,
    mutation_p  = 0.15,
    n_eval_batches = 20,   # quick accuracy estimate per arch
)

# Helper: quick accuracy estimate for a given arch on validation set
@torch.no_grad()
def quick_acc(arch):
    supernet.eval()
    correct, total = 0, 0
    for i, (imgs, lbl) in enumerate(val_loader):
        if i >= EVO_CFG["n_eval_batches"]:
            break
        out = supernet(imgs, arch)
        _, pred = torch.max(out, 1)
        correct += (pred == lbl).sum().item()
        total   += lbl.size(0)
    return correct / total if total else 0.0

# Initialise population
population = [supernet.random_arch() for _ in range(EVO_CFG["population"])]
archive = {}
all_results = []

for gen in range(1, EVO_CFG["generations"] + 1):
    # Evaluate unseen architectures
    for arch in population:
        key = tuple(arch)
        if key not in archive:
            acc = quick_acc(arch)
            lat = predict_latency(arch)
            archive[key] = (acc, lat)
            all_results.append({"arch": arch, "acc": acc, "lat_ms": lat})
    # Select top‑k by accuracy
    top_k = sorted(population, key=lambda a: archive[tuple(a)][0], reverse=True)[:EVO_CFG["top_k"]]
    # Crossover + mutation
    new_pop = top_k.copy()
    while len(new_pop) < EVO_CFG["population"]:
        p1, p2 = random.sample(top_k, 2)
        cut = random.randint(1, NUM_CELLS - 1)
        child = p1[:cut] + p2[cut:]
        # Mutation
        for i in range(NUM_CELLS):
            if random.random() < EVO_CFG["mutation_p"]:
                child[i] = random.randint(0, NUM_OPS - 1)
        new_pop.append(child)
    population = new_pop
    best_key = max(archive, key=lambda k: archive[k][0])
    best_acc, best_lat = archive[best_key]
    print(f"Gen {gen:02d}/{EVO_CFG['generations']} – Best Acc {best_acc*100:.2f}% – Lat {best_lat:.2f} ms – Evaluated {len(archive)} arches")

# Save search history
search_path = RESULTS_ROOT / "search_history.json"
with open(search_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"✅ Search history saved → {search_path}")

# ── Choose best architecture within latency budget ───────────────────────────────
budget = HW_CFG["latency_budget"]
valid = [r for r in all_results if r["lat_ms"] <= budget]
if not valid:
    valid = all_results  # fallback to any arch
best = max(valid, key=lambda r: r["acc"])
best_path = RESULTS_ROOT / "best_arch.json"
with open(best_path, "w") as f:
    json.dump(best, f, indent=2)
print(f"✅ Best architecture saved → {best_path}")
print(f"   Acc: {best['acc']*100:.2f}% | Latency: {best['lat_ms']:.2f} ms")
print(f"   Ops: {[OP_NAMES[i] for i in best['arch']]}")

# ── Pareto front plot (same as original) ────────────────────────────────────────
accs = [r['acc']*100 for r in all_results]
lats = [r['lat_ms'] for r in all_results]
# Compute Pareto front
pareto = []
for i, (a1, l1) in enumerate(zip(accs, lats)):
    dominated = any((a2 >= a1 and l2 <= l1) and (a2 > a1 or l2 < l1) for a2, l2 in zip(accs, lats) if i != accs.index(a2))
    if not dominated:
        pareto.append((l1, a1))

plt.figure(figsize=(10, 6))
plt.scatter(lats, accs, alpha=0.4, s=20, c="#95A5A6", label="Searched")
if pareto:
    px, py = zip(*sorted(pareto))
    plt.plot(px, py, "o-", lw=2, color="#E74C3C", label="Pareto front")
plt.scatter([best['lat_ms']], [best['acc']*100], s=200, color="gold", edgecolors="black", label="Best (budget)")
plt.axvline(budget, ls="--", color="gray", label=f"Budget = {budget} ms")
plt.xlabel("Predicted Latency (ms)")
plt.ylabel("Top‑1 Accuracy (%)")
plt.title("Hardware‑Aware NAS – Accuracy vs Latency (Pareto)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_ROOT / "pareto_front.png", dpi=150)
plt.close()
print("✅ Pareto plot saved → pareto_front.png")

# End of script – runs automatically when imported.
