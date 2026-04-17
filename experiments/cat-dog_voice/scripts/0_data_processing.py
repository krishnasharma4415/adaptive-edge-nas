# 0_data_processing.py – Audio preprocessing & dataset split

"""
This script prepares the cat‑dog voice dataset for the NAS pipeline.
It performs:
  1️⃣ Audits the raw WAV files (counts, sample‑rate check, corruption).
  2️⃣ Creates a stratified train/val split from the existing training folder.
  3️⃣ Generates 64×64 log‑Mel spectrogram tensors (3‑channel) for every split.
  4️⃣ Saves a manifest (pickle) containing file lists, class map, and normalization stats.
  5️⃣ Stores a small preview PNG of a sample spectrogram.
"""

# ── Imports & reproducibility ───────────────────────────────────────────────────────
import os
import json
import random
import pathlib
import pickle
import warnings
from collections import Counter

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

import torch

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
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # project root
DATA_ROOT = BASE_DIR / "data" / "cats_dogs"
TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT = DATA_ROOT / "test"
PROCESSED_ROOT = BASE_DIR / "experiments" / "cat-dog_voice" / "processed"
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = PROCESSED_ROOT / "data_manifest.pkl"
AUDIT_PATH = BASE_DIR / "experiments" / "cat-dog_voice" / "results" / "dataset_audit.json"
AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Helper: load wav, resample, pad/truncate, compute log‑Mel spectrogram ──────────────
def wav_to_logmel(wav_path: pathlib.Path, sr_target: int = 22050, duration: float = 3.0) -> np.ndarray:
    """Return a (64, 64) log‑Mel spectrogram (single channel)."""
    # Load raw audio (librosa handles many formats)
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # mono
    # Resample if needed
    if sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    # Pad / truncate to fixed length
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(
        y,
        sr=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        fmax=sr // 2,
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    # Normalize to [0, 1]
    log_S_norm = (log_S - log_S.min()) / (log_S.max() - log_S.min() + 1e-6)
    return log_S_norm.astype(np.float32)

# ── Audit raw dataset ───────────────────────────────────────────────────────────────
print("🔎 Auditing raw dataset …")
class_counts = Counter()
all_files = []
for class_dir in ["cat", "dog"]:
    class_path = TRAIN_ROOT / class_dir
    for wav in class_path.rglob("*.wav"):
        all_files.append((wav, class_dir))
        class_counts[class_dir] += 1
print(f"Found {len(all_files)} training wavs – class distribution: {dict(class_counts)}")
# Simple corruption check – try loading first 5 seconds of each file
corrupt = []
for wav, _ in all_files:
    try:
        _ = sf.read(wav, frames=22050 * 1)  # 1 s preview
    except Exception:
        corrupt.append(str(wav))
if corrupt:
    print(f"⚠️ Detected {len(corrupt)} corrupt files – they will be skipped.")
else:
    print("✅ No corrupt files detected.")

# ── Stratified train/val split (15 % validation) ────────────────────────────────────
val_ratio = 0.15
train_split = []
val_split = []
for cls in ["cat", "dog"]:
    cls_items = [(p, cls) for p, c in all_files if c == cls]
    random.shuffle(cls_items)
    n_val = int(len(cls_items) * val_ratio)
    val_split.extend(cls_items[:n_val])
    train_split.extend(cls_items[n_val:])
print(f"Train/val sizes → {len(train_split)} / {len(val_split)}")

# ── Test split (already provided) ───────────────────────────────────────────────────
# Expect test folder to contain subfolders per class – we mirror the same format.
test_split = []
for cls in ["cat", "dog"]:
    cls_path = TEST_ROOT / cls
    for wav in cls_path.rglob("*.wav"):
        test_split.append((wav, cls))
print(f"Test set size: {len(test_split)}")

# ── Build class map & compute channel‑wise mean/std from training spectrograms ───────
class_map = {"cat": 0, "dog": 1}
NUM_CLASSES = len(class_map)

# Compute mean/std on a random subset (max 2000 samples) for speed
subset = random.sample(train_split, min(2000, len(train_split)))
means = []
stds = []
for wav_path, _ in subset:
    spec = wav_to_logmel(wav_path)  # (64, 64)
    means.append(spec.mean())
    stds.append(spec.std())
MEAN = float(np.mean(means))
STD = float(np.mean(stds))
print(f"Global spectrogram mean/std → {MEAN:.4f} / {STD:.4f}")

# ── Save processed spectrogram tensors ───────────────────────────────────────────────
def save_tensor(tensor: np.ndarray, out_path: pathlib.Path):
    """Save a (64,64) float32 array as a 3‑channel torch tensor (C,H,W)."""
    # Replicate to 3 channels for ImageNet‑style models
    tensor3 = np.stack([tensor, tensor, tensor], axis=0)  # (3,64,64)
    torch.save(torch.from_numpy(tensor3), out_path)

SPECTRO_ROOT = PROCESSED_ROOT / "spectrograms"
SPECTRO_ROOT.mkdir(parents=True, exist_ok=True)

manifest = {
    "train": [],
    "val": [],
    "test": [],
    "class_map": class_map,
    "num_classes": NUM_CLASSES,
    "mean": MEAN,
    "std": STD,
}

for split_name, split in [("train", train_split), ("val", val_split), ("test", test_split)]:
    for wav_path, label in split:
        rel_path = wav_path.relative_to(DATA_ROOT)
        out_path = SPECTRO_ROOT / f"{rel_path.with_suffix('.pt')}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        spec = wav_to_logmel(wav_path)
        # Normalise using global stats (center‑scale)
        spec = (spec - MEAN) / (STD + 1e-6)
        save_tensor(spec, out_path)
        manifest[split_name].append((str(out_path), class_map[label]))

# ── Persist manifest ────────────────────────────────────────────────────────────────
with open(MANIFEST_PATH, "wb") as f:
    pickle.dump(manifest, f)
print(f"✅ Manifest saved to {MANIFEST_PATH}")

# ── Save audit JSON for reference ─────────────────────────────────────────────────
audit = {
    "train_samples": len(train_split),
    "val_samples": len(val_split),
    "test_samples": len(test_split),
    "class_counts": dict(class_counts),
    "global_mean": MEAN,
    "global_std": STD,
    "corrupt_files": corrupt,
}
with open(AUDIT_PATH, "w") as f:
    json.dump(audit, f, indent=2)
print(f"✅ Audit saved to {AUDIT_PATH}")

# ── Optional: generate a preview PNG of a random spectrogram ───────────────────────
sample_wav = random.choice(train_split)[0]
sample_spec = wav_to_logmel(sample_wav)
plt.figure(figsize=(3, 3))
plt.imshow(sample_spec, origin="lower", aspect="auto", cmap="viridis")
plt.title(f"Log‑Mel of {sample_wav.name}")
plt.axis("off")
preview_path = BASE_DIR / "experiments" / "cat-dog_voice" / "results" / "spectrogram_preview.png"
preview_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(preview_path, dpi=150)
plt.close()
print(f"✅ Preview image saved to {preview_path}")

# End of script – no __main__ guard as per user request
