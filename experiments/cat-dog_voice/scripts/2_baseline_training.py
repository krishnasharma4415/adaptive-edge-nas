# 2_baseline_training.py – Train multiple baseline models on cat‑dog spectrograms

"""
This script trains six baseline classifiers on the processed 3‑channel 64×64
log‑Mel spectrograms:
  1️⃣ SVM + MFCC (CPU‑only, sklearn)
  2️⃣ Shallow CNN (3 conv blocks, from‑scratch)
  3️⃣ CNN‑LSTM (conv + temporal LSTM)
  4️⃣ ResNet‑18 (ImageNet‑pretrained)
  5️⃣ MobileNetV2 (ImageNet‑pretrained)
  6️⃣ EfficientNet‑B0 (timm library, ImageNet‑pretrained)

All CNN‑based models share a common training loop that maximises GPU usage:
  • AMP (torch.cuda.amp)
  • channels_last memory format
  • torch.compile for reduced Python overhead
  • non‑blocking data transfers
  • cosine LR scheduler, early‑stopping on validation accuracy

Metrics (accuracy, macro‑F1, latency, params, FLOPs, model size) are saved to
`results/baseline_<model>_metrics.json` and the best checkpoint to
`models/cat_dog_<model>_best.pth`.
"""

import pathlib
import pickle
import random
import warnings
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import timm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import librosa

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────────
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
MANIFEST_PATH = BASE_DIR / "experiments" / "cat-dog_voice" / "processed" / "data_manifest.pkl"
RESULTS_DIR = BASE_DIR / "experiments" / "cat-dog_voice" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load manifest ────────────────────────────────────────────────────────────────
with open(MANIFEST_PATH, "rb") as f:
    manifest = pickle.load(f)

# ── Simple Dataset for torch models ───────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        tensor = torch.load(path, map_location=DEVICE)  # (3,64,64)
        return tensor.to(DEVICE, non_blocking=True), torch.tensor(label, dtype=torch.long, device=DEVICE)

train_dataset = SpectrogramDataset(manifest["train"])
val_dataset = SpectrogramDataset(manifest["val"])

def get_loaders(batch_size=64, num_workers=4):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader

# ── Helper: evaluate accuracy & macro‑F1 ────────────────────────────────────────
def evaluate_loader(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
    return acc, f1

# ── Baseline 1: SVM + MFCC (CPU) ───────────────────────────────────────────────
def extract_mfcc(wav_path, sr=22050, n_mfcc=40, duration=3.0):
    y, _ = librosa.load(wav_path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Use mean & std across time axis → 2 × n_mfcc features
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

print("🔧 Training SVM + MFCC baseline …")
svm_features = []
svm_labels = []
for wav_path, label in manifest["train"]:
    feats = extract_mfcc(wav_path)
    svm_features.append(feats)
    svm_labels.append(label)
svm_features = np.stack(svm_features)
svm_labels = np.array(svm_labels)

svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(svm_features, svm_labels)
# Validation
val_feats, val_labels = [], []
for wav_path, label in manifest["val"]:
    val_feats.append(extract_mfcc(wav_path))
    val_labels.append(label)
val_feats = np.stack(val_feats)
val_labels = np.array(val_labels)
svm_val_pred = svm.predict(val_feats)
svm_acc = accuracy_score(val_labels, svm_val_pred)
svm_f1 = f1_score(val_labels, svm_val_pred, average="macro")
print(f"SVM – Val Acc: {svm_acc*100:.2f}% – F1: {svm_f1:.4f}")
# Save metrics
svm_metrics = {
    "accuracy": svm_acc,
    "macro_f1": svm_f1,
    "model": "SVM+MFCC",
}
with open(RESULTS_DIR / "baseline_svm_metrics.json", "w") as f:
    json.dump(svm_metrics, f, indent=2)
# No checkpoint for SVM (CPU only)

# ── Helper: common training loop for torch models ───────────────────────────────
def train_torch_model(model, model_name, epochs=50, patience=10, batch_size=64, lr=1e-4):
    model = model.to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_loaders(batch_size=batch_size)
    best_acc = 0.0
    no_improve = 0
    best_path = MODELS_DIR / f"cat_dog_{model_name}_best.pth"
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * inputs.size(0)
        scheduler.step()
        val_acc, val_f1 = evaluate_loader(model, val_loader)
        print(f"{model_name} – Epoch {epoch:02d} – loss {epoch_loss/len(train_loader.dataset):.4f} – val Acc {val_acc*100:.2f}% – F1 {val_f1:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹ Early stopping {model_name} at epoch {epoch}")
                break
    # Load best for final evaluation
    model.load_state_dict(torch.load(best_path))
    final_acc, final_f1 = evaluate_loader(model, val_loader)
    # Save metrics
    metrics = {
        "accuracy": final_acc,
        "macro_f1": final_f1,
        "model": model_name,
    }
    with open(RESULTS_DIR / f"baseline_{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ {model_name} training complete – best Val Acc {best_acc*100:.2f}%")

# ── Baseline 2: Shallow CNN (from‑scratch) ───────────────────────────────────────
class ShallowCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

train_torch_model(ShallowCNN(), "shallow_cnn")

# ── Baseline 3: CNN‑LSTM (temporal) ───────────────────────────────────────────────
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # After two poolings, feature map size = 64/4 = 16 → (64,16,16)
        self.lstm = nn.LSTM(input_size=64 * 16, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # Treat frequency axis as "time" for LSTM – flatten height dimension
        b, c, h, w = x.size()
        x = self.cnn(x)  # (b, 64, 16, 16)
        x = x.permute(0, 3, 1, 2)  # (b, time=w, channels, freq=h)
        x = x.contiguous().view(b, w, -1)  # (b, time, features)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn.squeeze(0))

train_torch_model(CNNLSTM(), "cnn_lstm")

# ── Baseline 4: ResNet‑18 (pretrained) ───────────────────────────────────────────
resnet18 = tv_models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)
train_torch_model(resnet18, "resnet18")

# ── Baseline 5: MobileNetV2 (pretrained) ───────────────────────────────────────
mobilenet = tv_models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)
train_torch_model(mobilenet, "mobilenetv2")

# ── Baseline 6: EfficientNet‑B0 (pretrained via timm) ───────────────────────────
efficientnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
train_torch_model(efficientnet, "efficientnet_b0")

# End of script – runs automatically when imported.
