# ── 1. Imports & Utils ────────────────────────────────────────────────────────
import os, math, copy, random, gc, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import torchvision.transforms as T
import numpy as np

# ── ARGPARSE (NEW) ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Training Pipeline")
parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
args = parser.parse_args()

DATA_DIR = args.data_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "./checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def evaluate(model, loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# ── 2. Hyperparameters ────────────────────────────────────────────────────────
IMG_SIZE    = 96
NUM_CLASSES = 10

TEACHER_CHANNELS = [64, 128, 256, 512]
TEACHER_DEPTHS   = [2, 2, 6, 2]

STUDENT_CHANNELS = [32, 96, 120, 232]
STUDENT_DEPTHS   = [2, 2, 4, 2]

BYOL_EPOCHS = 150
BYOL_LR     = 5e-4
BYOL_BATCH  = 256

EMA_BASE_M = 0.960
EMA_MAX_M  = 0.999

PROJ_HIDDEN = 1024
PROJ_OUT    = 256

KD_EPOCHS = 120
KD_LR     = 1e-3

# ── 3. Data ───────────────────────────────────────────────────────────────────
MEAN = [0.4467, 0.4398, 0.4066]
STD  = [0.2603, 0.2566, 0.2713]

class Aug:
    def __init__(self):
        self.t = T.Compose([
            T.RandomResizedCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])
    def __call__(self, x):
        return self.t(x), self.t(x)

ft_train_transform = T.Compose([
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])

# 🔥 UPDATED: Use DATA_DIR instead of "./data"
train = STL10(DATA_DIR, split="train", download=True, transform=ft_train_transform)
test  = STL10(DATA_DIR, split="test", download=True, transform=T.Compose([
    T.Resize(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)
]))
unlabeled = STL10(DATA_DIR, split="unlabeled", download=True, transform=Aug())

_loader_kwargs = dict(pin_memory=True, num_workers=6, persistent_workers=True)

train_loader = DataLoader(train, batch_size=256, shuffle=True, **_loader_kwargs)
test_loader  = DataLoader(test, batch_size=256, **_loader_kwargs)
unlab_loader = DataLoader(unlabeled, batch_size=BYOL_BATCH, shuffle=True, **_loader_kwargs)

# ── (REST OF YOUR CODE REMAINS IDENTICAL) ─────────────────────────────────────
# ⚠️ No further changes needed below this line

# [KEEP EVERYTHING ELSE EXACTLY SAME — MODEL, BYOL, KD, FT, TTA]
