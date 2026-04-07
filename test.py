import os, argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import STL10

# ── ARGPARSE (NEW) ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluation Pipeline")
parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
args = parser.parse_args()

DATA_DIR = args.data_dir
CKPT_PATH = args.ckpt_path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

STUDENT_CHANNELS = [32, 96, 120, 232]
STUDENT_DEPTHS   = [2, 2, 4, 2]

IMG_SIZE = 96
NUM_CLASSES = 10

MEAN = [0.4467, 0.4398, 0.4066]
STD  = [0.2603, 0.2566, 0.2713]

_loader_kwargs = dict(pin_memory=True, num_workers=6, persistent_workers=True)

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

# ── DATA (UPDATED) ────────────────────────────────────────────────────────────
test  = STL10(DATA_DIR, split="test", download=True, transform=T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
]))

test_loader = DataLoader(test, batch_size=256, **_loader_kwargs)

# ── MODEL ─────────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x / keep_prob * rand
        
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pw1 = nn.Linear(dim, dim*4)
        self.pw2 = nn.Linear(dim*4, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self,x):
        res = x
        x = self.dw(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.pw2(F.gelu(self.pw1(x)))
        x = x.permute(0,3,1,2)
        return res + self.drop_path(x) 
    
def make_downsample(in_ch, out_ch):
    return nn.Sequential(
        nn.GroupNorm(1, in_ch),
        nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
    )

class ConvNeXt(nn.Module):
    def __init__(self, ch, depth, recurrent=False):
        super().__init__()
        self.recurrent = recurrent
        self.stem = nn.Conv2d(3, ch[0], 4, 4)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_rate = 0.1
        total_blocks = sum(depth)
        dp_rates = torch.linspace(0, dp_rate, total_blocks).tolist()
        
        cur = 0
        for i, c in enumerate(ch):
            blocks = []
            for j in range(depth[i]):
                blocks.append(Block(c, dp_rates[cur]))
                cur += 1
            self.stages.append(nn.Sequential(*blocks))
            if i < len(ch) - 1:
                self.downsamples.append(make_downsample(ch[i], ch[i + 1]))
        self.norm = nn.LayerNorm(ch[-1])
        self.head = nn.Linear(ch[-1], NUM_CLASSES)

    def forward_features(self, x):
        x = self.stem(x)
        for i, s in enumerate(self.stages):
            x = s(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        if self.recurrent:
            for _ in range(2):
                x = x + self.stages[-1](x)
        x = x.mean([-2, -1])
        return self.norm(x)

    def forward(self, x):
        return self.head(self.forward_features(x))

# ── LOAD MODEL (UPDATED) ──────────────────────────────────────────────────────
student = ConvNeXt(STUDENT_CHANNELS, STUDENT_DEPTHS, recurrent=True).to(DEVICE)
student.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
student.eval()

print("Student params (M):", count_params(student))

# ── TTA EVALUATION ────────────────────────────────────────────────────────────
NUM_CROPS = 5
crop_transform = T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0))

correct = 0
total = 0

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        batch_probs = torch.zeros(images.size(0), NUM_CLASSES, device=DEVICE)

        for _ in range(NUM_CROPS):
            crops = torch.stack([crop_transform(img) for img in images.cpu()])
            crops = crops.to(DEVICE, non_blocking=True)

            probs = torch.softmax(student(crops), dim=1)
            probs_flipped = torch.softmax(student(torch.flip(crops, dims=[3])), dim=1)

            batch_probs += (probs + probs_flipped) / 2

        batch_probs /= NUM_CROPS
        _, predicted = batch_probs.max(1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Final TTA Test Accuracy: {correct / total * 100:.2f}%")
