# -*- coding: utf-8 -*-
# Tile2Vec training script for 68x68, 4-channel (B2,B3,B4,B8) 16-bit GeoTIFF (tifffile version)
# Dependencies: pip install tifffile tqdm torch

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import tifffile as tiff
import torch.nn.functional as F

# -----------------------
# 0) Channel normalization parameters (example values, replace with stats from your dataset)
# -----------------------
# Suggestion: compute per-channel mean/std on a batch of training data; below are placeholder values
MEAN = torch.tensor([0.12, 0.13, 0.14, 0.35], dtype=torch.float32).view(4, 1, 1)
STD  = torch.tensor([0.08, 0.08, 0.09, 0.15], dtype=torch.float32).view(4, 1, 1)

# -----------------------
# 1) Utility functions: geographic distance & filename parsing
# -----------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c  # km

def parse_lat_lon(filename):
    # Convention: satellite_image_<lat>_<lon>.tif
    base = os.path.basename(filename)
    base = base.replace('satellite_image_', '')
    base = os.path.splitext(base)[0]
    lat, lon = base.split('_')[:2]
    return float(lat), float(lon)

# -----------------------
# 2) Read GeoTIFF (B2,B3,B4,B8 channels, normalized to [0,1], no resize) + per-channel normalization
# -----------------------
def read_s2_core4_tif(path):
    """
    Read 16-bit 4-channel (B2,B3,B4,B8) tif, return torch.float32, shape [4, 68, 68]
    Processing: convert to [C,H,W] → normalize to [0,1] → per-channel normalization
    """
    arr = tiff.imread(path)  # (H,W,4)

    # Convert to (C,H,W)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Expect (H,W,4), got {arr.shape} for {path}")

    # Convert to float32 and normalize to [0,1]
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr / 10000.0, 0.0, 1.0)

    # Convert to Tensor and apply per-channel normalization
    x = torch.from_numpy(arr)  # [4,H,W], float32
    x = (x - MEAN) / (STD + 1e-6)

    return x


# -----------------------
# 3) Dataset (triplet sampling, positive/negative thresholds based on geographic distance in km)
# -----------------------
class SatelliteTripletDatasetWithGeo(Dataset):
    def __init__(self, data_dir, positive_threshold=1.0, negative_threshold=5.0):
        self.data_dir = data_dir
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        self.image_info = []
        for fname in os.listdir(self.data_dir):
            if fname.lower().endswith(('.tif', '.tiff')):
                fpath = os.path.join(self.data_dir, fname)
                lat, lon = parse_lat_lon(fname)
                self.image_info.append((fpath, lat, lon))
        if not self.image_info:
            raise ValueError(f"No GeoTIFF images found in {self.data_dir}")

    def __len__(self):
        return len(self.image_info)

    def _load(self, path):
        # No resize: directly read 68x68 four channels
        return read_s2_core4_tif(path)  # [4,68,68] float32

    def __getitem__(self, idx):
        anchor_path, alat, alon = self.image_info[idx]
        anchor = self._load(anchor_path)

        pos_cands, neg_cands = [], []
        for path, lat, lon in self.image_info:
            if path == anchor_path:
                continue
            d = haversine(alat, alon, lat, lon)
            if d < self.positive_threshold:
                pos_cands.append(path)
            if d > self.negative_threshold:
                neg_cands.append(path)

        if not pos_cands:
            pos_cands = [p for p,_,_ in self.image_info if p != anchor_path]
        if not neg_cands:
            neg_cands = [p for p,_,_ in self.image_info if p != anchor_path]

        pos_path = random.choice(pos_cands)
        neg_path = random.choice(neg_cands)

        positive = self._load(pos_path)
        negative = self._load(neg_path)

        return anchor, positive, negative

# -----------------------
# 4) Model: two-layer CNN + BN + global pooling + FC
# Input channels=4, output=embedding_dim
# -----------------------
class Tile2VecModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Added BatchNorm2d for each conv layer
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=5, stride=2, padding=2),  # [B,128,34,34]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),# [B,256,16,16]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)                                 # [B,256,1,1]
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.encoder(x).flatten(1)  # [B,256]
        x = self.fc(x)
        return x  # No L2 normalization here, embedding L2 regularization applied in loss

# -----------------------
# 5) Multi-city configuration
# -----------------------
cities = {
    "Adelaide": "data/images/Adelaide",
    "Brisbane": "data/images/Brisbane",
    "Canberra": "data/images/Canberra",
    "Darwin": "data/images/Darwin",
    "Hobart": "data/images/Hobart",
    "Melbourne": "data/images/Melbourne",
    "Perth": "data/images/Perth",
    "Sydney": "data/images/Sydney"
}

positive_threshold = 1.0
negative_threshold = 5.0
batch_size = 32
embedding_dim = 128
num_epochs = 300
learning_rate = 1e-4
embedding_l2 = 1e-2  # L2 regularization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# 6) Main loop: train each city in sequence
# -----------------------
if __name__ == '__main__':
    for city, data_dir in cities.items():
        print(f"\n==================== Training on {city} ====================")

        dataset = SatelliteTripletDatasetWithGeo(
            data_dir=data_dir,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
            pin_memory=True
        )

        model = Tile2VecModel(embedding_dim=embedding_dim).to(device)
        criterion = nn.TripletMarginLoss(margin=1.0, p=2)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = GradScaler()

        best_loss = float('inf')
        patience = 30
        trigger_times = 0
        model.train()

        save_path = f"tile2vec_model_core4_{city}.pth"

        for epoch in range(1, num_epochs+1):
            running = 0.0
            pbar = tqdm(dataloader, desc=f"{city} Epoch {epoch}/{num_epochs}")
            for anchor, positive, negative in pbar:
                anchor = anchor.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                negative = negative.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    a = model(anchor)
                    p = model(positive)
                    n = model(negative)
                    triplet_loss = criterion(a, p, n)

                    l2_term = (a.norm(dim=1).mean() +
                               p.norm(dim=1).mean() +
                               n.norm(dim=1).mean())
                    loss = triplet_loss + embedding_l2 * l2_term

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 tri=f"{triplet_loss.item():.4f}",
                                 l2=f"{(embedding_l2 * l2_term).item():.4f}")

            avg = running / len(dataloader)
            print(f"{city} Epoch [{epoch}/{num_epochs}]  Loss: {avg:.4f}")

            if avg < best_loss - 0.001:
                best_loss = avg
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"{city}: Early stopping at epoch {epoch}")
                    break

        torch.save(model.state_dict(), save_path)
        print(f"{city} model saved to {save_path}")

        del model, optimizer, dataloader
        torch.cuda.empty_cache()
