# -----------------------
# 1. Define Triplet Dataset with Sampling (unchanged)
# -----------------------

import os
import random
import math
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Calculate distance between two points on Earth's surface (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Extract latitude and longitude from filename
def parse_lat_lon(filename):
    parts = filename.replace('satellite_image_', '').replace('.png', '').split('_')
    lat, lon = float(parts[0]), float(parts[1])
    return lat, lon

# Triplet dataset
class SatelliteTripletDatasetWithGeo(Dataset):
    def __init__(self, data_dir, tile_size=640, transform=None, positive_threshold=1.0, negative_threshold=5.0):
        self.data_dir = data_dir
        self.tile_size = tile_size
        self.transform = transform
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        self.image_info = []
        for fname in os.listdir(self.data_dir):
            if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg'):
                lat, lon = parse_lat_lon(fname)
                full_path = os.path.join(self.data_dir, fname)
                self.image_info.append((full_path, lat, lon))

        if len(self.image_info) == 0:
            raise ValueError(f"No images found in {self.data_dir}")

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        anchor_path, anchor_lat, anchor_lon = self.image_info[idx]
        anchor_img = Image.open(anchor_path).convert('RGB')

        # Positive sample
        positive_candidates = []
        for path, lat, lon in self.image_info:
            if path == anchor_path:
                continue
            dist = haversine(anchor_lat, anchor_lon, lat, lon)
            if dist < self.positive_threshold:
                positive_candidates.append(path)

        if not positive_candidates:
            positive_candidates = [p[0] for p in self.image_info if p[0] != anchor_path]
        positive_path = random.choice(positive_candidates)
        positive_img = Image.open(positive_path).convert('RGB')

        # Negative sample
        negative_candidates = []
        for path, lat, lon in self.image_info:
            if path == anchor_path:
                continue
            dist = haversine(anchor_lat, anchor_lon, lat, lon)
            if dist > self.negative_threshold:
                negative_candidates.append(path)

        if not negative_candidates:
            negative_candidates = [p[0] for p in self.image_info if p[0] != anchor_path]
        negative_path = random.choice(negative_candidates)
        negative_img = Image.open(negative_path).convert('RGB')

        # Resize
        anchor_img = anchor_img.resize((self.tile_size, self.tile_size))
        positive_img = positive_img.resize((self.tile_size, self.tile_size))
        negative_img = negative_img.resize((self.tile_size, self.tile_size))

        # Convert to Tensor
        if self.transform:
            anchor = self.transform(anchor_img)
            positive = self.transform(positive_img)
            negative = self.transform(negative_img)
        else:
            anchor = torch.from_numpy(np.array(anchor_img)).permute(2, 0, 1).float() / 255.
            positive = torch.from_numpy(np.array(positive_img)).permute(2, 0, 1).float() / 255.
            negative = torch.from_numpy(np.array(negative_img)).permute(2, 0, 1).float() / 255.

        return anchor, positive, negative

# -----------------------
# 2. Define Tile2Vec Model (unchanged)
# -----------------------

import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # Added: Mixed precision training support

class Tile2VecModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(Tile2VecModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -----------------------
# 3. Parameter Settings
# -----------------------

data_dir = 'data/images/Adelaide'
tile_size = 640
positive_threshold = 1.0
negative_threshold = 5.0
batch_size = 64  # Modified: increased batch size
embedding_dim = 128
num_epochs = 50
learning_rate = 1e-3

# -----------------------
# 4. Dataset and Training Preparation
# -----------------------

dataset = SatelliteTripletDatasetWithGeo(
    data_dir=data_dir,
    tile_size=tile_size,
    positive_threshold=positive_threshold,
    negative_threshold=negative_threshold
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)  # Modified: increased num_workers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Tile2VecModel(embedding_dim=embedding_dim).to(device)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scaler = GradScaler()  # Added: GradScaler for mixed precision

# -----------------------
# 5. Start Training (with AMP)
# -----------------------

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Use tqdm to show epoch progress
    for batch_idx, (anchor, positive, negative) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        with autocast():  # Use mixed precision
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


# -----------------------
# 6. Save Model
# -----------------------

save_path = 'tile2vec_model.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
