##!!! final version

import os
import random
import math
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os

# Added: Read GeoTIFF
import tifffile as tiff

# ---------------------------
# Helper Functions
# ---------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def parse_lat_lon(filename):
    parts = filename.replace('satellite_image_', '').replace('.png', '').replace('.jpg', '').replace('.jpeg', '').replace('.tif', '').replace('.tiff', '').split('_')
    lat, lon = float(parts[0]), float(parts[1])
    return lat, lon

# ---------------------------
# Dataset
# ---------------------------

class SatelliteTripletDatasetWithGeo(Dataset):
    def __init__(self, data_dir, tile_size=68, transform=None, positive_threshold=1.0, negative_threshold=5.0):
        self.data_dir = data_dir
        self.tile_size = tile_size
        self.transform = transform
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        self.image_info = []
        for fname in os.listdir(self.data_dir):
            if fname.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                lat, lon = parse_lat_lon(fname)
                full_path = os.path.join(self.data_dir, fname)
                self.image_info.append((full_path, lat, lon))

    def __len__(self):
        return len(self.image_info)

    def _load_img_as_tensor(self, path):
        """
        Return tensor (C,H,W), normalized according to image type:
        - GeoTIFF (uint16, S2 SR): divide by 10000.0 then clamp to [0,1]
        - PNG/JPG: divide by 255.0 to [0,1]
        """
        if path.lower().endswith(('.tif', '.tiff')):
            arr = tiff.imread(path)  # may be (H,W,C) or (C,H,W)
            if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] != arr.shape[2]:
                # convert to (H,W,C)
                arr = np.transpose(arr, (1, 2, 0))
            # convert to float32 and scale to [0,1] (S2 usually 0-10000 range)
            img = torch.from_numpy(arr.astype(np.float32)) / 10000.0
            # ensure (C,H,W)
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = img.permute(2, 0, 1)
            img = img.clamp(0.0, 1.0)
            return img
        else:
            img = Image.open(path).convert('RGB')
            np_img = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(np_img).permute(2, 0, 1)

    def __getitem__(self, idx):
        anchor_path, anchor_lat, anchor_lon = self.image_info[idx]

        # Read anchor
        anchor = self._load_img_as_tensor(anchor_path)

        # Select positive sample
        positive_candidates = [p[0] for p in self.image_info
                               if p[0] != anchor_path and haversine(anchor_lat, anchor_lon, p[1], p[2]) < self.positive_threshold]
        if not positive_candidates:
            positive_candidates = [p[0] for p in self.image_info if p[0] != anchor_path]
        pos_path = random.choice(positive_candidates)
        positive = self._load_img_as_tensor(pos_path)

        # Select negative sample
        negative_candidates = [p[0] for p in self.image_info
                               if p[0] != anchor_path and haversine(anchor_lat, anchor_lon, p[1], p[2]) > self.negative_threshold]
        if not negative_candidates:
            negative_candidates = [p[0] for p in self.image_info if p[0] != anchor_path]
        neg_path = random.choice(negative_candidates)
        negative = self._load_img_as_tensor(neg_path)

        # No resize: tiff already 68×68; PNG/JPG can be pre-converted offline to 68×68 to avoid interpolation
        # If resizing is required, add interpolate here to (self.tile_size, self.tile_size)

        # Apply tensor-level transform (e.g. normalization) if provided
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

# ---------------------------
# Transformer Components
# ---------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# ---------------------------
# Hybrid CNN-ViT Tile2Vec Model
# ---------------------------

class Tile2VecHybrid(nn.Module):
    def __init__(self, image_size=68, patch_size=4, dim=256, depth=4, heads=4, mlp_dim=512, dim_head=64, emb_dropout=0.1, dropout=0.1, embedding_dim=128):
        super().__init__()

        self.cnn_stem = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),  # +++
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),  # +++
            nn.ReLU()
        )

        # self.cnn_stem = nn.Sequential(
        #     nn.Conv2d(4, 128, kernel_size=5, stride=2, padding=2),  # in_channels: 4
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU()
        # )
        feature_size = image_size // 4  # three times stride=2 downsampling: 68 -> 8
        # patch_size set to 8, so feature_size//patch_size = 1
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(dim * patch_size * patch_size),
            nn.Linear(dim * patch_size * patch_size, dim),
            nn.LayerNorm(dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, (feature_size // patch_size) ** 2 + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.embedding = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embedding_dim))

    def forward(self, x):
        x = self.cnn_stem(x)  # (B, dim, 8, 8)
        x = self.to_patch_embedding(x)  # (B, 1, dim)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 2, dim)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.embedding(x[:, 0])

def compute_similarity_matrix(embeddings):
    norms = embeddings.norm(dim=1, keepdim=True)  # [B, 1]
    normed = embeddings / norms.clamp(min=1e-12)  # L2 normalize, avoid zero
    return torch.mm(normed, normed.t())  # cosine similarity matrix W ∈ [B, B]

def build_edge_mask(batch_size):
    A = torch.ones((batch_size, batch_size), dtype=torch.bool)
    A.fill_diagonal_(False)  # mask diagonal
    return A  # True means eligible as negative samples

def compute_attention_weights(W, mask, scale=10.0):
    W_scaled = W * scale
    W_scaled = W_scaled.masked_fill(~mask, float('-inf'))  # mask out positives
    attn_weights = torch.softmax(W_scaled, dim=1)  # each row: anchor-to-all-samples attention
    return attn_weights

def select_hard_negatives(attn_weights, embeddings):
    # Method 1: max attention
    max_indices = torch.argmax(attn_weights, dim=1)
    return embeddings[max_indices]
    # Method 2: weighted sum (optional)
    # return torch.matmul(attn_weights, embeddings)

# ---------------------------
# Training Loop
# ---------------------------

city = os.environ["CITY"]

if __name__ == '__main__':
    data_dir = f'data/images/{city}'
    tile_size = 68           # same as tiff size (no resize by default)
    embedding_dim = 128
    batch_size = 32
    num_epochs = 300
    learning_rate = 1e-4

    dataset = SatelliteTripletDatasetWithGeo(data_dir=data_dir, tile_size=tile_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # image_size=68, patch_size=8 to match 8x8 feature map after downsampling
    model = Tile2VecHybrid(image_size=68, patch_size=4, embedding_dim=embedding_dim).to(device)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = GradScaler()

    warmup_epochs = 10  # first 10 epochs use random negatives

    best_loss = float('inf')
    patience = 30
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            anchor = anchor.to(device)
            positive = positive.to(device)

            optimizer.zero_grad()
            with autocast():
                anchor_embed = model(anchor)
                positive_embed = model(positive)

                # 🚦 WARMUP: first 10 epochs use random negatives
                if epoch < warmup_epochs:
                    negative = negative.to(device)
                    negative_embed = model(negative)
                else:
                    # 🔥 GCA-style hard negative mining
                    all_imgs = torch.cat([anchor, positive], dim=0)
                    all_embeddings = model(all_imgs)  # [2B, D]
                    B = anchor.shape[0]

                    W = compute_similarity_matrix(all_embeddings[:B])  # [B, B]
                    A = build_edge_mask(B).to(device)  # [B, B]
                    attn_weights = compute_attention_weights(W, A, scale=10.0)  # softmax(10*W)
                    negative_embed = select_hard_negatives(attn_weights, anchor_embed)  # [B, D]

                # loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
                trip = triplet_loss(anchor_embed, positive_embed, negative_embed)

                l2_term = (anchor_embed.norm(dim=1).mean()
                        + positive_embed.norm(dim=1).mean()
                        + negative_embed.norm(dim=1).mean())
                embedding_l2 = 1e-2
                loss = trip + embedding_l2 * l2_term

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

        current_loss = running_loss / len(dataloader)
        if epoch >= warmup_epochs:
            if current_loss < best_loss - 0.001:  # significant drop
                best_loss = current_loss
                trigger_times = 0  # reset
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    torch.save(model.state_dict(), f'GeoVit_HNM_model_{city}.pth')
    print(f"Model saved to GeoVit_HNM_model_{city}.pth")
