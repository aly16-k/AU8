# -*- coding: utf-8 -*-
# Feature extraction + XGBoost prediction script for 68x68, 4-channel (B2,B3,B4,B8) 16-bit GeoTIFF (aligned with 4-channel training code)

import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import tifffile as tiff

# ============================
# 0. Reproducibility (global seed & determinism)
# ============================
SEED = 42

def set_seed(seed: int = 42, cuda_deterministic: bool = True):
    # Execute as early as possible
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by some CUDA ops (only when use_deterministic_algorithms=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN determinism (trade benchmark speed for reproducibility)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Stronger determinism (unsupported ops will throw exception; set False to skip)
    try:
        torch.use_deterministic_algorithms(cuda_deterministic)
    except Exception:
        pass

# Set seed first
set_seed(SEED)

# ---------------------------
# 1. Device and common configs
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 128

# === Channel normalization parameters consistent with training (replace with final stats from training) ===
MEAN = torch.tensor([0.12, 0.13, 0.14, 0.35], dtype=torch.float32).view(4, 1, 1)
STD  = torch.tensor([0.08, 0.08, 0.09, 0.15], dtype=torch.float32).view(4, 1, 1)

# ---------------------------
# 2. Read GeoTIFF into Tensor ([4,68,68], normalized same as training)
# ---------------------------
def read_s2_core4_tif(path):
    arr = tiff.imread(path)  # Expect (H,W,4) or (4,H,W)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = np.transpose(arr, (2, 0, 1))  # -> (4,H,W)
    elif arr.ndim == 3 and arr.shape[0] == 4:
        pass  # Already (4,H,W)
    else:
        raise ValueError(f"Unexpected TIFF shape {arr.shape} for {path}")

    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr / 10000.0, 0.0, 1.0)  # Same scaling as training (0-1)

    x = torch.from_numpy(arr)               # [4,H,W], float32
    # === Per-channel normalization consistent with training ===
    x = (x - MEAN) / (STD + 1e-6)
    return x

# ---------------------------
# 3. 4-channel CNN model consistent with training (with BatchNorm)
# ---------------------------
class Tile2VecModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=5, stride=2, padding=2),  # 68 -> 34
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),# 34 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)                                 # -> [B,256,1,1]
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.encoder(x).flatten(1)  # [B,256]
        return self.fc(x)               # [B,embedding_dim]

# ---------------------------
# 4. Feature extraction function (read .tif → forward to get embedding)
# ---------------------------
def extract_features(image_folder, df, image_column, model):
    features = []
    for img_name in tqdm(df[image_column], desc="Extracting features"):
        # Original path (could be .png)
        img_path = os.path.join(image_folder, img_name)

        # If not exist, try replacing with .tif
        if not os.path.exists(img_path):
            base, _ = os.path.splitext(img_name)
            tif_name = base + ".tif"
            img_path = os.path.join(image_folder, tif_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_name} or {tif_name}")

        # Read 4-channel tif
        x = read_s2_core4_tif(img_path).unsqueeze(0).to(device)  # [1,4,68,68]
        with torch.no_grad():
            feat = model(x)
        features.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
    return np.asarray(features, dtype=np.float32)

# ---------------------------
# 5. Data loop & evaluation
# ---------------------------
cities = ["Adelaide","Brisbane","Canberra","Darwin","Hobart","Melbourne","Perth","Sydney"]
variables = [
    "Population_density_persons_km2__2023_log",
    "2020_Number of jobs_mean_log",
    "2021_Total persons employed aged 15 years and over (no.)_mean_log",
    "Median_total_income_excl_Government_pensions_and_allowances__2020_log",
    "Total_number_of_businesses_2023_mean_log",
    "Median_price_of_established_house_transfers__2023_log",
    "Total_protected_land_area_ha__2022_mean_log",
    "2021_Area of agricultural land (ha)_mean_log",
    "2016_Rural residential and farm infrastructure (ha)_mean_log"
]
root = ["population","number_jobs","persons_employed","income","num_business","house",
        "protected_land","agricultural_land","rural_residential"]

MODEL_PATH_TEMPLATE = "tile2vec_model_core4_{city}.pth"
IMAGE_FOLDER_TEMPLATE = "data/images/{city}"   # Directory should contain .tif files
TRAIN_CSV_TEMPLATE = "data/train_test/{var}/{city}_train.csv"
TEST_CSV_TEMPLATE  = "data/train_test/{var}/{city}_test.csv"
IMAGE_COL = "image_name"

results = []

for city in cities:
    # Load 4-channel model structure consistent with training (with BatchNorm)
    model = Tile2VecModel(embedding_dim=embedding_dim).to(device)
    model_path = MODEL_PATH_TEMPLATE.format(city=city)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()   # VERY IMPORTANT: BN in eval mode uses training statistics

    for i, var in enumerate(root):
        print(f"\n--- Processing {city} - {var} ---")

        train_csv = TRAIN_CSV_TEMPLATE.format(var=var, city=city)
        test_csv  = TEST_CSV_TEMPLATE.format(var=var, city=city)
        image_folder = IMAGE_FOLDER_TEMPLATE.format(city=city)
        y_name = variables[i]

        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)

        # Extract features (read .tif directly, normalization inside read_s2_core4_tif)
        X_train = extract_features(image_folder, train_df, image_column=IMAGE_COL, model=model)
        y_train = train_df[y_name].values.astype(np.float32)

        X_test  = extract_features(image_folder, test_df,  image_column=IMAGE_COL, model=model)
        y_test  = test_df[y_name].values.astype(np.float32)

        # ---------------------------
        # 6. Train & evaluate XGBoost (fixed randomness)
        # ---------------------------
        regressor = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,   # Fix randomness
            n_jobs=1,            # Reduce multithread nondeterminism
            tree_method="hist",  # xgboost>=2.0, with device="cuda" uses GPU
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE:  {mae:.4f}")
        print(f"Test R2:   {r2:.4f}")

        results.append({
            "city": city,
            "variable_name": var,
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae)
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("tile2vec_results.csv", index=False)
print("All results saved to tile2vec_results.csv")
print(f"Total rows saved: {len(results_df)}")
