# -*- coding: utf-8 -*-
import os
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

import tifffile as tiff  # Read .tif
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor  # For regressor function return

# =======================
# Global parameters
# =======================
SEED = 42  # image_folder=f"{BASE_IMG_ROOT}/{city}_UTM"
ALLOW_MISSING = True         # Whether to skip missing images in CSV
IMG_SCALE = 10000.0          # 16-bit reflectance common scaling (adjust according to your data)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Provided regressor (kept unchanged)
# =======================
def make_xgb_gpu_regressor(seed: int = SEED):
    import xgboost as xgb
    ver_major, ver_minor = map(int, xgb.__version__.split('.')[:2])
    base = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
    )
    if (ver_major, ver_minor) >= (2, 0):
        # v2.x: use only device="cuda" + tree_method="hist"
        return XGBRegressor(
            **base,
            device="cuda",
            tree_method="hist"
        )

# =======================
# Read 4-channel TIF as tensor (B2,B3,B4,B8)
# =======================
# First 3 channels use ImageNet stats, the 4th channel gets a neutral value (adjust if necessary)
_mean4 = torch.tensor([0.485, 0.456, 0.406, 0.5], dtype=torch.float32).view(4, 1, 1)
_std4  = torch.tensor([0.229, 0.224, 0.225, 0.5], dtype=torch.float32).view(4, 1, 1)

def load_tif_as_tensor_4ch(path: str) -> torch.Tensor:
    """
    Read 16-bit 4-channel TIF (assume channel order is B2,B3,B4,B8), return [4,224,224] float32 normalized tensor.
    Processing steps:
    - tifffile read -> (H,W,C); if (C,H,W), transpose automatically
    - / IMG_SCALE to map into [0,1]
    - Take first 4 channels (B2,B3,B4,B8)
    - Bilinear resize to 224x224
    - Per-channel normalization (_mean4/_std4)
    """
    arr = tiff.imread(path)
    if arr.ndim == 2:
        arr = arr[..., None]  # (H,W,1)
    elif arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))  # (C,H,W) -> (H,W,C)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape {arr.shape} for {path}")

    arr = arr.astype(np.float32) / IMG_SCALE
    if arr.shape[2] < 4:
        raise ValueError(f"Expect 4-channel tif, got shape {arr.shape} at {path}")

    # Take first 4 channels as B2,B3,B4,B8; adjust order here if your file differs
    arr4 = arr[:, :, :4]  # (H,W,4)

    ten = torch.from_numpy(arr4).permute(2, 0, 1)  # [4,H,W], float32 in [0,1]
    # Resize to 224×224 (same as ResNet input)
    ten = F.interpolate(ten.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    # Normalize
    ten = (ten - _mean4) / _std4
    return ten

# =======================
# ResNet18 modified for 4-channel input (Option A)
# =======================
# Use pretrained weights and change conv1 from 3ch to 4ch; 4th channel weights = mean of RGB
weights = ResNet18_Weights.DEFAULT
resnet = resnet18(weights=weights)

w = resnet.conv1.weight  # [64,3,7,7]
resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    resnet.conv1.weight[:, :3] = w
    resnet.conv1.weight[:, 3:4] = w.mean(1, keepdim=True)  # Initialize 4th channel with mean

resnet.fc = nn.Identity()  # Output 512 dims
for p in resnet.parameters():
    p.requires_grad = False
resnet.eval().to(device)

# =======================
# Dataset and feature extraction
# =======================
class ImgDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_folder: str, y_col: str, include_target: bool):
        self.df = df
        self.image_folder = image_folder
        self.y_col = y_col
        self.include_target = include_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["image_name"]
        # If CSV stores .png, automatically replace with .tif
        if fname.lower().endswith(".png"):
            fname = os.path.splitext(fname)[0] + ".tif"
        img_path = os.path.join(self.image_folder, fname)
        if not os.path.exists(img_path):
            if ALLOW_MISSING:
                raise KeyError(f"missing:{img_path}")
            else:
                raise FileNotFoundError(f"Missing image: {img_path}")

        # Read directly as 4-channel tensor ([4,224,224]), bypass PIL/weights.transforms
        tensor = load_tif_as_tensor_4ch(img_path)

        if self.include_target:
            y = float(row[self.y_col])
            return tensor, y, row["image_name"]
        else:
            return tensor, row["image_name"]

def extract_resnet_features(df: pd.DataFrame, image_folder: str, y_col: str,
                            include_target: bool, batch_size: int = 32, num_workers: int = 4):
    ds = ImgDataset(df, image_folder, y_col, include_target)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    feats_list, y_list, name_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Feat@{os.path.basename(image_folder)}"):
            try:
                if include_target:
                    imgs, ys, names = batch
                else:
                    imgs, names = batch
                    ys = None
            except Exception:
                continue

            imgs = imgs.to(device, non_blocking=True)  # [B,4,224,224]
            vec = resnet(imgs).cpu().numpy()          # (B,512)
            feats_list.append(vec)
            name_list.extend(list(names))
            if include_target:
                y_list.extend([float(v) for v in ys])

    X = np.vstack(feats_list).astype(np.float32) if len(feats_list) else np.zeros((0, 512), dtype=np.float32)
    y_arr = np.asarray(y_list, dtype=np.float32) if include_target else None
    return X, y_arr, name_list

# =======================
# Main loop: City × Indicator
# =======================
if __name__ == "__main__":
    set_seed(SEED)

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
    root = ["population","number_jobs","persons_employed","income","num_business",
            "house","protected_land","agricultural_land","rural_residential"]

    results = []

    for city in cities:
        for i, var_root in enumerate(root):
            y_name = variables[i]
            print(f"\n=== Processing: City = {city} | Target = {var_root} ===")

            train_csv = f'data/train_test/{var_root}/{city}_train.csv'
            test_csv  = f'data/train_test/{var_root}/{city}_test.csv'
            image_folder = f'data/images/{city}'

            # 1) Read CSV
            train_df = pd.read_csv(train_csv)
            test_df  = pd.read_csv(test_csv)

            # 2) Extract ResNet18 features (4-channel input)
            X_train, y_train, _ = extract_resnet_features(train_df, image_folder, y_name, include_target=True)
            X_test,  _, _ = extract_resnet_features(test_df,  image_folder, y_name, include_target=False)

            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                warnings.warn(f"[{city}-{var_root}] Empty training or testing samples, skipped.")
                continue

            print(f"Feature shape: train={X_train.shape}, test={X_test.shape}")

            # 3) Downstream regression: use the provided XGBoost GPU regressor (function unchanged)
            regressor = make_xgb_gpu_regressor(SEED)
            regressor.fit(X_train, y_train)

            # 4) Evaluation
            y_pred = regressor.predict(X_test)
            if y_name in test_df.columns:
                y_test = test_df[y_name].to_numpy(dtype=np.float32)
                mse = mean_squared_error(y_test, y_pred)
                rmse = float(np.sqrt(mse))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            else:
                rmse = mae = r2 = np.nan

            print(f"[Test] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")

            # 5) Record results
            results.append({
                "city": city,
                "variable_name": var_root,
                "r2": float(r2) if r2 == r2 else None,
                "rmse": float(rmse) if rmse == rmse else None,
                "mae": float(mae) if mae == mae else None,
            })

    # 6) Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("resnet18_results.csv", index=False)
    print("\nAll results saved to resnet18_results.csv")
    print(f"Total rows saved: {len(results_df)}")
