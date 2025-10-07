# -*- coding: utf-8 -*-
# UrbanCLIP (CoCa-ViT-L-14) downstream regression with XGBoost (seed/stability preserved)

import os
import random
import pandas as pd
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import xgboost as xgb

import tifffile as tiff
import open_clip  # UrbanCLIP backbone (CoCa-ViT-L-14)

# ---------------------------
# 0) Reproducibility
# ---------------------------
SEED = 42

def set_seed(seed: int = 42, cuda_deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(cuda_deterministic)
    except Exception:
        pass

# ---------------------------
# 1) Device & model loader
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

URBANCLIP_NAME = "coca_ViT-L-14"               # consistent with training

def load_urbanclip_model(ckpt_path: str):
    """
    Exactly the same inference loading logic as in main.py:
    - First create CoCa-ViT-L-14 and transform (pretrained=None)
    - Then explicitly load best_model.bin using torch.load(state_dict) + model.load_state_dict(...)
    """
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=URBANCLIP_NAME,
        pretrained=None  # do not pass best.bin here, explicitly load as in main.py
    )
    state = torch.load(ckpt_path, map_location="cpu")  # best_model.bin is a state_dict
    model.load_state_dict(state)                       # explicit loading
    model.to(device)
    model.eval()
    return model, transform


# ---------------------------
# 2) Image I/O for UrbanCLIP (match utils.CoCaDataset exactly)
# ---------------------------
def to8_fixed(x16, vmin=0, vmax=10000):
    x = x16.astype(np.float32)
    x = (x - vmin) / (vmax - vmin)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)

def _tif_to_pil_rgb(path: str) -> Image.Image:
    """
    Assume input is always Sentinel-2 SR 4-channel GeoTIFF, channel order [B2, B3, B4, B8], 16-bit.
    Steps:
      - If needed: CHW -> HWC
      - Extract true color: RGB = [B4, B3, B2] = indices [2,1,0]
      - Linearly map to 8-bit and return PIL.Image(RGB)
    """
    arr = tiff.imread(path)  # (H,W,4) or (4,H,W)
    # If channels are first, convert to HWC
    if arr.ndim == 3 and arr.shape[0] == 4 and (arr.shape[-1] != 4):
        arr = np.transpose(arr, (1, 2, 0))  # -> (H, W, 4)

    # Strong check: must be HWC and 4 channels
    assert arr.ndim == 3 and arr.shape[2] == 4, f"Expected HWC 4-channel, got {arr.shape}"

    # True color: B4,B3,B2 -> R,G,B
    rgb16 = arr[:, :, [2, 1, 0]]  # [B4,B3,B2]
    rgb8  = to8_fixed(rgb16, vmin=0, vmax=10000)
    return Image.fromarray(rgb8, mode="RGB")

def _load_pil_rgb(path: str) -> Image.Image:
    # Data is guaranteed to be .tif/.tiff
    return _tif_to_pil_rgb(path)


# ---------------------------
# 3) Feature extraction (UrbanCLIP)
# ---------------------------
@torch.no_grad()
def extract_uc_features(image_folder, df, image_column, model, transform):
    feats = []
    for img_name in tqdm(df[image_column], desc="Extracting UrbanCLIP features"):
        base, ext = os.path.splitext(img_name)
        # Allow csv to contain .png/.jpg: replace with .tif first (if your data is tiff)
        if ext.lower() in [".png", ".jpg", ".jpeg"]:
            tif_try = os.path.join(image_folder, base + ".tif")
            if os.path.exists(tif_try):
                img_path = tif_try
            else:
                img_path = os.path.join(image_folder, img_name)
        else:
            img_path = os.path.join(image_folder, img_name)

        pil = _load_pil_rgb(img_path)
        img = transform(pil).unsqueeze(0).to(device)  # [1,3,H,W]
        feat = model.encode_image(img)                # [1, D]
        feats.append(feat.cpu().numpy().astype(np.float32).squeeze(0))
    return np.stack(feats, axis=0).astype(np.float32)

# ---------------------------
# 4) XGBoost (GPU, deterministic as possible)
# ---------------------------
def make_xgb_gpu_regressor(seed: int = SEED):
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
        return XGBRegressor(**base, device="cuda" if torch.cuda.is_available() else "cpu", tree_method="hist")


# ---------------------------
# 5) Main: per-city, per-variable
# ---------------------------
if __name__ == "__main__":
    set_seed(SEED)

    # You can extend:
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
        # Load UrbanCLIP
        URBANCLIP_CKPT = f"checkpoints/best_model_{city}.bin"   # <- your UrbanCLIP training output weights path
        model, transform = load_urbanclip_model(URBANCLIP_CKPT)
        print(f"[Info] Loaded UrbanCLIP weights from: {URBANCLIP_CKPT}")
        print(f"[Info] xgboost version: {xgb.__version__}")
        for i, var_key in enumerate(root):
            print(f"\n--- Processing {city} - {var_key} ---")

            train_csv = f'data/train_test/{var_key}/{city}_train.csv'
            test_csv  = f'data/train_test/{var_key}/{city}_test.csv'
            image_folder = f'data/images/{city}'
            y_name = variables[i]

            train_df = pd.read_csv(train_csv)
            test_df  = pd.read_csv(test_csv)

            # Feature extraction (UrbanCLIP)
            X_train = extract_uc_features(image_folder, train_df, image_column='image_name', model=model, transform=transform)
            y_train = train_df[y_name].values.astype(np.float32)

            X_test  = extract_uc_features(image_folder, test_df,  image_column='image_name', model=model, transform=transform)
            y_test  = test_df[y_name].values.astype(np.float32)

            # XGBoost: keep consistent with your previous setup
            regressor = make_xgb_gpu_regressor(seed=SEED)
            regressor.fit(X_train, y_train)

            # Evaluation
            y_pred = regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"Test RMSE: {rmse:.4f}")
            print(f"Test MAE:  {mae:.4f}")
            print(f"Test R2:   {r2:.4f}")

            results.append({
                "city": city,
                "variable_name": var_key,
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae)
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("Urbanclip_results.csv", index=False)
    print("All results saved to Urbanclip_results.csv")
    print(f"Total rows saved: {len(results_df)}")
