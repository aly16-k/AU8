# -*- coding: utf-8 -*-
# PCA + your provided XGBoost GPU regressor (function unchanged), run full pipeline per city × indicator and save results

import os
import warnings
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# =======================
# Global parameters
# =======================
SEED = 42
PCA_VARIANCE = 0.90         # Retain 90% variance
IMG_SCALE = 10000.0         # Common scaling for 16-bit reflectance
ALLOW_MISSING = True        # Whether to skip if CSV has missing images

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
# Utility functions
# =======================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def safe_read_tif(img_path: str) -> np.ndarray:
    """Read GeoTIFF, return (H, W, C) float32, apply simple normalization. Compatible with (C,H,W) and (H,W,C)."""
    arr = tiff.imread(img_path)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3:
        # Case like (C,H,W) with small C (≤8), convert to (H,W,C)
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported TIF shape: {arr.shape}")
    arr = arr.astype(np.float32) / IMG_SCALE
    return arr

def df_standardize_id(df: pd.DataFrame) -> pd.DataFrame:
    """Unify image_name column to 'id' column."""
    if "id" in df.columns:
        return df
    if "image_name" in df.columns:
        return df.rename(columns={"image_name": "id"})
    raise KeyError("CSV is missing 'id' or 'image_name' column.")

def build_Xy_from_df(df: pd.DataFrame, image_folder: str, y_name: str):
    """Map DataFrame IDs to images, read and flatten them; return X(np.ndarray[N,D]) and y(np.ndarray[N])."""
    df = df_standardize_id(df)
    X_list, y_list, missed = [], [], 0

    for fname, yval in tqdm(zip(df["id"], df[y_name]), total=len(df),
                            desc=f"Reading images {os.path.basename(image_folder)}"):
        # If suffix is .png, automatically switch to .tif
        if fname.lower().endswith(".png"):
            fname = os.path.splitext(fname)[0] + ".tif"

        img_path = os.path.join(image_folder, fname)
        if not os.path.exists(img_path):
            missed += 1
            if not ALLOW_MISSING:
                raise FileNotFoundError(f"Missing image: {img_path}")
            continue

        img = safe_read_tif(img_path)
        X_list.append(img.reshape(-1))
        y_list.append(float(yval))

    if missed > 0:
        warnings.warn(f"[{image_folder}] {missed} images listed in CSV not found, skipped.")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y

# =======================
# Main pipeline
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

            # 2) Build raw features/labels
            X_train_raw, y_train = build_Xy_from_df(train_df, image_folder, y_name)
            X_test_raw,  y_test  = build_Xy_from_df(test_df,  image_folder, y_name)

            if len(X_train_raw) == 0 or len(X_test_raw) == 0:
                warnings.warn(f"[{city}-{var_root}] Training or testing samples are empty, skipped.")
                continue

            print(f"Original dimension: train={X_train_raw.shape}, test={X_test_raw.shape}")

            # 3) PCA: fit only on training set, then transform both training/testing (avoid leakage)
            pca = PCA(n_components=PCA_VARIANCE, svd_solver="auto", random_state=SEED)
            X_train = pca.fit_transform(X_train_raw)
            X_test  = pca.transform(X_test_raw)

            kept_var = float(np.sum(pca.explained_variance_ratio_))
            print(f"After PCA: train={X_train.shape}, test={X_test.shape} | Variance retained≈{kept_var:.3f}")

            # 4) Train & evaluate: using the provided XGBoost GPU regressor
            regressor = make_xgb_gpu_regressor(SEED)
            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"[Test] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")

            # 5) Record results
            results.append({
                "city": city,
                "variable_name": var_root,
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
            })

    # 6) Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("PCA_results.csv", index=False)
    print("\nAll results saved to PCA_results.csv")
    print(f"Total rows saved: {len(results_df)}")
