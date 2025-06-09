import math
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

import torch
from torch.utils.data import DataLoader
import open_clip
from utils import (
    LinearProbDataset,
    set_random_seed,
)
from tqdm import tqdm
from loguru import logger


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Adelaide")
    parser.add_argument("--test_file", type=str, default="./data/downstream_task/Adelaide_test.csv")
    parser.add_argument("--indicator", type=str, default="Median_price_of_established_house_transfers__2023_log")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrained_model", type=str, default="./checkpoints/best_model.bin")
    parser.add_argument("--img_embedding_dim", type=int, default=768)
    parser.add_argument("--seed", type=int, default=132)
    parser.add_argument("--logging_dir", type=str, default="logs/downtask1")
    parser.add_argument("--train_dataset_ratio", type=float, default=0.8)
    args = parser.parse_args()
    return args


def extract_features(dataset, model, device):
    model.eval()
    all_features, all_labels = [], []
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.encode_image(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_features), np.hstack(all_labels)


def main():
    args = create_args()
    set_random_seed(args.seed)

    print("✅ Initializing parameters and logging...")
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir)
    logger.remove()
    logger.add(os.path.join(args.logging_dir, str(args.seed) + ".log"), level="INFO")
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Loading CoCa model...")
    start_time = time.time()
    coca_model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14", pretrained=args.pretrained_model
    )
    coca_model.to(device)
    print(f"✅ CoCa model loaded! Time taken: {time.time() - start_time:.2f} seconds")

    print("✅ Loading dataset...")
    data = pd.read_csv(f"data/downstream_task/{args.dataset}_train.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    train_data = data[: int(len(data) * args.train_dataset_ratio)].reset_index(drop=True)
    val_data = data[int(len(data) * args.train_dataset_ratio):].reset_index(drop=True)

    mean = np.mean(train_data[args.indicator])
    std = np.std(train_data[args.indicator])

    train_dataset = LinearProbDataset(args.dataset, train_data, args.indicator, transform, mean, std, False)
    val_dataset = LinearProbDataset(args.dataset, val_data, args.indicator, transform, mean, std, False)
    test_data = pd.read_csv(args.test_file)
    test_dataset = LinearProbDataset(args.dataset, test_data, args.indicator, transform, mean, std, True)
    print("✅ Dataset loaded!")

    print("✅ Extracting image features...")
    X_train, y_train = extract_features(train_dataset, coca_model, device)
    X_val, y_val = extract_features(val_dataset, coca_model, device)
    X_test, y_test = extract_features(test_dataset, coca_model, device)
    print("✅ Feature extraction completed!")

    print("✅ Performing GridSearchCV for XGBoost...")
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.005, 0.01],
        'subsample': [0.4, 0.6],
        'colsample_bytree': [0.4, 0.6],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 5]
    }

    grid_search = GridSearchCV(
        estimator=XGBRegressor(
            n_estimators=1000,
            tree_method='hist',
            eval_metric="rmse",
            device='cuda',
            random_state=42
        ),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    print("✅ Training XGBoost with best parameters and early stopping...")
    print("📌 Best parameter combination:", best_params)

    regressor = XGBRegressor(
        **best_params,
        n_estimators=1000,
        tree_method='hist',
        eval_metric="rmse",
        device='cuda',
        random_state=42,
        early_stopping_rounds=50
    )

    regressor.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Save model
    model_save_dir = os.path.join("checkpoints", "downtask2")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"best_{args.indicator}.json")
    regressor.save_model(model_save_path)
    print(f"✅ XGBoost model saved to: {model_save_path}")

    print("✅ Evaluating model performance on validation set...")
    val_pred = regressor.predict(X_val) * std + mean
    y_val_real = y_val * std + mean

    val_r2 = r2_score(y_val_real, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_real, val_pred))
    val_mae = mean_absolute_error(y_val_real, val_pred)

    logger.info(f"Validation R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
    print(f"📊 Validation performance: R² = {val_r2:.4f}, RMSE = {val_rmse:.4f}, MAE = {val_mae:.4f}")

    print("✅ Predicting test set and saving results...")
    pred_dir = os.path.join("data", "downstream_task", f"{args.dataset}_urbxg")
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, os.path.basename(args.test_file).replace(".csv", "_predicted.csv"))
    test_pred = regressor.predict(X_test) * std + mean

    test_df = pd.read_csv(args.test_file)
    test_df[args.indicator + "_predict"] = test_pred
    test_df.to_csv(pred_path, index=False)
    print(f"✅ Test set prediction saved to: {pred_path}")
    y_test_real = test_data[args.indicator].values
    test_r2 = r2_score(y_test_real, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
    test_mae = mean_absolute_error(y_test_real, test_pred)

    # Save evaluation metrics
    eval_path = os.path.join("data", "downstream_task", f"{args.dataset}_evaluation_metrics.csv")
    result_row = pd.DataFrame([{
        "target_variable": args.indicator,
        "R2": round(test_r2, 4),
        "RMSE": round(test_rmse, 4),
        "MAE": round(test_mae, 4)
    }])
    if os.path.exists(eval_path):
        existing = pd.read_csv(eval_path)
        existing = existing[existing["target_variable"] != args.indicator]
        combined = pd.concat([existing, result_row], ignore_index=True)
    else:
        combined = result_row
    combined.to_csv(eval_path, index=False)
    print(f"📄 Evaluation metrics written to: {eval_path}")


if __name__ == "__main__":
    main()