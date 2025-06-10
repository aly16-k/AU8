import os
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
import random
from xgboost import XGBRegressor

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)  

# ==== Parameters and Paths ====
image_dir = "data/images/Adelaide"
train_csv = "data/downstream_task/Adelaide_train.csv"
test_csv = "data/downstream_task/Adelaide_test.csv"
target_variable = "Median_price_of_established_house_transfers__2023_log"  

# ==== Load Data ====
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# ==== Image Preprocessing ====
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==== Load Pretrained ResNet18 (for feature extraction) ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()
resnet18.to(device)
resnet18.eval()

# ==== Custom Dataset ====
class SatelliteDataset(Dataset):
    def __init__(self, df, image_dir, transform, include_target=True):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.include_target = include_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        name = row["image_name"]
        if self.include_target:
            target = row[target_variable]
            return tensor, target, name
        else:
            return tensor, name

# ==== Extract Features in Batches ====
def extract_features_batch(df, include_target=True, batch_size=64):
    dataset = SatelliteDataset(df, image_dir, preprocess, include_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features, targets, names = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Features"):
            if include_target:
                imgs, ys, ns = batch
            else:
                imgs, ns = batch
                ys = None
            imgs = imgs.to(device)
            feats = resnet18(imgs).cpu().numpy()
            features.append(feats)
            names.extend(ns)
            if include_target:
                targets.extend(ys.numpy())

    features = np.vstack(features)
    if include_target:
        return features, np.array(targets), names
    else:
        return features, names

if __name__ == '__main__':
    # ==== Extract Features for Train and Test Sets ====
    X_train, y_train, _ = extract_features_batch(train_df, include_target=True)
    X_test, test_image_names = extract_features_batch(test_df, include_target=False)
    
    # ==== XGBoost Regressor ====
    
    regressor = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    regressor.fit(X_train, y_train)
    
    # ==== Prediction ====
    y_pred = regressor.predict(X_test)
    
    # ==== Save Results ====
    result_df = pd.DataFrame({
        "image_name": test_image_names,
        "y_prediction": y_pred
    })
    
    # Merge with ground truth if available
    if target_variable in test_df.columns:
        result_df = result_df.merge(test_df[["image_name", target_variable]], on="image_name", how="left")
        result_df.rename(columns={target_variable: "y_true"}, inplace=True)
    
    # Save CSV
    output_path = os.path.join("data", "downstream_task", "Adelaide_res", f"predicted_test_{target_variable}.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ Prediction results saved to: {output_path}")
    
    # ==== Evaluation ====
    if "y_true" in result_df.columns:
        y_true = result_df["y_true"].values
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
    
        print("\n📊 Evaluation Metrics:")
        print(f"R² Score  : {r2:.4f}")
        print(f"RMSE      : {rmse:.4f}")
        print(f"MAE       : {mae:.4f}")
    
        # ==== Save Evaluation Results ====
        eval_file = os.path.join("data", "downstream_task", "Adelaide_evaluation_metrics.csv")
        result_row = pd.DataFrame([{
            "target_variable": target_variable,
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        }])
    
        if os.path.exists(eval_file):
            existing = pd.read_csv(eval_file)
            # Update existing or append new
            existing = existing[existing["target_variable"] != target_variable]
            combined = pd.concat([existing, result_row], ignore_index=True)
        else:
            combined = result_row
    
        combined.to_csv(eval_file, index=False)
        print(f"📄 Evaluation results saved to: {eval_file}")

