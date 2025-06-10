# ---------------------------
# 1. Import required libraries
# ---------------------------
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ---------------------------
# 2. Define the Tile2Vec model (consistent with training)
# ---------------------------
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

# ---------------------------
# 3. Load the pre-trained Tile2Vec model
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 128

model = Tile2VecModel(embedding_dim=embedding_dim).to(device)
model.load_state_dict(torch.load('tile2vec_model.pth', map_location=device))
model.eval()

# ---------------------------
# 4. Feature extraction function
# ---------------------------
# ---------------------------
# 4. Feature extraction function (batch version)
# ---------------------------
class Tile2VecDataset(Dataset):
    def __init__(self, df, image_folder, image_column):
        self.df = df
        self.image_folder = image_folder
        self.image_column = image_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][self.image_column]
        img_path = os.path.join(self.image_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((640, 640))
        img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, img_name

def extract_features(image_folder, df, image_column, batch_size=64):
    dataset = Tile2VecDataset(df, image_folder, image_column)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    features = []
    model.eval()
    with torch.no_grad():
        for batch_imgs, _ in tqdm(dataloader, desc="Extracting features"):
            batch_imgs = batch_imgs.to(device)
            batch_feats = model(batch_imgs)
            features.append(batch_feats.cpu().numpy())
    return np.vstack(features)


# ---------------------------
# 5. Load data and extract features
# ---------------------------
train_csv = 'data/downstream_task/Adelaide_train.csv'
test_csv = 'data/downstream_task/Adelaide_test.csv'
image_folder = 'data/images/Adelaide'
y_name = 'Median_price_of_established_house_transfers__2023_log'

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

X_train = extract_features(image_folder, train_df, image_column='image_name')
y_train = train_df[y_name].values

X_test = extract_features(image_folder, test_df, image_column='image_name')
y_test = test_df[y_name].values

# ---------------------------
# 6. Train the XGBoost regressor
# ---------------------------
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

# ---------------------------
# 7. Prediction and evaluation
# ---------------------------
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R2: {r2:.4f}")

# ---------------------------
# 8. Save prediction results
# ---------------------------
output_df = pd.DataFrame({
    'image_name': test_df['image_name'],
    'y_true': y_test,
    'y_pred': y_pred
})
output_df.to_csv('Adelaide_prediction_tile2vec_xgb.csv', index=False)
print("Prediction results saved")

