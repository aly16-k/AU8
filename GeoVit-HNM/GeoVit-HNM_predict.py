
# ---------------------------
# 1. Import required libraries
# ---------------------------
import os
import random
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import xgboost as xgb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import tifffile as tiff

# ---------------------------
# 1.1 Reproducibility (全局种子与确定性)
# ---------------------------
SEED = 42

def set_seed(seed: int = 42, cuda_deterministic: bool = True):
    # 尽量在脚本最早执行
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 对某些 CUDA 算子需要该环境变量（仅在 use_deterministic_algorithms=True 时）
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN 确定性（放弃 benchmark）
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 更强的确定性（不支持的算子会抛异常；若不想抛异常可传 False 或 try/except）
    try:
        torch.use_deterministic_algorithms(cuda_deterministic)
    except Exception:
        pass

# ---------------------------
# 2. Hybrid CNN-ViT Model (对齐训练时的结构与超参)
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

class Tile2VecHybrid(nn.Module):
    """
    与训练脚本对齐：
    - 四通道输入（B2,B3,B4,B8）
    - cnn_stem: 68 -> 34 -> 17 -> 16（最后一层 k=2, s=1, p=0）
    - patch_size=4 → 16×16 → 4*4=16 patch；+CLS 共 17 tokens
    """
    def __init__(self, image_size=68, patch_size=4, dim=256, depth=4, heads=4,
                 mlp_dim=512, dim_head=64, emb_dropout=0.1, dropout=0.1, embedding_dim=128):
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
        feature_size = image_size // 4  # 17（用于 pos_embedding 上限；真实边长为 16）
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),  # -> (B,16,256)
            nn.LayerNorm(dim * patch_size * patch_size),   # 256 * 4 * 4 = 4096
            nn.Linear(dim * patch_size * patch_size, dim), # 4096 -> 256
            nn.LayerNorm(dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, (feature_size // patch_size) ** 2 + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.embedding = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, embedding_dim))
    def forward(self, x):
        x = self.cnn_stem(x)                 # (B, 256, 16, 16)
        x = self.to_patch_embedding(x)       # (B, 16, 256)
        b, n, _ = x.shape                    # n=16
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 17, 256)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.embedding(x[:, 0])        # (B, embedding_dim)

# ---------------------------
# 3. Device & embedding dim
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 128

# ---------------------------
# 4. 影像读取（与训练完全一致的归一化/尺寸/通道）
# ---------------------------
def _load_img_as_tensor_infer(path):
    """
    - GeoTIFF (uint16, S2 SR): /10000 → clamp[0,1]，保证 (4,H,W)，尺寸 68×68
    - PNG/JPG 兼容：RGB/255，并补齐到 4 通道；尺寸 68×68
    """
    if path.lower().endswith(('.tif', '.tiff')):
        arr = tiff.imread(path)  # (H,W,C) 或 (C,H,W)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] != arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))  # -> (H,W,C)
        img = torch.from_numpy(arr.astype(np.float32)) / 10000.0
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img.permute(2, 0, 1)  # -> (C,H,W)
        # 通道容错：截断/补零到4
        if img.shape[0] > 4:
            img = img[:4]
        elif img.shape[0] < 4:
            pad = torch.zeros((4 - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
            img = torch.cat([img, pad], dim=0)
        img = img.clamp(0.0, 1.0)
        # 尺寸对齐到 68×68（若已是 68 就不变）
        if img.shape[1] != 68 or img.shape[2] != 68:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(68, 68), mode='bilinear', align_corners=False
            ).squeeze(0)
        return img
    else:
        pil = Image.open(path).convert('RGB').resize((68, 68))
        np_img = np.array(pil).astype(np.float32) / 255.0
        img = torch.from_numpy(np_img).permute(2, 0, 1)  # (3,68,68)
        # 补一个零通道到4
        pad = torch.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype)
        img = torch.cat([img, pad], dim=0)  # (4,68,68)
        return img

def extract_features(image_folder, df, image_column, model):
    features = []
    for img_name in tqdm(df[image_column], desc="Extracting features"):
        # 自动替换掉后缀
        base, ext = os.path.splitext(img_name)
        if ext.lower() in ['.png', '.jpg', '.jpeg']:
            img_name = base + '.tif'
        img_path = os.path.join(image_folder, img_name)

        img = _load_img_as_tensor_infer(img_path).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img)
        features.append(feat.cpu().numpy().astype(np.float32).flatten())  # 明确为 float32
    return np.array(features, dtype=np.float32)  # XGBoost 输入统一 float32

# ---------------------------
# 4.1 构造 GPU + 尽量确定性的 XGBoost 回归器
# ---------------------------
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
        # v2.x: 只用 device="cuda" + tree_method="hist"
        return XGBRegressor(
            **base,
            device="cuda",
            tree_method="hist"
        )


# ---------------------------
# 5. Load data, extract features, XGBoost, evaluate
# ---------------------------
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
        # 构建模型（与训练一致）
        model = Tile2VecHybrid(embedding_dim=embedding_dim).to(device)
        model.eval()

        # 权重加载：优先城市专用，其次通用
        ckpt_path_city = f'GeoVit_HNM_model_{city}.pth'
        if os.path.exists(ckpt_path_city):
            ckpt_path = ckpt_path_city
        else:
            raise FileNotFoundError(
                f"未找到权重文件：{ckpt_path_city}。请确认保存的 pth 路径。"
            )
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"[Info] Loaded weights from: {ckpt_path}")
        print(f"[Info] xgboost version: {xgb.__version__}")

        for i in range(len(root)):  # i = 0 ~ 8
            var = root[i]
            print(f"\n--- Processing {city} - {var} ---")

            train_csv = f'data/train_test/{root[i]}/{city}_train.csv'
            test_csv  = f'data/train_test/{root[i]}/{city}_test.csv'
            image_folder = f'data/images/{city}'
            y_name = variables[i]

            train_df = pd.read_csv(train_csv)
            test_df  = pd.read_csv(test_csv)

            # 提取特征（float32）
            X_train = extract_features(image_folder, train_df, image_column='image_name', model=model)
            y_train = train_df[y_name].values.astype(np.float32)

            X_test  = extract_features(image_folder, test_df, image_column='image_name', model=model)
            y_test  = test_df[y_name].values.astype(np.float32)

            # 6. XGBoost：GPU + 尽量确定性设置（按版本自动选择）
            regressor = make_xgb_gpu_regressor(seed=SEED)

            regressor.fit(X_train, y_train)

            # 7. Evaluation
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
                "variable_name": var,
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae)
            })

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv("HNM_results.csv", index=False)
    print("All results saved to HNM_results.csv")
    print(f"Total rows saved: {len(results_df)}")

