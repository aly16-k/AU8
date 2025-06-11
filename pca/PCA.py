import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# =======================
# Your parameters
# =======================
# Target column name for prediction
target_col = "Median_price_of_established_house_transfers__2023_log"

# Automatically identify city list from current directory
all_files = os.listdir("data/")
cities = sorted(list(set(f.split("_")[0] for f in all_files if f.endswith("_train.csv"))))

# To store results for each city
results = []

# =======================
# Iterate through each city
# =======================
for city in tqdm(cities, desc="Iterating over cities"):
    print(f"\n🚀 Processing {city} ...")

    try:
        # Load training, test, and feature files
        train_df = pd.read_csv(f"data/{city}_train.csv")
        test_df = pd.read_csv(f"data/{city}_test.csv")
        feature_df = pd.read_csv(f"data/features/{city}_features.csv")

        # Standardize column name: change image_name to id for merging
        if 'image_name' in train_df.columns:
            train_df = train_df.rename(columns={'image_name': 'id'})
        if 'image_name' in test_df.columns:
            test_df = test_df.rename(columns={'image_name': 'id'})

        # Check for id column
        if 'id' not in train_df.columns or 'id' not in test_df.columns:
            print(f"❌ {city} train/test missing 'id' column, skipping!")
            continue
        if 'id' not in feature_df.columns:
            print(f"❌ {city} feature file missing 'id' column, skipping!")
            continue

        # Merge training set with features
        train_merged = pd.merge(train_df, feature_df, on="id", how="inner")
        test_merged = pd.merge(test_df, feature_df, on="id", how="inner")

        # Extract feature columns (excluding id)
        feature_cols = [col for col in feature_df.columns if col != "id"]

        X_train = train_merged[feature_cols].values
        y_train = train_merged[target_col].values

        X_test = test_merged[feature_cols].values
        y_test = test_merged[target_col].values

        # =========================
        # PCA for dimensionality reduction (retain 90% variance)
        # =========================
        pca = PCA(n_components=0.9)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        print(f"📏 {city}: PCA reduced to {X_train_pca.shape[1]} dimensions (90% variance retained)")

        # =========================
        # Train linear regression
        # =========================
        model = LinearRegression()
        model.fit(X_train_pca, y_train)

        # Training prediction
        y_train_pred = model.predict(X_train_pca)
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

        # Test prediction
        y_test_pred = model.predict(X_test_pca)
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

        # =========================
        # Save result for each city
        # =========================
        results.append({
            "City": city,
            "Train_R2": r2_train,
            "Train_MAE": mae_train,
            "Train_RMSE": rmse_train,
            "Test_R2": r2_test,
            "Test_MAE": mae_test,
            "Test_RMSE": rmse_test
        })

    except Exception as e:
        print(f"⚠️ Error processing {city}, error: {e}")
        continue

# =======================
# After all cities, save summary
# =======================
results_df = pd.DataFrame(results)
print("\n🎯 Summary results for all cities:")
print(results_df)

# Save to CSV
results_df.to_csv(f"{target_col}_results.csv", index=False)
print("\n✅ Results saved")
