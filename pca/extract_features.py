# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 22:20:14 2025

@author: tjn_3
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 model, remove the last layer (for feature extraction only)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Data path (you manually specified the path for each city)
root_dir = "data/images"  # You manually specified the path
output_dir = "data/features"
os.makedirs(output_dir, exist_ok=True)

# Iterate through image folders of each city
city_list = os.listdir(root_dir)

for city in city_list:
    city_path = os.path.join(root_dir, city)
    if not os.path.isdir(city_path):
        continue

    feature_list = []
    image_name_list = []

    print(f"Extracting image features in {city}...")
    for filename in tqdm(os.listdir(city_path)):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(city_path, filename)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(image_tensor).cpu().numpy().squeeze()
            feature_list.append(feature)
            image_name_list.append(filename)

    # Save features to CSV
    df = pd.DataFrame(feature_list)
    df.insert(0, "image_name", image_name_list)
    df.to_csv(os.path.join(output_dir, f"{city}_features.csv"), index=False)

