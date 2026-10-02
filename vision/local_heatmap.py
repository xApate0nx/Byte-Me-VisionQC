from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "normal"
DEFECT_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "defects"

OUTPUT_DIR = PROJECT_ROOT / "data" / "heatmaps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# --------------------------------------------------
# Load pretrained ResNet
# --------------------------------------------------

weights = models.ResNet18_Weights.DEFAULT

resnet = models.resnet18(weights=weights)

# Keep everything up to layer4.
feature_model = nn.Sequential(
    resnet.conv1,
    resnet.bn1,
    resnet.relu,
    resnet.maxpool,
    resnet.layer1,
    resnet.layer2,
    resnet.layer3,
    resnet.layer4,
)

feature_model = feature_model.to(DEVICE)
feature_model.eval()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

preprocess = weights.transforms()


# --------------------------------------------------
# Extract spatial feature map
# --------------------------------------------------

def extract_feature_map(image_path):

    image = Image.open(image_path).convert("RGB")

    original_image = image.copy()

    tensor = preprocess(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feature_map = feature_model(tensor)

    # Shape:
    # [1, 512, 7, 7]

    feature_map = feature_map.squeeze(0).cpu()

    return original_image, feature_map


# --------------------------------------------------
# Get image files
# --------------------------------------------------

def get_images(folder):

    return sorted(
        [
            p for p in folder.iterdir()
            if p.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png"
            ]
        ]
    )


normal_files = get_images(NORMAL_DIR)
defect_files = get_images(DEFECT_DIR)

print("Normal images:", len(normal_files))
print("Defect images:", len(defect_files))


# --------------------------------------------------
# Build normal reference
# --------------------------------------------------

print("\nExtracting normal feature maps...")

normal_maps = []

for path in normal_files:

    _, feature_map = extract_feature_map(path)

    normal_maps.append(feature_map)

normal_maps = torch.stack(normal_maps)

print(
    "Normal feature-map matrix:",
    tuple(normal_maps.shape)
)


# --------------------------------------------------
# Normal reference
# --------------------------------------------------

# Median is more robust than simply taking the first image.

normal_reference = torch.median(
    normal_maps,
    dim=0
).values


# --------------------------------------------------
# Calculate local anomaly map
# --------------------------------------------------

def calculate_anomaly_map(feature_map):

    # Difference between new image and normal reference

    difference = torch.abs(
        feature_map - normal_reference
    )

    # Average across feature channels

    anomaly_map = difference.mean(dim=0)

    anomaly_map = anomaly_map.numpy()

    # Normalize 0-1

    minimum = anomaly_map.min()
    maximum = anomaly_map.max()

    if maximum > minimum:

        anomaly_map = (
            anomaly_map - minimum
        ) / (
            maximum - minimum
        )

    else:

        anomaly_map = np.zeros_like(anomaly_map)

    return anomaly_map


# --------------------------------------------------
# Create heatmap
# --------------------------------------------------

def save_heatmap(image_path):

    image, feature_map = extract_feature_map(image_path)

    anomaly_map = calculate_anomaly_map(feature_map)

    output_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_heatmap.png"
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(image)
    plt.imshow(
        anomaly_map,
        cmap="jet",
        alpha=0.45,
        extent=(
            0,
            image.width,
            image.height,
            0
        )
    )

    plt.axis("off")
    plt.title("VisionQC Local Anomaly Heatmap")

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print("Saved:", output_path)


# --------------------------------------------------
# Test on defects
# --------------------------------------------------

print("\nGenerating defect heatmaps...")

for path in defect_files:

    save_heatmap(path)


print("\nDone.")
print("Heatmaps saved to:")
print(OUTPUT_DIR)