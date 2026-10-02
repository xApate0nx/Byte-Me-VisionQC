from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "normal"
DEFECT_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "defects"


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Pretrained ResNet-18
# --------------------------------------------------

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Remove the final classification layer.
# The output will be a visual feature vector instead.
model.fc = nn.Identity()

model = model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

preprocess = weights.transforms()


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_feature(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feature = model(tensor)

    feature = feature.squeeze(0).cpu().numpy()

    return feature


# --------------------------------------------------
# Test the extractor
# --------------------------------------------------

normal_files = sorted(
    [
        p for p in NORMAL_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
)

defect_files = sorted(
    [
        p for p in DEFECT_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
)


print("Device:", DEVICE)
print("Normal images:", len(normal_files))
print("Defect images:", len(defect_files))


if normal_files:

    feature = extract_feature(normal_files[0])

    print("Feature vector shape:", feature.shape)
    print("First 10 feature values:")
    print(feature[:10])