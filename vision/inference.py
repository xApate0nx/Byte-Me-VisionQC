from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
    / "normal"
)

MODEL_DIR = PROJECT_ROOT / "models" / "visionqc_resnet18"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "normal_reference.pt"

INSPECTION_DIR = PROJECT_ROOT / "data" / "inspections"
INSPECTION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# RESNET-18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

resnet = models.resnet18(weights=weights)

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

preprocess = weights.transforms()


# ============================================================
# IMAGE HELPERS
# ============================================================

def get_images(folder):
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        ]
    )


def extract_feature_map(image_path):
    """
    Convert one image into a ResNet-18 layer4 feature map.

    Output shape:
        [512, 7, 7]
    """

    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feature_map = feature_model(tensor)

    return feature_map.squeeze(0).cpu()


# ============================================================
# BUILD NORMAL REFERENCE
# ============================================================

def build_normal_reference():
    """
    Learn normal appearance from the good-product images.

    Uses the median feature map across all normal examples.
    """

    normal_files = get_images(NORMAL_DIR)

    if not normal_files:
        raise RuntimeError(
            f"No normal images found in: {NORMAL_DIR}"
        )

    print()
    print("Building normal reference...")
    print("Normal images:", len(normal_files))

    feature_maps = []

    for index, image_path in enumerate(normal_files, start=1):

        print(
            f"[{index}/{len(normal_files)}] "
            f"{image_path.name}"
        )

        feature_map = extract_feature_map(image_path)

        feature_maps.append(feature_map)

    feature_maps = torch.stack(feature_maps)

    print(
        "Feature-map matrix:",
        tuple(feature_maps.shape)
    )

    # Robust representative of normal appearance.
    normal_reference = torch.median(
        feature_maps,
        dim=0
    ).values

    torch.save(
        {
            "reference": normal_reference,
            "feature_shape": tuple(normal_reference.shape),
            "num_normal_images": len(normal_files),
            "backbone": "resnet18",
            "layer": "layer4",
        },
        MODEL_FILE
    )

    print()
    print("Normal reference saved:")
    print(MODEL_FILE)

    return normal_reference


# ============================================================
# LOAD NORMAL REFERENCE
# ============================================================

def load_normal_reference():

    if not MODEL_FILE.exists():

        print(
            "Normal reference does not exist."
        )

        return build_normal_reference()

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    reference = checkpoint["reference"]

    print(
        "Loaded normal reference:",
        MODEL_FILE
    )

    print(
        "Reference shape:",
        tuple(reference.shape)
    )

    return reference


# ============================================================
# ANOMALY MAP
# ============================================================

def calculate_anomaly_map(
    feature_map,
    normal_reference
):
    """
    Compare a new feature map against the normal reference.

    Returns:
        raw 7x7 anomaly map
        normalized 7x7 anomaly map
    """

    difference = torch.abs(
        feature_map - normal_reference
    )

    # Average deviation across the 512 channels.
    anomaly_map = difference.mean(dim=0)

    anomaly_map = anomaly_map.numpy()

    minimum = float(anomaly_map.min())
    maximum = float(anomaly_map.max())

    if maximum > minimum:

        normalized_map = (
            anomaly_map - minimum
        ) / (
            maximum - minimum
        )

    else:

        normalized_map = np.zeros_like(
            anomaly_map
        )

    return anomaly_map, normalized_map


# ============================================================
# REGION EXTRACTION
# ============================================================

def get_anomaly_region(
    anomaly_map,
    image_width,
    image_height
):
    """
    Convert the 7x7 anomaly map into an approximate
    anomalous region on the original image.

    This is a prototype localization mechanism.
    """

    threshold = np.percentile(
        anomaly_map,
        75
    )

    mask = anomaly_map >= threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:

        return None

    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())

    grid_height, grid_width = anomaly_map.shape

    x1 = int(
        x_min
        * image_width
        / grid_width
    )

    x2 = int(
        (x_max + 1)
        * image_width
        / grid_width
    )

    y1 = int(
        y_min
        * image_height
        / grid_height
    )

    y2 = int(
        (y_max + 1)
        * image_height
        / grid_height
    )

    return {
        "x": x1,
        "y": y1,
        "width": max(1, x2 - x1),
        "height": max(1, y2 - y1),
    }


# ============================================================
# HEATMAP
# ============================================================

def save_heatmap(
    image,
    anomaly_map,
    output_path
):

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(image)

    plt.imshow(
        anomaly_map,
        cmap="jet",
        alpha=0.45,
        extent=(
            0,
            image.width,
            image.height,
            0,
        ),
        interpolation="bilinear",
    )

    plt.axis("off")

    plt.title(
        "VisionQC Local Anomaly Heatmap"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=150,
    )

    plt.close()


# ============================================================
# SINGLE IMAGE INFERENCE
# ============================================================

def inspect_image(
    image_path,
    save_outputs=True
):
    """
    Run VisionQC inference on ONE image.

    Returns a dictionary containing:

        anomaly_score
        anomaly_region
        heatmap_path
        image_path
    """

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print()
    print("=" * 60)
    print("VisionQC Inspection")
    print("=" * 60)

    print("Image:", image_path)

    # Load normal reference.
    normal_reference = load_normal_reference()

    # Load original image.
    image = Image.open(
        image_path
    ).convert("RGB")

    print(
        "Image size:",
        image.size
    )

    # Extract feature map.
    feature_map = extract_feature_map(
        image_path
    )

    print(
        "Feature map:",
        tuple(feature_map.shape)
    )

    # Calculate local anomaly map.
    raw_map, normalized_map = (
        calculate_anomaly_map(
            feature_map,
            normal_reference
        )
    )

    # Prototype global anomaly score.
    anomaly_score = float(
        raw_map.mean()
    )

    # Find approximate anomalous region.
    anomaly_region = get_anomaly_region(
        normalized_map,
        image.width,
        image.height,
    )

    heatmap_path = None

    if save_outputs:

        output_stem = (
            image_path.stem
            + "_inspection"
        )

        heatmap_path = (
            INSPECTION_DIR
            / f"{output_stem}_heatmap.png"
        )

        result_path = (
            INSPECTION_DIR
            / f"{output_stem}.json"
        )

        save_heatmap(
            image,
            normalized_map,
            heatmap_path,
        )

        result = {
            "image": str(image_path),
            "anomaly_score": anomaly_score,
            "anomaly_region": anomaly_region,
            "heatmap": str(heatmap_path),
            "backbone": "resnet18",
            "feature_layer": "layer4",
            "feature_shape": list(
                feature_map.shape
            ),
        }

        with open(
            result_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
            )

        print()
        print(
            "Heatmap saved:",
            heatmap_path
        )

        print(
            "Result saved:",
            result_path
        )

    else:

        result = {
            "image": str(image_path),
            "anomaly_score": anomaly_score,
            "anomaly_region": anomaly_region,
            "heatmap": None,
        }

    print()
    print("ANOMALY SCORE:", anomaly_score)

    print(
        "ANOMALOUS REGION:",
        anomaly_region
    )

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    defect_files = get_images(
        PROJECT_ROOT
        / "data"
        / "products"
        / "water_cap_v1"
        / "defects"
    )

    if not defect_files:

        raise RuntimeError(
            "No defect test images found."
        )

    test_image = defect_files[0]

    print()
    print(
        "Using test image:"
    )
    print(test_image)

    result = inspect_image(
        test_image
    )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=4
        )
    )