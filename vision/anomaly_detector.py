from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NORMAL_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "normal"
DEFECT_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "defects"

IMAGE_SIZE = (256, 256)


# ---------------------------------------------------------
# Image loading
# ---------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load an image and convert it to RGB."""

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)

    return image


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Convert an image into a compact numerical representation.

    We currently use:
    - color information
    - grayscale structure
    """

    # Resize again defensively
    image = cv2.resize(image, IMAGE_SIZE)

    # Color histogram
    hist = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256],
    )

    hist = cv2.normalize(hist, hist).flatten()

    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Downsample structural information
    small_gray = cv2.resize(gray, (32, 32)).flatten()
    small_gray = small_gray.astype(np.float32) / 255.0

    return np.concatenate([hist, small_gray])


# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------

def load_dataset(directory: Path):
    """Load all JPEG/PNG images from a directory."""

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    paths = [
        path
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in extensions
    ]

    images = []
    features = []

    for path in paths:
        image = load_image(path)
        feature_vector = extract_features(image)

        images.append(image)
        features.append(feature_vector)

    return paths, images, np.array(features)


# ---------------------------------------------------------
# Train normal model
# ---------------------------------------------------------

def train_normal_model(normal_features: np.ndarray):

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    model.fit(normal_features)

    return model


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("VisionQC — Normal Reference Model")
    print("=" * 60)

    print(f"\nNormal directory:")
    print(NORMAL_DIR)

    print(f"\nDefect directory:")
    print(DEFECT_DIR)

    # Load normal images
    normal_paths, normal_images, normal_features = load_dataset(NORMAL_DIR)

    print(f"\nLoaded normal images: {len(normal_paths)}")
    print(f"Feature vector size: {normal_features.shape}")

    # Train model
    print("\nTraining normal-reference model...")

    model = train_normal_model(normal_features)

    print("Model trained successfully.")

    # Test normal images
    normal_predictions = model.predict(normal_features)

    print("\nNormal image predictions:")
    print(normal_predictions)

    # Load defects
    defect_paths, defect_images, defect_features = load_dataset(DEFECT_DIR)

    print(f"\nLoaded defect images: {len(defect_paths)}")

    # Test defects
    defect_predictions = model.predict(defect_features)

    print("\nDefect image predictions:")
    print(defect_predictions)

    # Convert predictions
    normal_anomalies = np.sum(normal_predictions == -1)
    defect_anomalies = np.sum(defect_predictions == -1)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Normal images detected as anomalies: "
        f"{normal_anomalies}/{len(normal_predictions)}"
    )

    print(
        f"Defect images detected as anomalies: "
        f"{defect_anomalies}/{len(defect_predictions)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
