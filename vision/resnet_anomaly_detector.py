from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from feature_extractor import extract_feature


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "normal"
DEFECT_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "defects"


# --------------------------------------------------
# Get image files
# --------------------------------------------------

def get_images(folder):

    return sorted(
        [
            p for p in folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )


normal_files = get_images(NORMAL_DIR)
defect_files = get_images(DEFECT_DIR)


print("=" * 60)
print("VisionQC ResNet Anomaly Detector")
print("=" * 60)

print(f"Normal images : {len(normal_files)}")
print(f"Defect images : {len(defect_files)}")


# --------------------------------------------------
# Extract normal features
# --------------------------------------------------

print("\nExtracting normal features...")

normal_features = np.array(
    [extract_feature(path) for path in normal_files]
)

print("Normal feature matrix:", normal_features.shape)


# --------------------------------------------------
# Train anomaly detector
# --------------------------------------------------

detector = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=42
)

detector.fit(normal_features)


# --------------------------------------------------
# Evaluate normal images
# --------------------------------------------------

print("\nEvaluating normal images...")

normal_predictions = detector.predict(normal_features)

normal_anomalies = np.sum(normal_predictions == -1)

print(
    f"Normal images detected as anomalies: "
    f"{normal_anomalies}/{len(normal_files)}"
)


# --------------------------------------------------
# Extract defect features
# --------------------------------------------------

print("\nExtracting defect features...")

defect_features = np.array(
    [extract_feature(path) for path in defect_files]
)

print("Defect feature matrix:", defect_features.shape)


# --------------------------------------------------
# Evaluate defects
# --------------------------------------------------

print("\nEvaluating defect images...")

defect_predictions = detector.predict(defect_features)

defect_anomalies = np.sum(defect_predictions == -1)

print(
    f"Defect images detected as anomalies: "
    f"{defect_anomalies}/{len(defect_files)}"
)


# --------------------------------------------------
# Anomaly scores
# --------------------------------------------------

normal_scores = detector.decision_function(normal_features)
defect_scores = detector.decision_function(defect_features)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Normal anomaly detections : "
    f"{normal_anomalies}/{len(normal_files)}"
)

print(
    f"Defect anomaly detections : "
    f"{defect_anomalies}/{len(defect_files)}"
)


print("\nNormal scores:")
for path, score in zip(normal_files, normal_scores):
    print(f"{path.name[:45]:45} {score:.4f}")


print("\nDefect scores:")
for path, score in zip(defect_files, defect_scores):
    print(f"{path.name[:45]:45} {score:.4f}")