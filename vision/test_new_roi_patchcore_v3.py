from pathlib import Path
import sys
import numpy as np

# ------------------------------------------------------------
# Make project root importable when running:
# python vision\test_new_roi_patchcore_v3.py
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# VisionQC — PatchCore v3 Normal ROI Evaluation
#
# Model:
#   v3 trained on detector-generated canonical ROIs
#
# Dataset:
#   21 normal canonical ROIs
#
# IMPORTANT:
#   This script only measures raw PatchCore scores.
#   It does NOT apply the old threshold.
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap_v3"
    / "Patchcore"
    / "water_cap_v1_canonical_roi"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "new_roi"
)


def run_patchcore(image_path):
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )

    # We want the raw PatchCore distance, not Anomalib's
    # adaptive post-processing threshold.
    model.post_processor = None

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    dataset = PredictDataset(
        path=str(image_path),
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT),
    )

    if not predictions:
        raise RuntimeError(f"No prediction returned for {image_path}")

    prediction = predictions[0]

    score = prediction.pred_score

    if hasattr(score, "item"):
        score = score.item()

    return float(score)


def main():

    print("=" * 70)
    print("VISIONQC — PATCHCORE V3 NORMAL ROI EVALUATION")
    print("=" * 70)

    print(f"\nCheckpoint:")
    print(CHECKPOINT)

    print(f"\nNormal ROI directory:")
    print(NORMAL_DIR)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    images = sorted(
        list(NORMAL_DIR.glob("*.png"))
        + list(NORMAL_DIR.glob("*.jpg"))
        + list(NORMAL_DIR.glob("*.jpeg"))
    )

    print(f"\nNormal ROIs found: {len(images)}")

    if len(images) != 21:
        raise RuntimeError(
            f"Expected 21 normal ROIs, found {len(images)}"
        )

    scores = []

    print("\nRunning PatchCore v3...\n")

    for index, image_path in enumerate(images, start=1):

        score = run_patchcore(image_path)

        scores.append(score)

        print(
            f"{index:02d}/21  "
            f"{image_path.name:<65} "
            f"score={score:.6f}"
        )

    scores = np.array(scores, dtype=float)

    print("\n" + "=" * 70)
    print("PATCHCORE V3 NORMAL RESULTS")
    print("=" * 70)

    print(f"Images tested : {len(scores)}")
    print(f"Minimum       : {scores.min():.6f}")
    print(f"Maximum       : {scores.max():.6f}")
    print(f"Mean          : {scores.mean():.6f}")
    print(f"Median        : {np.median(scores):.6f}")
    print(f"Std deviation : {scores.std():.6f}")
    print(f"90th percentile: {np.percentile(scores, 90):.6f}")
    print(f"95th percentile: {np.percentile(scores, 95):.6f}")
    print(f"99th percentile: {np.percentile(scores, 99):.6f}")

    print("\nOLD THRESHOLD — DO NOT USE")
    print("12.3785223961")

    print("\nRecommended next step:")
    print("Run the same v3 checkpoint against all 23 defect ROIs.")
    print("Then compare the two score distributions.")

    print("=" * 70)


if __name__ == "__main__":
    main()