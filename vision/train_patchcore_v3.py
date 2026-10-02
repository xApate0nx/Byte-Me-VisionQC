from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# VisionQC — PatchCore v3 Training
#
# Training representation:
# original image
#     ↓
# product detector
#     ↓
# canonical ROI
#     ↓
# 224 × 224 PNG
#
# Only GOOD/normal canonical ROIs are used for training.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "new_roi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap_v3"
)


def main():
    print("=" * 70)
    print("VisionQC — PatchCore v3 Training")
    print("=" * 70)

    print("\nTraining representation:")
    print("Original image → product detector → canonical ROI → PatchCore")

    print("\nNormal ROI dataset:")
    print(NORMAL_DIR)

    images = (
        list(NORMAL_DIR.glob("*.jpg"))
        + list(NORMAL_DIR.glob("*.jpeg"))
        + list(NORMAL_DIR.glob("*.png"))
    )

    print(f"\nNormal ROI images found: {len(images)}")

    if len(images) != 21:
        raise RuntimeError(
            f"Expected 21 canonical normal ROIs, found {len(images)}"
        )

    print("\nPatchCore configuration:")
    print("Backbone     : ResNet-18")
    print("Layers       : layer2 + layer3")
    print("Neighbors    : 9")
    print("Device       : CPU")
    print("Training set : 21 normal canonical ROIs")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    #
    # Every image in this directory is a GOOD product ROI.
    #
    datamodule = Folder(
        name="water_cap_v1_canonical_roi",
        root=str(NORMAL_DIR),
        normal_dir=".",
        abnormal_dir=None,
        normal_split_ratio=0.0,
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )

    # --------------------------------------------------------
    # PatchCore
    # --------------------------------------------------------

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )

    # --------------------------------------------------------
    # Training engine
    # --------------------------------------------------------

    engine = Engine(
        default_root_dir=str(OUTPUT_DIR),
        accelerator="cpu",
        devices=1,
    )

    print("\nStarting PatchCore v3 training...\n")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\n" + "=" * 70)
    print("PatchCore v3 training completed.")
    print("=" * 70)

    print(f"\nModel output:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()