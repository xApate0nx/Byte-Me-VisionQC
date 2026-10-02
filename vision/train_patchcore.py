from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# VisionQC — PatchCore Training
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "normal"
OUTPUT_DIR = PROJECT_ROOT / "models" / "patchcore_water_cap"


def main():
    print("=" * 60)
    print("VisionQC — PatchCore Training")
    print("=" * 60)

    print(f"\nNormal dataset:")
    print(NORMAL_DIR)

    images = list(NORMAL_DIR.glob("*.jpg")) + list(NORMAL_DIR.glob("*.jpeg")) + list(NORMAL_DIR.glob("*.png"))

    print(f"Normal images found: {len(images)}")

    if len(images) == 0:
        raise RuntimeError("No images found in the normal dataset.")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    #
    # Only GOOD images are supplied.
    # Anomalib treats them as the normal/reference class.
    #
    datamodule = Folder(
        name="water_cap_v1",
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
    #
    # ResNet-18 keeps the prototype relatively lightweight.
    #
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

    print("\nStarting PatchCore training...")
    print("Backbone : ResNet-18")
    print("Layers   : layer2 + layer3")
    print("Device   : CPU")
    print()

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\n" + "=" * 60)
    print("PatchCore training completed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()