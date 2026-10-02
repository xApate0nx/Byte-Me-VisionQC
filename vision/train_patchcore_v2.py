from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_candidates"
    / "large"
    / "normal"
)

OUTPUT_DIR = PROJECT_ROOT / "models" / "patchcore_water_cap_v2"


def main():
    print("=" * 70)
    print("VisionQC — PatchCore v2 Training")
    print("=" * 70)

    print(f"\nTraining images:")
    print(NORMAL_DIR)

    images = (
        list(NORMAL_DIR.glob("*.jpg"))
        + list(NORMAL_DIR.glob("*.jpeg"))
        + list(NORMAL_DIR.glob("*.png"))
    )

    print(f"Normal images found: {len(images)}")

    if len(images) != 21:
        raise RuntimeError(
            f"Expected 21 normal images, found {len(images)}"
        )

    datamodule = Folder(
        name="water_cap_v1_roi80",
        root=str(NORMAL_DIR),
        normal_dir=".",
        abnormal_dir=None,
        normal_split_ratio=0.0,
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=str(OUTPUT_DIR),
        accelerator="cpu",
        devices=1,
    )

    print("\nStarting PatchCore v2 training...\n")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\n" + "=" * 70)
    print("PatchCore v2 training completed.")
    print("=" * 70)
    print(f"\nModel output:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()