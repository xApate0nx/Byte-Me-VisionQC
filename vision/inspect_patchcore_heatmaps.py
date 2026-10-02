from pathlib import Path

import cv2
import numpy as np
import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap"
    / "Patchcore"
    / "water_cap_v1"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "patchcore_heatmaps"


def save_heatmap(image_path, anomaly_map, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # Resize anomaly map to original image size
    anomaly_map = cv2.resize(
        anomaly_map,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Normalize ONLY for visualization
    min_value = anomaly_map.min()
    max_value = anomaly_map.max()

    if max_value > min_value:
        normalized = (anomaly_map - min_value) / (
            max_value - min_value
        )
    else:
        normalized = np.zeros_like(anomaly_map)

    heat = (normalized * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heat,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        image,
        0.60,
        heatmap,
        0.40,
        0,
    )

    # Add statistics
    cv2.putText(
        overlay,
        f"max={max_value:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        overlay,
        f"mean={anomaly_map.mean():.4f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        overlay,
    )


def main():

    print("=" * 70)
    print("VisionQC — PatchCore Heatmap Inspection")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    datamodule = Folder(
        name="water_cap_evaluation",
        root=str(DATA_ROOT),
        normal_dir="normal",
        abnormal_dir="defects",
        normal_split_ratio=0.0,
        test_split_mode="from_dir",
        test_split_ratio=0.0,
        val_split_mode="none",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )

    datamodule.setup()

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    print("\nLoading PatchCore checkpoint...")

    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT),
    )

    print("\nGenerating heatmaps...\n")

    count = 0

    for prediction in predictions:

        paths = prediction.image_path
        maps = prediction.anomaly_map

        if torch.is_tensor(maps):
            maps_np = maps.detach().cpu().numpy()
        else:
            maps_np = maps

        for i, path in enumerate(paths):

            path = Path(path)

            category = path.parent.name

            output_name = (
                path.stem
                + "_patchcore_heatmap.jpg"
            )

            output_path = (
                OUTPUT_DIR
                / category
                / output_name
            )

            save_heatmap(
                path,
                maps_np[i],
                output_path,
            )

            print(
                f"[{count:02d}] "
                f"{category} / {path.name}"
            )

            count += 1

    print("\n" + "=" * 70)
    print(f"Generated {count} heatmaps")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()