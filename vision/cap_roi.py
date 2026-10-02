from __future__ import annotations
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when this file is run as:
# python vision/cap_roi.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from vision.product_detector import (
    load_image,
    detect_product,
)
from pathlib import Path

import cv2
import numpy as np

from vision.product_detector import (
    load_image,
    build_color_mask,
    detect_product,
)


def extract_canonical_roi(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_size: int = 224,
    padding: float = 0.30,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:

    height, width = image.shape[:2]

    x1, y1, x2, y2 = bbox

    box_width = x2 - x1
    box_height = y2 - y1

    # Make the crop square using the larger dimension.
    side = max(box_width, box_height)

    # Add contextual margin around the product.
    side = int(
        round(side * (1.0 + padding))
    )

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    crop_x1 = int(
        round(center_x - side / 2)
    )

    crop_y1 = int(
        round(center_y - side / 2)
    )

    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side

    # Clamp to image boundaries.
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)

    crop_x2 = min(width, crop_x2)
    crop_y2 = min(height, crop_y2)

    # If clamping made the crop rectangular,
    # resize it back to a square later.
    roi = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    if roi.size == 0:
        raise ValueError(
            "Canonical ROI is empty."
        )

    roi = cv2.resize(
        roi,
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )

    return (
        roi,
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        ),
    )


def process_image(
    image_path: str | Path,
    output_path: str | Path,
) -> bool:

    image_path = Path(image_path)
    output_path = Path(output_path)

    image = load_image(image_path)

    result = detect_product(image_path)

    if not result.detected:
        print(
            f"SKIP: product not detected: "
            f"{image_path.name}"
        )
        return False

    if result.bbox is None:
        print(
            f"SKIP: no bounding box: "
            f"{image_path.name}"
        )
        return False

    roi, crop_box = extract_canonical_roi(
        image,
        result.bbox,
        output_size=224,
        padding=0.30,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        roi,
    )

    print(
        f"OK: {image_path.name}"
    )

    print(
        f"    detection confidence: "
        f"{result.confidence:.3f}"
    )

    print(
        f"    product bbox: "
        f"{result.bbox}"
    )

    print(
        f"    canonical crop: "
        f"{crop_box}"
    )

    print(
        f"    output: "
        f"{output_path}"
    )

    return True


def process_dataset():

    dataset_root = Path(
        "data/products/water_cap_v1"
    )

    output_root = Path(
        "data/evaluation/roi"
    )

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    total = 0
    success = 0

    for category in [
        "normal",
        "defects",
    ]:

        source_dir = (
            dataset_root / category
        )

        output_dir = (
            output_root / category
        )

        if not source_dir.exists():
            continue

        for image_path in sorted(
            source_dir.iterdir()
        ):

            if (
                not image_path.is_file()
                or image_path.suffix.lower()
                not in extensions
            ):
                continue

            total += 1

            output_path = (
                output_dir
                / f"{image_path.stem}_roi.png"
            )

            try:

                if process_image(
                    image_path,
                    output_path,
                ):
                    success += 1

            except Exception as exc:

                print(
                    f"ERROR: {image_path.name}"
                )

                print(
                    f"    {exc}"
                )

    print()
    print("=" * 60)
    print("ROI EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Images: {total}")
    print(f"Successful: {success}")

    if total:
        print(
            f"Success rate: "
            f"{success / total * 100:.1f}%"
        )

    print(
        f"Output: {output_root}"
    )


if __name__ == "__main__":
    process_dataset()