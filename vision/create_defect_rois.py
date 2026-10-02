from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from vision.product_detector import (
    load_image,
    detect_product,
)


# ============================================================
# PATHS
# ============================================================

DEFECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
    / "defects"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "defects_new_roi"
)


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_SIZE = 224
PADDING = 0.30

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# NEW CANONICAL ROI
# ============================================================

def create_new_roi(
    image,
    bbox,
    output_size=OUTPUT_SIZE,
    padding=PADDING,
):
    """
    Create the same detector-driven canonical ROI
    used for the 21 normal images.
    """

    height, width = image.shape[:2]

    if bbox is None:
        raise ValueError(
            "Bounding box is None."
        )

    if len(bbox) != 4:
        raise ValueError(
            f"Expected bbox with 4 values, got: {bbox}"
        )

    x1, y1, x2, y2 = map(
        int,
        bbox,
    )

    # --------------------------------------------------------
    # Clamp detector bbox
    # --------------------------------------------------------

    x1 = max(
        0,
        min(x1, width - 1),
    )

    y1 = max(
        0,
        min(y1, height - 1),
    )

    x2 = max(
        0,
        min(x2, width),
    )

    y2 = max(
        0,
        min(y2, height),
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bbox: {(x1, y1, x2, y2)}"
        )

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # --------------------------------------------------------
    # Square crop
    # --------------------------------------------------------

    square_size = max(
        bbox_width,
        bbox_height,
    )

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    # --------------------------------------------------------
    # Add same 30% contextual padding
    # --------------------------------------------------------

    crop_size = int(
        round(
            square_size
            * (1.0 + padding)
        )
    )

    crop_size = max(
        crop_size,
        1,
    )

    half_size = crop_size / 2.0

    crop_x1 = int(
        round(
            center_x - half_size
        )
    )

    crop_y1 = int(
        round(
            center_y - half_size
        )
    )

    crop_x2 = (
        crop_x1
        + crop_size
    )

    crop_y2 = (
        crop_y1
        + crop_size
    )

    # --------------------------------------------------------
    # Shift crop inside image
    # --------------------------------------------------------

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0

    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0

    if crop_x2 > width:
        shift = crop_x2 - width
        crop_x1 -= shift
        crop_x2 = width

    if crop_y2 > height:
        shift = crop_y2 - height
        crop_y1 -= shift
        crop_y2 = height

    crop_x1 = max(
        0,
        crop_x1,
    )

    crop_y1 = max(
        0,
        crop_y1,
    )

    crop_x2 = min(
        width,
        crop_x2,
    )

    crop_y2 = min(
        height,
        crop_y2,
    )

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError(
            "Final ROI is invalid."
        )

    roi = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    roi = cv2.resize(
        roi,
        (
            output_size,
            output_size,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return roi


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VISIONQC — CREATE DEFECT NEW ROIS")
    print("=" * 70)

    if not DEFECT_DIR.exists():
        raise FileNotFoundError(
            f"Defect directory not found:\n"
            f"{DEFECT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        [
            path
            for path in DEFECT_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ]
    )

    print()
    print(
        f"Defect images found: "
        f"{len(image_paths)}"
    )

    print()

    success_count = 0
    failure_count = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:

            # ------------------------------------------------
            # LOAD
            # ------------------------------------------------

            image = load_image(
                str(image_path)
            )

            if image is None:
                raise ValueError(
                    "load_image() returned None."
                )

            # ------------------------------------------------
            # DETECTOR
            # ------------------------------------------------

            detection = detect_product(
                str(image_path)
            )

            detected = bool(
                getattr(
                    detection,
                    "detected",
                    False,
                )
            )

            bbox = getattr(
                detection,
                "bbox",
                None,
            )

            confidence = getattr(
                detection,
                "confidence",
                None,
            )

            print(
                f"       detector type: "
                f"{type(detection).__name__}"
            )

            print(
                f"       detected : "
                f"{detected}"
            )

            print(
                f"       bbox     : "
                f"{bbox}"
            )

            print(
                f"       confidence: "
                f"{confidence}"
            )

            if not detected:
                print(
                    "       SKIPPED: "
                    "product not detected."
                )

                failure_count += 1

                print()
                continue

            if bbox is None:
                print(
                    "       SKIPPED: "
                    "no bbox returned."
                )

                failure_count += 1

                print()
                continue

            # ------------------------------------------------
            # NEW ROI
            # ------------------------------------------------

            roi = create_new_roi(
                image,
                bbox,
            )

            output_path = (
                OUTPUT_DIR
                / f"{image_path.stem}_new_roi.png"
            )

            cv2.imwrite(
                str(output_path),
                roi,
            )

            success_count += 1

            print(
                f"       ROI      : "
                f"{roi.shape[1]}x"
                f"{roi.shape[0]}"
            )

            print(
                f"       saved    : "
                f"{output_path.name}"
            )

        except Exception as exc:

            failure_count += 1

            print(
                f"       ERROR: "
                f"{exc}"
            )

        print()

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 70)
    print("DEFECT ROI GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Total defects     : "
        f"{len(image_paths)}"
    )

    print(
        f"Successful ROIs   : "
        f"{success_count}"
    )

    print(
        f"Failures/skipped  : "
        f"{failure_count}"
    )

    if image_paths:

        detection_rate = (
            success_count
            / len(image_paths)
            * 100
        )

        print(
            f"Success rate      : "
            f"{detection_rate:.2f}%"
        )

    print()
    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()


if __name__ == "__main__":
    main()