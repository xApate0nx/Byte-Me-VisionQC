from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from vision.product_detector import (
    load_image,
    detect_product,
)


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
)

OLD_ROI_DIR = OUTPUT_DIR / "old_roi"
NEW_ROI_DIR = OUTPUT_DIR / "new_roi"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"


# ============================================================
# SETTINGS
# ============================================================

OLD_CANVAS_SIZE = 400
OLD_ROI_RATIO = 0.80

NEW_ROI_SIZE = 224
NEW_ROI_PADDING = 0.30

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

OLD_ROI_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NEW_ROI_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMPARISON_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OLD ROI
# ============================================================

def create_old_roi(
    image: np.ndarray,
) -> np.ndarray:
    """
    Recreate the old PatchCore training ROI:

        original image
        -> fit inside 400x400 white canvas
        -> take central 80%
        -> resize to 256x256
    """

    height, width = image.shape[:2]

    scale = min(
        OLD_CANVAS_SIZE / width,
        OLD_CANVAS_SIZE / height,
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )

    new_height = max(
        1,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (
            OLD_CANVAS_SIZE,
            OLD_CANVAS_SIZE,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    x_offset = (
        OLD_CANVAS_SIZE - new_width
    ) // 2

    y_offset = (
        OLD_CANVAS_SIZE - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    roi_size = int(
        OLD_CANVAS_SIZE * OLD_ROI_RATIO
    )

    start = (
        OLD_CANVAS_SIZE - roi_size
    ) // 2

    old_roi = canvas[
        start:start + roi_size,
        start:start + roi_size,
    ]

    old_roi = cv2.resize(
        old_roi,
        (
            256,
            256,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return old_roi


# ============================================================
# NEW CANONICAL ROI
# ============================================================

def create_new_roi(
    image: np.ndarray,
    bbox,
    output_size: int = NEW_ROI_SIZE,
    padding: float = NEW_ROI_PADDING,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Create the new detector-driven canonical ROI.

    bbox is expected to represent:

        x1, y1, x2, y2

    The crop is made square using the largest bbox dimension,
    then expanded with contextual padding.
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
    # Clamp detector bbox to image boundaries
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
            f"Invalid bbox after clamping: "
            f"{(x1, y1, x2, y2)}"
        )

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # --------------------------------------------------------
    # Make square around detector bbox
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
    # Add contextual margin
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
        round(center_x - half_size)
    )

    crop_y1 = int(
        round(center_y - half_size)
    )

    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    # --------------------------------------------------------
    # Shift crop to stay inside image
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

    # Final safety clamp
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
            "New ROI crop became invalid."
        )

    new_roi = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    new_roi = cv2.resize(
        new_roi,
        (
            output_size,
            output_size,
        ),
        interpolation=cv2.INTER_AREA,
    )

    actual_roi_bbox = (
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
    )

    return (
        new_roi,
        actual_roi_bbox,
    )


# ============================================================
# DRAW DETECTOR / ROI
# ============================================================

def draw_detection(
    image: np.ndarray,
    detector_bbox,
    roi_bbox,
) -> np.ndarray:
    """
    Draw:

        green = detector bbox
        red   = final canonical ROI
    """

    result = image.copy()

    if detector_bbox is not None:
        x1, y1, x2, y2 = map(
            int,
            detector_bbox,
        )

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            result,
            "DETECTOR BBOX",
            (
                max(5, x1),
                max(25, y1 - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if roi_bbox is not None:
        x1, y1, x2, y2 = map(
            int,
            roi_bbox,
        )

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            result,
            "CANONICAL ROI",
            (
                max(5, x1),
                min(
                    result.shape[0] - 10,
                    y2 + 25,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return result


# ============================================================
# CONTACT SHEET
# ============================================================

def add_label(
    image: np.ndarray,
    label: str,
) -> np.ndarray:
    """
    Add a readable label above an image.
    """

    h, w = image.shape[:2]

    header_height = 45

    output = np.full(
        (
            h + header_height,
            w,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    output[
        header_height:,
        :
    ] = image

    cv2.putText(
        output,
        label,
        (
            10,
            30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return output


def create_comparison(
    original: np.ndarray,
    detected_image: np.ndarray,
    old_roi: np.ndarray,
    new_roi: np.ndarray,
) -> np.ndarray:
    """
    Create a 4-panel comparison:

        Original
        Detector + ROI
        Old ROI
        New ROI
    """

    panel_size = 400

    def resize_panel(
        img: np.ndarray,
    ) -> np.ndarray:

        return cv2.resize(
            img,
            (
                panel_size,
                panel_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

    panels = [
        add_label(
            resize_panel(original),
            "ORIGINAL",
        ),
        add_label(
            resize_panel(detected_image),
            "DETECTOR + NEW ROI",
        ),
        add_label(
            resize_panel(old_roi),
            "OLD ROI",
        ),
        add_label(
            resize_panel(new_roi),
            "NEW CANONICAL ROI",
        ),
    ]

    return np.hstack(
        panels
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "VISIONQC ROI COMPARISON"
    )
    print(
        "=" * 60
    )

    if not NORMAL_DIR.exists():
        print(
            f"ERROR: Normal directory does not exist:\n"
            f"{NORMAL_DIR}"
        )
        return

    image_paths = sorted(
        [
            path
            for path in NORMAL_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ]
    )

    print(
        f"Normal images found: "
        f"{len(image_paths)}"
    )

    print()

    if not image_paths:
        print(
            "ERROR: No normal images found."
        )
        return

    detected_count = 0
    comparison_count = 0
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
            # LOAD ORIGINAL
            # ------------------------------------------------

            image = load_image(
                str(image_path)
            )

            if image is None:
                raise ValueError(
                    "load_image() returned None."
                )

            print(
                f"       image size: "
                f"{image.shape[1]}x{image.shape[0]}"
            )

            # ------------------------------------------------
            # OLD ROI
            # ------------------------------------------------

            old_roi = create_old_roi(
                image
            )

            old_roi_path = (
                OLD_ROI_DIR
                / f"{image_path.stem}_old_roi.png"
            )

            cv2.imwrite(
                str(old_roi_path),
                old_roi,
            )

            print(
                f"       old ROI  : "
                f"{old_roi.shape[1]}x"
                f"{old_roi.shape[0]}"
            )

            # ------------------------------------------------
            # DETECTOR
            # ------------------------------------------------

            detection = detect_product(
                str(image_path)
            )

            # detect_product() returns a DetectionResult
            # object, not a dictionary.

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
                    "detector returned no bbox."
                )

                failure_count += 1
                print()
                continue

            detected_count += 1

            # ------------------------------------------------
            # NEW CANONICAL ROI
            # ------------------------------------------------

            new_roi, roi_bbox = create_new_roi(
                image,
                bbox,
            )

            new_roi_path = (
                NEW_ROI_DIR
                / f"{image_path.stem}_new_roi.png"
            )

            cv2.imwrite(
                str(new_roi_path),
                new_roi,
            )

            print(
                f"       new ROI  : "
                f"{new_roi.shape[1]}x"
                f"{new_roi.shape[0]}"
            )

            print(
                f"       ROI bbox : "
                f"{roi_bbox}"
            )

            # ------------------------------------------------
            # DRAW DETECTOR + ROI
            # ------------------------------------------------

            detected_image = draw_detection(
                image,
                bbox,
                roi_bbox,
            )

            # ------------------------------------------------
            # COMPARISON SHEET
            # ------------------------------------------------

            comparison = create_comparison(
                image,
                detected_image,
                old_roi,
                new_roi,
            )

            comparison_path = (
                COMPARISON_DIR
                / f"{image_path.stem}_comparison.jpg"
            )

            cv2.imwrite(
                str(comparison_path),
                comparison,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95,
                ],
            )

            comparison_count += 1

            print(
                f"       comparison: "
                f"{comparison_path.name}"
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

    print(
        "=" * 60
    )

    print(
        "ROI COMPARISON COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Total normal images : "
        f"{len(image_paths)}"
    )

    print(
        f"Detector successes  : "
        f"{detected_count}"
    )

    print(
        f"ROI comparisons     : "
        f"{comparison_count}"
    )

    print(
        f"Failures / skipped  : "
        f"{failure_count}"
    )

    if len(image_paths) > 0:

        detection_rate = (
            detected_count
            / len(image_paths)
            * 100
        )

        print(
            f"Detection rate      : "
            f"{detection_rate:.2f}%"
        )

    print()
    print(
        "Output directories:"
    )

    print(
        f"  Old ROI      : "
        f"{OLD_ROI_DIR}"
    )

    print(
        f"  New ROI      : "
        f"{NEW_ROI_DIR}"
    )

    print(
        f"  Comparisons  : "
        f"{COMPARISON_DIR}"
    )

    print()


if __name__ == "__main__":
    main()