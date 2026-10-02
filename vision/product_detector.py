from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "detection"
)


@dataclass
class DetectionResult:
    detected: bool
    confidence: float
    bbox: tuple[int, int, int, int] | None
    center: tuple[float, float] | None
    occupancy: float
    circularity: float
    aspect_ratio: float
    color_ratio: float
    alignment: str
    reason: str


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    return cv2.cvtColor(
        np.asarray(image),
        cv2.COLOR_RGB2BGR,
    )


def build_color_mask(image: np.ndarray) -> np.ndarray:
    """
    Adaptive teal/cyan segmentation.

    Uses HSV for hue/saturation and Lab for color separation.
    No exact RGB value is assumed.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # Teal/cyan occupies a broad hue interval.
    # OpenCV hue range = 0..179.
    hue_mask = (
        (h >= 70)
        & (h <= 105)
    )

    # The cap is more saturated than the gray/white background.
    saturation_mask = s >= 45

    # Remove extremely dark pixels.
    value_mask = v >= 45

    # Teal/cyan tends toward the blue-green side of Lab.
    lab_mask = (
        (b <= 145)
        & (a <= 145)
    )

    mask = (
        hue_mask
        & saturation_mask
        & value_mask
        & lab_mask
    )

    mask = (
        mask.astype(np.uint8)
        * 255
    )

    # Remove tiny isolated regions.
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1,
    )

    # Join nearby parts belonging to the cap.
    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_large,
        iterations=2,
    )

    # Fill small internal holes.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cleaned = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= 500:
            cv2.drawContours(
                cleaned,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED,
            )

    return cleaned


def candidate_score(
    contour: np.ndarray,
    image_shape: tuple[int, int],
    mask: np.ndarray,
) -> tuple[float, DetectionResult | None]:

    height, width = image_shape[:2]

    area = cv2.contourArea(contour)

    if area <= 0:
        return -1.0, None

    image_area = width * height
    occupancy = area / image_area

    # Reject tiny noise and huge background regions.
    if occupancy < 0.015:
        return -1.0, None

    if occupancy > 0.70:
        return -1.0, None

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return -1.0, None

    aspect_ratio = min(w, h) / max(w, h)

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    if perimeter <= 0:
        return -1.0, None

    circularity = (
        4.0
        * np.pi
        * area
        / (perimeter * perimeter)
    )

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return -1.0, None

    center_x = (
        moments["m10"]
        / moments["m00"]
    )

    center_y = (
        moments["m01"]
        / moments["m00"]
    )

    # Product should be reasonably central, but don't
    # require exact coordinates.
    frame_center_x = width / 2
    frame_center_y = height / 2

    dx = center_x - frame_center_x
    dy = center_y - frame_center_y

    distance = np.sqrt(
        dx * dx + dy * dy
    )

    maximum_distance = np.sqrt(
        frame_center_x ** 2
        + frame_center_y ** 2
    )

    centrality = max(
        0.0,
        1.0 - distance / maximum_distance,
    )

    # Calculate how much of the candidate bounding box
    # is actually teal/cyan.
    candidate_mask = mask[
        y:y + h,
        x:x + w,
    ]

    color_ratio = (
        np.count_nonzero(candidate_mask)
        / candidate_mask.size
        if candidate_mask.size
        else 0.0
    )

    # Score components.
    circularity_score = np.clip(
        circularity,
        0.0,
        1.0,
    )

    aspect_score = np.clip(
        aspect_ratio,
        0.0,
        1.0,
    )

    color_score = np.clip(
        color_ratio / 0.55,
        0.0,
        1.0,
    )

    # Prefer a reasonable cap size.
    if 0.04 <= occupancy <= 0.45:
        size_score = 1.0
    elif occupancy < 0.04:
        size_score = occupancy / 0.04
    else:
        size_score = max(
            0.0,
            1.0 - (occupancy - 0.45) / 0.25,
        )

    score = (
        0.25 * circularity_score
        + 0.20 * aspect_score
        + 0.30 * color_score
        + 0.15 * centrality
        + 0.10 * size_score
    )

    alignment = "VALID"

    # Position alignment.
    if centrality < 0.55:
        alignment = "POSITION_WARNING"

    # Very elongated candidate.
    if aspect_ratio < 0.65:
        alignment = "SHAPE_WARNING"

    result = DetectionResult(
        detected=True,
        confidence=float(score),
        bbox=(
            int(x),
            int(y),
            int(x + w),
            int(y + h),
        ),
        center=(
            float(center_x),
            float(center_y),
        ),
        occupancy=float(occupancy),
        circularity=float(circularity),
        aspect_ratio=float(aspect_ratio),
        color_ratio=float(color_ratio),
        alignment=alignment,
        reason="Teal/cyan product candidate detected.",
    )

    return score, result


def detect_product(
    image_path: str | Path,
    minimum_confidence: float = 0.45,
) -> DetectionResult:

    image_path = Path(image_path)

    image = load_image(image_path)

    mask = build_color_mask(image)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_score = -1.0
    best_result = None

    for contour in contours:

        score, result = candidate_score(
            contour,
            image.shape,
            mask,
        )

        if result is None:
            continue

        if score > best_score:
            best_score = score
            best_result = result

    if best_result is None:
        return DetectionResult(
            detected=False,
            confidence=0.0,
            bbox=None,
            center=None,
            occupancy=0.0,
            circularity=0.0,
            aspect_ratio=0.0,
            color_ratio=0.0,
            alignment="INVALID",
            reason="No teal/cyan product candidate detected.",
        )

    if best_result.confidence < minimum_confidence:
        best_result.detected = False
        best_result.alignment = "INVALID"
        best_result.reason = (
            "Candidate detected but confidence "
            "is below the minimum threshold."
        )

    return best_result


def save_debug_image(
    image_path: Path,
    result: DetectionResult,
    output_path: Path,
) -> None:

    image = load_image(image_path)

    if result.bbox is not None:

        x1, y1, x2, y2 = result.bbox

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4,
        )

    if result.center is not None:

        cx, cy = result.center

        cv2.circle(
            image,
            (
                int(round(cx)),
                int(round(cy)),
            ),
            10,
            (0, 0, 255),
            -1,
        )

    status = (
        "DETECTED"
        if result.detected
        else "NOT DETECTED"
    )

    lines = [
        f"Product: {status}",
        f"Confidence: {result.confidence:.3f}",
        f"Occupancy: {result.occupancy:.3f}",
        f"Circularity: {result.circularity:.3f}",
        f"Aspect ratio: {result.aspect_ratio:.3f}",
        f"Teal ratio: {result.color_ratio:.3f}",
        f"Alignment: {result.alignment}",
    ]

    y = 40

    for line in lines:

        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        y += 35

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        image,
    )


def save_mask(
    image_path: Path,
    output_path: Path,
) -> None:

    image = load_image(image_path)

    mask = build_color_mask(image)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        mask,
    )


def process_dataset() -> None:

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    images = []

    for category in (
        "normal",
        "defects",
    ):

        category_dir = (
            DATASET_DIR / category
        )

        if not category_dir.exists():
            print(
                f"WARNING: missing folder: "
                f"{category_dir}"
            )
            continue

        for path in sorted(
            category_dir.iterdir()
        ):

            if (
                path.is_file()
                and path.suffix.lower()
                in image_extensions
            ):
                images.append(
                    (category, path)
                )

    total = len(images)
    detected_count = 0

    print()
    print(
        "VISIONQC ADAPTIVE PRODUCT DETECTION"
    )
    print("=" * 60)
    print(
        f"Images found: {total}"
    )
    print()

    for index, (category, path) in enumerate(
        images,
        start=1,
    ):

        result = detect_product(path)

        if result.detected:
            detected_count += 1

        safe_name = (
            f"{category}_{index:02d}"
            f"_{path.stem}"
        )

        debug_path = (
            DEBUG_DIR
            / f"{safe_name}_debug.jpg"
        )

        mask_path = (
            DEBUG_DIR
            / f"{safe_name}_mask.png"
        )

        save_debug_image(
            path,
            result,
            debug_path,
        )

        save_mask(
            path,
            mask_path,
        )

        print(
            f"[{index:02d}/{total}] "
            f"{category:7s} "
            f"{path.name}"
        )

        print(
            f"    detected={result.detected} "
            f"confidence={result.confidence:.3f} "
            f"occupancy={result.occupancy:.3f} "
            f"circularity={result.circularity:.3f} "
            f"teal={result.color_ratio:.3f} "
            f"alignment={result.alignment}"
        )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Detected: {detected_count}/{total}"
    )

    if total:
        print(
            f"Detection rate: "
            f"{detected_count / total * 100:.1f}%"
        )

    print()
    print(
        f"Debug output: {DEBUG_DIR}"
    )


if __name__ == "__main__":
    process_dataset()