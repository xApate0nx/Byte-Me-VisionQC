from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_test"
)


def detect_product(image):
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth small lighting/noise variations
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Try both dark-object and bright-object segmentation
    _, threshold_dark = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    _, threshold_bright = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    candidates = []

    for mask in [threshold_dark, threshold_bright]:

        # Remove small noise
        kernel = np.ones((7, 7), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_area = image.shape[0] * image.shape[1]

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < image_area * 0.01:
                continue

            if area > image_area * 0.80:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            aspect = w / max(h, 1)

            # Product should not be an extremely thin strip
            if aspect < 0.20 or aspect > 5.0:
                continue

            candidates.append(
                (
                    area,
                    x,
                    y,
                    w,
                    h,
                )
            )

    if not candidates:
        return None

    # Largest plausible object
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, x, y, w, h = candidates[0]

    # Add padding
    padding_x = int(w * 0.15)
    padding_y = int(h * 0.15)

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)

    x2 = min(
        image.shape[1],
        x + w + padding_x,
    )

    y2 = min(
        image.shape[0],
        y + h + padding_y,
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def process(path, category):

    image = cv2.imread(str(path))

    if image is None:
        print(f"Could not read: {path}")
        return

    bbox = detect_product(image)

    visualization = image.copy()

    if bbox is None:

        cv2.putText(
            visualization,
            "NO ROI DETECTED",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )

        print(
            f"[FAIL] {category}/{path.name}"
        )

    else:

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            visualization,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5,
        )

        cv2.putText(
            visualization,
            f"ROI {x2-x1}x{y2-y1}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        print(
            f"[OK]   {category}/{path.name}"
            f" -> ({x1},{y1})-({x2},{y2})"
        )

    output_dir = OUTPUT_ROOT / category

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{path.stem}_roi.jpg"
    )

    cv2.imwrite(
        str(output_path),
        visualization,
    )


def main():

    print("=" * 70)
    print("VisionQC — ROI Detection Test")
    print("=" * 70)

    for category in ["normal", "defects"]:

        folder = DATA_ROOT / category

        files = sorted(
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
            + list(folder.glob("*.png"))
        )

        print(
            f"\n{category.upper()}: "
            f"{len(files)} images"
        )

        for path in files:

            process(
                path,
                category,
            )

    print("\n" + "=" * 70)
    print("ROI test completed.")
    print(f"Results: {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()