from pathlib import Path

import cv2
import numpy as np


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
    / "stable_region"
)

SIZE = 512


def load_normal(path):
    image = cv2.imread(str(path))

    if image is None:
        return None

    # Rotate portrait images into landscape.
    h, w = image.shape[:2]

    if h > w:
        image = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    image = cv2.resize(
        image,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return image


def main():

    print("=" * 70)
    print("VisionQC — Stable Product Region Diagnostic")
    print("=" * 70)

    files = sorted(
        list(NORMAL_DIR.glob("*.jpg"))
        + list(NORMAL_DIR.glob("*.jpeg"))
        + list(NORMAL_DIR.glob("*.png"))
    )

    print(f"\nNormal images: {len(files)}")

    images = []

    for path in files:

        image = load_normal(path)

        if image is not None:
            images.append(image)

    if not images:
        raise RuntimeError("No normal images found.")

    stack = np.stack(images).astype(np.float32)

    # Pixel-wise variation across normal images.
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)

    # Convert to grayscale variation.
    std_gray = np.mean(std, axis=2)

    # Low-variation regions are more visually consistent.
    normalized = cv2.normalize(
        std_gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    # Save variation visualization.
    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "normal_variation_heatmap.jpg"),
        heatmap,
    )

    # Mean image.
    mean_uint8 = np.clip(
        mean,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(OUTPUT_DIR / "normal_mean_image.jpg"),
        mean_uint8,
    )

    # Create a low-variation mask.
    threshold = np.percentile(
        std_gray,
        35,
    )

    stable_mask = (
        std_gray <= threshold
    ).astype(np.uint8) * 255

    kernel = np.ones(
        (15, 15),
        np.uint8,
    )

    stable_mask = cv2.morphologyEx(
        stable_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    stable_mask = cv2.morphologyEx(
        stable_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "stable_mask.jpg"),
        stable_mask,
    )

    # Bounding rectangle of the largest stable region.
    contours, _ = cv2.findContours(
        stable_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    print("\nLargest stable regions:")

    for i, contour in enumerate(contours[:10]):

        area = cv2.contourArea(contour)

        if area < SIZE * SIZE * 0.01:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        print(
            f"{i + 1:02d}. "
            f"area={area:.0f} "
            f"bbox=({x},{y})-({x+w},{y+h}) "
            f"size={w}x{h}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Draw the largest regions.
    visualization = mean_uint8.copy()

    for contour in contours[:5]:

        area = cv2.contourArea(contour)

        if area < SIZE * SIZE * 0.01:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            visualization,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3,
        )

    cv2.imwrite(
        str(OUTPUT_DIR / "stable_regions.jpg"),
        visualization,
    )

    print("\nResults:")
    print(OUTPUT_DIR)

    print("\nGenerated:")
    print("  normal_mean_image.jpg")
    print("  normal_variation_heatmap.jpg")
    print("  stable_mask.jpg")
    print("  stable_regions.jpg")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()