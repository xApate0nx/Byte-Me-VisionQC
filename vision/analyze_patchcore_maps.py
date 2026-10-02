from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HEATMAP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_heatmaps"
)


def analyze_image(path):
    image = cv2.imread(str(path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Approximate high-heat regions from the visualization
    threshold = np.percentile(gray, 95)

    mask = gray >= threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    width = image.shape[1]
    height = image.shape[0]

    return {
        "width": width,
        "height": height,
        "hot_pixels": len(xs),
        "hot_percent": len(xs) / (width * height) * 100,
        "x_center": (x_min + x_max) / 2,
        "y_center": (y_min + y_max) / 2,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def main():

    print("=" * 70)
    print("VisionQC — PatchCore Heatmap Analysis")
    print("=" * 70)

    for category in ["normal", "defects"]:

        folder = HEATMAP_ROOT / category

        files = sorted(folder.glob("*.jpg"))

        print("\n" + "=" * 70)
        print(category.upper())
        print("=" * 70)

        if not files:
            print("No files found.")
            continue

        for path in files:

            result = analyze_image(path)

            if result is None:
                continue

            print(
                f"\n{path.name}"
            )

            print(
                f"  image       : "
                f"{result['width']} x {result['height']}"
            )

            print(
                f"  hot pixels  : "
                f"{result['hot_pixels']}"
            )

            print(
                f"  hot area    : "
                f"{result['hot_percent']:.3f}%"
            )

            print(
                f"  hot center  : "
                f"({result['x_center']:.1f}, "
                f"{result['y_center']:.1f})"
            )

            print(
                f"  bbox        : "
                f"x={result['x_min']}-{result['x_max']} "
                f"y={result['y_min']}-{result['y_max']}"
            )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()