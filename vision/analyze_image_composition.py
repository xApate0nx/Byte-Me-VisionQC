from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
)


def analyze(path, category):

    image = cv2.imread(str(path))

    if image is None:
        return

    height, width = image.shape[:2]

    # Print the image dimensions and center.
    print(
        f"{category:8s} | "
        f"{path.name:65s} | "
        f"{width:4d}x{height:<4d} | "
        f"center=({width/2:.0f},{height/2:.0f})"
    )


def main():

    print("=" * 100)
    print("VisionQC — Dataset Composition Diagnostic")
    print("=" * 100)

    for category in ["normal", "defects"]:

        folder = DATA_ROOT / category

        files = sorted(
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
            + list(folder.glob("*.png"))
        )

        print(f"\n{category.upper()} — {len(files)} images")

        for path in files:
            analyze(path, category)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()