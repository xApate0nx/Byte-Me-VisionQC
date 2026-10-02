from pathlib import Path
import numpy as np
from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROI_DIR = PROJECT_ROOT / "data" / "inspections" / "_temporary_roi"


FILES = [
    ("GOOD_13.44.15", "WhatsApp Image 2026-10-02 at 13.44.15_roi80.jpg"),
    ("GOOD_13.44.45", "WhatsApp Image 2026-10-02 at 13.44.45_roi80.jpg"),
    ("FALSE_13.44.48", "WhatsApp Image 2026-10-02 at 13.44.48_roi80.jpg"),
    ("FALSE_wsp", "wsp_roi80.jpg"),
]


def analyze_image(label, filename):
    path = ROI_DIR / filename

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    if not path.exists():
        print("ERROR — file not found:")
        print(path)
        return

    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32)

    h, w = arr.shape[:2]

    print(f"Image size        : {w} x {h}")

    # Overall brightness
    gray = (
        0.299 * arr[:, :, 0]
        + 0.587 * arr[:, :, 1]
        + 0.114 * arr[:, :, 2]
    )

    print(f"Brightness mean   : {gray.mean():.2f}")
    print(f"Brightness std    : {gray.std():.2f}")
    print(f"Brightness min    : {gray.min():.2f}")
    print(f"Brightness max    : {gray.max():.2f}")

    # RGB means
    print(f"Red mean          : {arr[:, :, 0].mean():.2f}")
    print(f"Green mean        : {arr[:, :, 1].mean():.2f}")
    print(f"Blue mean         : {arr[:, :, 2].mean():.2f}")

    # Contrast between center and border
    center = gray[
        h // 4 : 3 * h // 4,
        w // 4 : 3 * w // 4
    ]

    border_mask = np.ones_like(gray, dtype=bool)

    border_mask[
        h // 4 : 3 * h // 4,
        w // 4 : 3 * w // 4
    ] = False

    border = gray[border_mask]

    print(f"Center brightness : {center.mean():.2f}")
    print(f"Border brightness : {border.mean():.2f}")
    print(f"Center-border diff: {abs(center.mean() - border.mean()):.2f}")

    # Simple edge strength
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))

    edge_strength = (dx.mean() + dy.mean()) / 2

    print(f"Edge strength     : {edge_strength:.2f}")

    # Bright/dark pixel percentages
    bright = (gray > 220).mean() * 100
    dark = (gray < 50).mean() * 100

    print(f"Very bright pixels: {bright:.2f}%")
    print(f"Very dark pixels  : {dark:.2f}%")

    # Image quadrant means
    print("\nQuadrant brightness:")

    for row in range(2):
        values = []

        y1 = row * h // 2
        y2 = (row + 1) * h // 2

        for col in range(2):
            x1 = col * w // 2
            x2 = (col + 1) * w // 2

            region = gray[y1:y2, x1:x2]
            values.append(f"{region.mean():.1f}")

        print(" | ".join(values))


def main():
    print("=" * 70)
    print("VisionQC — False Positive Image Comparison")
    print("=" * 70)

    for label, filename in FILES:
        analyze_image(label, filename)


if __name__ == "__main__":
    main()