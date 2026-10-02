from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_ROI = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "new_roi"
)

DEFECT_ROI = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "defects_new_roi"
)

HEATMAP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_v3_heatmaps"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_v3_heatmap_comparisons"
)


CASES = {
    "normal": [
        "WhatsApp Image 2026-10-02 at 13.44.15",
        "WhatsApp Image 2026-10-02 at 13.44.43",
        "wsp",
    ],
    "defects": [
        "WhatsApp Image 2026-10-02 at 13.51.07",
        "WhatsApp Image 2026-10-02 at 13.51.14",
        "WhatsApp Image 2026-10-02 at 13.51.15",
    ],
}


def find_file(folder, stem, suffix):
    matches = list(folder.glob(f"{stem}*{suffix}"))

    if not matches:
        return None

    return matches[0]


def add_label(image, text):
    image = image.copy()

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 35),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return image


def make_comparison(category, stem):
    if category == "normal":
        roi_dir = NORMAL_ROI
    else:
        roi_dir = DEFECT_ROI

    roi = find_file(
        roi_dir,
        stem,
        ".png",
    )

    heatmap = find_file(
        HEATMAP_ROOT / category,
        stem,
        "_heatmap.png",
    )

    anomaly = find_file(
        HEATMAP_ROOT / category,
        stem,
        "_anomaly_map.npy",
    )

    if roi is None:
        print(f"ROI missing: {stem}")
        return

    if heatmap is None:
        print(f"Heatmap missing: {stem}")
        return

    if anomaly is None:
        print(f"Anomaly map missing: {stem}")
        return

    original = cv2.imread(str(roi))
    heat = cv2.imread(str(heatmap))
    anomaly_map = np.load(anomaly)

    if original is None or heat is None:
        print(f"Could not read image: {stem}")
        return

    original = cv2.resize(
        original,
        (512, 512),
    )

    heat = cv2.resize(
        heat,
        (512, 512),
    )

    # Blend ROI and heatmap.
    overlay = cv2.addWeighted(
        original,
        0.55,
        heat,
        0.45,
        0,
    )

    # Find strongest anomaly point.
    y, x = np.unravel_index(
        np.argmax(anomaly_map),
        anomaly_map.shape,
    )

    scale_x = 512 / anomaly_map.shape[1]
    scale_y = 512 / anomaly_map.shape[0]

    px = int(x * scale_x)
    py = int(y * scale_y)

    cv2.circle(
        overlay,
        (px, py),
        8,
        (255, 255, 255),
        2,
    )

    cv2.circle(
        overlay,
        (px, py),
        3,
        (0, 0, 0),
        -1,
    )

    original = add_label(
        original,
        "CANONICAL ROI",
    )

    heat = add_label(
        heat,
        "PATCHCORE V3 HEATMAP",
    )

    overlay = add_label(
        overlay,
        f"HIGHEST ANOMALY: ({x}, {y})",
    )

    comparison = np.hstack(
        [
            original,
            heat,
            overlay,
        ]
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT
        / f"{category}_{stem}_comparison.png"
    )

    cv2.imwrite(
        str(output_path),
        comparison,
    )

    print(f"Created: {output_path}")


def main():
    print("=" * 80)
    print("PATCHCORE V3 HEATMAP COMPARISON")
    print("=" * 80)

    for category, stems in CASES.items():
        for stem in stems:
            make_comparison(
                category,
                stem,
            )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"\nOutput folder:")
    print(OUTPUT)


if __name__ == "__main__":
    main()