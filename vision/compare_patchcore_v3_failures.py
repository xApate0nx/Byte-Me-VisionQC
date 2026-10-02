from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROI_ROOT = PROJECT_ROOT / "data" / "evaluation" / "roi_comparison"
HEATMAP_ROOT = PROJECT_ROOT / "data" / "evaluation" / "patchcore_v3_heatmaps"

OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "patchcore_v3_failure_cases"

CASES = [
    ("NORMAL_FALSE_POSITIVE", "normal", "WhatsApp Image 2026-10-02 at 13.44.48"),
    ("NORMAL_FALSE_POSITIVE", "normal", "WhatsApp Image 2026-10-02 at 13.44.35"),

    ("DEFECT_MISSED", "defects", "WhatsApp Image 2026-10-02 at 13.51.07"),
    ("DEFECT_MISSED", "defects", "WhatsApp Image 2026-10-02 at 13.51.18 (3)"),
    ("DEFECT_MISSED", "defects", "WhatsApp Image 2026-10-02 at 13.51.10"),
    ("DEFECT_MISSED", "defects", "WhatsApp Image 2026-10-02 at 13.51.16 (2)"),
    ("DEFECT_MISSED", "defects", "WhatsApp Image 2026-10-02 at 13.51.17"),
]


def find_matching(folder, stem, suffix):
    matches = list(folder.glob(f"{stem}*{suffix}"))
    return matches[0] if matches else None


def label(image, text):
    image = image.copy()

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 40),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return image


def process(kind, category, stem):
    roi_dir = (
        ROI_ROOT / "new_roi"
        if category == "normal"
        else ROI_ROOT / "defects_new_roi"
    )

    heat_dir = HEATMAP_ROOT / category

    roi = find_matching(
        roi_dir,
        stem,
        ".png",
    )

    heatmap = find_matching(
        heat_dir,
        stem,
        "_heatmap.png",
    )

    anomaly = find_matching(
        heat_dir,
        stem,
        "_anomaly_map.npy",
    )

    if not roi or not heatmap or not anomaly:
        print(f"Missing file(s): {stem}")
        return None

    original = cv2.imread(str(roi))
    heat = cv2.imread(str(heatmap))
    anomaly_map = np.load(anomaly)

    original = cv2.resize(original, (512, 512))
    heat = cv2.resize(heat, (512, 512))

    overlay = cv2.addWeighted(
        original,
        0.55,
        heat,
        0.45,
        0,
    )

    y, x = np.unravel_index(
        np.argmax(anomaly_map),
        anomaly_map.shape,
    )

    px = int(x * 512 / anomaly_map.shape[1])
    py = int(y * 512 / anomaly_map.shape[0])

    cv2.circle(
        overlay,
        (px, py),
        10,
        (255, 255, 255),
        3,
    )

    cv2.circle(
        overlay,
        (px, py),
        4,
        (0, 0, 0),
        -1,
    )

    original = label(original, "CANONICAL ROI")
    heat = label(heat, "PATCHCORE V3 HEATMAP")
    overlay = label(
        overlay,
        f"MAX ANOMALY ({x},{y})",
    )

    combined = np.hstack(
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

    filename = (
        f"{kind}_{stem}_comparison.png"
    )

    path = OUTPUT / filename

    cv2.imwrite(
        str(path),
        combined,
    )

    print(path)

    return path


def main():
    print("=" * 80)
    print("PATCHCORE V3 — FAILURE CASE VISUALIZATION")
    print("=" * 80)

    for kind, category, stem in CASES:
        process(
            kind,
            category,
            stem,
        )

    print("\nDONE")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()