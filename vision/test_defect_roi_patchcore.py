from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PATHS
# ============================================================

DEFECT_ROI_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "defects_new_roi"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "defect_new_roi_patchcore_scores.txt"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap_v2"
    / "Patchcore"
    / "water_cap_v1_roi80"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)


# ============================================================
# PATCHCORE
# ============================================================

def run_patchcore(
    image_path: Path,
) -> float:

    model = Patchcore(
        backbone="resnet18",
        layers=[
            "layer2",
            "layer3",
        ],
        pre_trained=True,
        num_neighbors=9,
    )

    # Raw PatchCore distance.
    model.post_processor = None

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    dataset = PredictDataset(
        path=str(image_path),
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT),
    )

    if not predictions:
        raise RuntimeError(
            "PatchCore returned no predictions."
        )

    prediction = predictions[0]

    score = prediction.pred_score

    if hasattr(score, "item"):
        score = score.item()

    return float(score)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VisionQC — DEFECT NEW ROI + EXISTING PATCHCORE TEST")
    print("=" * 70)

    if not DEFECT_ROI_DIR.exists():
        raise FileNotFoundError(
            f"Defect ROI directory not found:\n"
            f"{DEFECT_ROI_DIR}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"PatchCore checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    roi_files = sorted(
        [
            path
            for path in DEFECT_ROI_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }
        ]
    )

    if not roi_files:
        raise RuntimeError(
            "No defect ROI images found."
        )

    print()
    print(
        f"Defect ROI images : {len(roi_files)}"
    )

    print(
        f"Checkpoint        : {CHECKPOINT}"
    )

    print()

    results = []

    # --------------------------------------------------------
    # RUN PATCHCORE
    # --------------------------------------------------------

    for index, image_path in enumerate(
        roi_files,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(roi_files)}] "
            f"{image_path.name}"
        )

        try:

            score = run_patchcore(
                image_path
            )

            results.append(
                (
                    image_path.name,
                    score,
                )
            )

            print(
                f"       RAW score: "
                f"{score:.6f}"
            )

        except Exception as exc:

            print(
                f"       ERROR: {exc}"
            )

        print()

    if not results:
        raise RuntimeError(
            "PatchCore produced no usable scores."
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    scores = np.array(
        [
            score
            for _, score in results
        ],
        dtype=np.float64,
    )

    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    mean = float(np.mean(scores))
    median = float(np.median(scores))
    std = float(np.std(scores))

    percentile_10 = float(
        np.percentile(scores, 10)
    )

    percentile_25 = float(
        np.percentile(scores, 25)
    )

    percentile_75 = float(
        np.percentile(scores, 75)
    )

    percentile_90 = float(
        np.percentile(scores, 90)
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("=" * 70)
    print("DEFECT NEW ROI PATCHCORE RESULTS")
    print("=" * 70)

    print(
        f"Images tested : {len(results)}"
    )

    print(
        f"Minimum       : {minimum:.6f}"
    )

    print(
        f"10th percentile: {percentile_10:.6f}"
    )

    print(
        f"25th percentile: {percentile_25:.6f}"
    )

    print(
        f"Median        : {median:.6f}"
    )

    print(
        f"Mean          : {mean:.6f}"
    )

    print(
        f"75th percentile: {percentile_75:.6f}"
    )

    print(
        f"90th percentile: {percentile_90:.6f}"
    )

    print(
        f"Maximum       : {maximum:.6f}"
    )

    print(
        f"Std deviation : {std:.6f}"
    )

    print()

    print(
        "CURRENT OLD THRESHOLD"
    )

    print(
        "12.3785223961"
    )

    print()

    print(
        "Individual scores:"
    )

    for filename, score in results:

        print(
            f"  {filename}"
            f" -> {score:.6f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "VisionQC — Defect New ROI PatchCore Test\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Images tested: {len(results)}\n"
        )

        file.write(
            f"Minimum: {minimum:.6f}\n"
        )

        file.write(
            f"10th percentile: {percentile_10:.6f}\n"
        )

        file.write(
            f"25th percentile: {percentile_25:.6f}\n"
        )

        file.write(
            f"Median: {median:.6f}\n"
        )

        file.write(
            f"Mean: {mean:.6f}\n"
        )

        file.write(
            f"75th percentile: {percentile_75:.6f}\n"
        )

        file.write(
            f"90th percentile: {percentile_90:.6f}\n"
        )

        file.write(
            f"Maximum: {maximum:.6f}\n"
        )

        file.write(
            f"Std deviation: {std:.6f}\n"
        )

        file.write("\n")
        file.write("Individual scores:\n")

        for filename, score in results:

            file.write(
                f"{filename}\t"
                f"{score:.6f}\n"
            )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()


if __name__ == "__main__":
    main()