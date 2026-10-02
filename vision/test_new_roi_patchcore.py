from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NEW_ROI_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "new_roi"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "new_roi_patchcore_scores.txt"
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

    # We want the raw PatchCore distance.
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
    print("VisionQC — NEW ROI + EXISTING PATCHCORE TEST")
    print("=" * 70)

    if not NEW_ROI_DIR.exists():
        raise FileNotFoundError(
            f"New ROI directory not found:\n"
            f"{NEW_ROI_DIR}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"PatchCore checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    roi_files = sorted(
        [
            path
            for path in NEW_ROI_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }
        ]
    )

    if not roi_files:
        raise RuntimeError(
            "No ROI images found."
        )

    print()
    print(
        f"New ROI images : {len(roi_files)}"
    )

    print(
        f"Checkpoint     : {CHECKPOINT}"
    )

    print()

    results = []

    # --------------------------------------------------------
    # RUN ALL ROIS
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

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if not results:
        raise RuntimeError(
            "PatchCore produced no usable scores."
        )

    scores = np.array(
        [
            score
            for _, score in results
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    minimum = float(
        np.min(scores)
    )

    maximum = float(
        np.max(scores)
    )

    mean = float(
        np.mean(scores)
    )

    median = float(
        np.median(scores)
    )

    std = float(
        np.std(scores)
    )

    percentile_90 = float(
        np.percentile(
            scores,
            90,
        )
    )

    percentile_95 = float(
        np.percentile(
            scores,
            95,
        )
    )

    percentile_99 = float(
        np.percentile(
            scores,
            99,
        )
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("=" * 70)
    print("NEW ROI PATCHCORE RESULTS")
    print("=" * 70)

    print(
        f"Images tested : {len(results)}"
    )

    print(
        f"Minimum       : {minimum:.6f}"
    )

    print(
        f"Maximum       : {maximum:.6f}"
    )

    print(
        f"Mean          : {mean:.6f}"
    )

    print(
        f"Median        : {median:.6f}"
    )

    print(
        f"Std deviation : {std:.6f}"
    )

    print(
        f"90th percentile: {percentile_90:.6f}"
    )

    print(
        f"95th percentile: {percentile_95:.6f}"
    )

    print(
        f"99th percentile: {percentile_99:.6f}"
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
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "VisionQC — New ROI PatchCore Test\n"
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
            f"Maximum: {maximum:.6f}\n"
        )

        file.write(
            f"Mean: {mean:.6f}\n"
        )

        file.write(
            f"Median: {median:.6f}\n"
        )

        file.write(
            f"Std deviation: {std:.6f}\n"
        )

        file.write(
            f"90th percentile: "
            f"{percentile_90:.6f}\n"
        )

        file.write(
            f"95th percentile: "
            f"{percentile_95:.6f}\n"
        )

        file.write(
            f"99th percentile: "
            f"{percentile_99:.6f}\n"
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
        f"Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()


if __name__ == "__main__":
    main()