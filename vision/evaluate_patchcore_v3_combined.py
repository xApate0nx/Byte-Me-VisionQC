from pathlib import Path
import sys
import json
import numpy as np

# ------------------------------------------------------------
# Make project root importable when running:
# python vision\evaluate_patchcore_v3_combined.py
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# VisionQC — PatchCore v3 Combined Evaluation
#
# Pipeline:
#
# Original image
#       ↓
# Product detector
#       ↓
# Canonical ROI
#       ↓
# PatchCore v3
#       ↓
# Raw anomaly score
#
# Evaluates:
#   21 NORMAL canonical ROIs
#   23 DEFECT canonical ROIs
#
# Also performs threshold analysis.
#
# IMPORTANT:
#   This script does NOT modify the production threshold.
# ============================================================


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap_v3"
    / "Patchcore"
    / "water_cap_v1_canonical_roi"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "new_roi"
)

DEFECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "defects_new_roi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_v3"
)


# ------------------------------------------------------------
# Expected dataset sizes
# ------------------------------------------------------------

EXPECTED_NORMAL = 21
EXPECTED_DEFECT = 23


# ============================================================
# PatchCore inference
# ============================================================

def create_model():

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )

    # Disable Anomalib post-processing.
    # We want the raw PatchCore anomaly distance.
    model.post_processor = None

    return model


def run_patchcore(image_path):

    model = create_model()

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
            f"No prediction returned for:\n{image_path}"
        )

    prediction = predictions[0]

    score = prediction.pred_score

    if hasattr(score, "item"):
        score = score.item()

    return float(score)


# ============================================================
# Dataset evaluation
# ============================================================

def evaluate_directory(directory, expected_count, label):

    images = sorted(
        list(directory.glob("*.png"))
        + list(directory.glob("*.jpg"))
        + list(directory.glob("*.jpeg"))
    )

    print(f"\n{label} images found: {len(images)}")

    if len(images) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {label.lower()} images, "
            f"found {len(images)}"
        )

    results = []

    print(f"\nRunning PatchCore v3 on {label.lower()}...\n")

    for index, image_path in enumerate(images, start=1):

        score = run_patchcore(image_path)

        result = {
            "filename": image_path.name,
            "score": score,
            "label": label.lower(),
        }

        results.append(result)

        print(
            f"{index:02d}/{expected_count}  "
            f"{image_path.name:<65} "
            f"score={score:.6f}"
        )

    return results


# ============================================================
# Statistics
# ============================================================

def calculate_statistics(results):

    scores = np.array(
        [item["score"] for item in results],
        dtype=float,
    )

    return {
        "count": int(len(scores)),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std()),
        "p10": float(np.percentile(scores, 10)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


# ============================================================
# Threshold analysis
# ============================================================

def analyze_threshold(
    normal_results,
    defect_results,
    threshold,
):

    normal_scores = np.array(
        [item["score"] for item in normal_results],
        dtype=float,
    )

    defect_scores = np.array(
        [item["score"] for item in defect_results],
        dtype=float,
    )

    # Score < threshold = PASS
    # Score >= threshold = FAIL

    normal_pass = int(np.sum(normal_scores < threshold))
    normal_fail = int(np.sum(normal_scores >= threshold))

    defect_pass = int(np.sum(defect_scores < threshold))
    defect_fail = int(np.sum(defect_scores >= threshold))

    total = len(normal_scores) + len(defect_scores)

    correct = normal_pass + defect_fail

    accuracy = correct / total

    false_rejection_rate = (
        normal_fail / len(normal_scores)
    )

    defect_miss_rate = (
        defect_pass / len(defect_scores)
    )

    defect_detection_rate = (
        defect_fail / len(defect_scores)
    )

    return {
        "threshold": float(threshold),

        "normal_pass": normal_pass,
        "normal_fail": normal_fail,

        "defect_pass": defect_pass,
        "defect_fail": defect_fail,

        "accuracy": float(accuracy),

        "false_rejection_rate": float(
            false_rejection_rate
        ),

        "defect_miss_rate": float(
            defect_miss_rate
        ),

        "defect_detection_rate": float(
            defect_detection_rate
        ),
    }


# ============================================================
# Find useful threshold candidates
# ============================================================

def generate_thresholds(normal_results, defect_results):

    normal_scores = sorted(
        [item["score"] for item in normal_results]
    )

    defect_scores = sorted(
        [item["score"] for item in defect_results]
    )

    all_scores = sorted(
        normal_scores + defect_scores
    )

    candidates = set()

    # --------------------------------------------------------
    # Midpoints between every adjacent observed score.
    #
    # These are the points where classification changes.
    # --------------------------------------------------------

    for a, b in zip(all_scores[:-1], all_scores[1:]):

        if a != b:

            midpoint = (a + b) / 2.0

            candidates.add(
                round(midpoint, 6)
            )

    # --------------------------------------------------------
    # Also include boundaries around observed scores.
    # --------------------------------------------------------

    candidates.add(
        round(min(all_scores) - 0.001, 6)
    )

    candidates.add(
        round(max(all_scores) + 0.001, 6)
    )

    return sorted(candidates)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("VISIONQC — PATCHCORE V3 COMBINED EVALUATION")
    print("=" * 80)

    print("\nCheckpoint:")
    print(CHECKPOINT)

    print("\nNormal ROI directory:")
    print(NORMAL_DIR)

    print("\nDefect ROI directory:")
    print(DEFECT_DIR)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    if not NORMAL_DIR.exists():

        raise FileNotFoundError(
            f"Normal ROI directory not found:\n{NORMAL_DIR}"
        )

    if not DEFECT_DIR.exists():

        raise FileNotFoundError(
            f"Defect ROI directory not found:\n{DEFECT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Evaluate normal images
    # --------------------------------------------------------

    normal_results = evaluate_directory(
        NORMAL_DIR,
        EXPECTED_NORMAL,
        "NORMAL",
    )

    # --------------------------------------------------------
    # Evaluate defect images
    # --------------------------------------------------------

    defect_results = evaluate_directory(
        DEFECT_DIR,
        EXPECTED_DEFECT,
        "DEFECT",
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    normal_stats = calculate_statistics(
        normal_results
    )

    defect_stats = calculate_statistics(
        defect_results
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    thresholds = generate_thresholds(
        normal_results,
        defect_results,
    )

    threshold_results = []

    for threshold in thresholds:

        result = analyze_threshold(
            normal_results,
            defect_results,
            threshold,
        )

        threshold_results.append(result)

    # --------------------------------------------------------
    # Find thresholds with zero normal false rejection
    # --------------------------------------------------------

    zero_normal_rejection = [
        result
        for result in threshold_results
        if result["normal_fail"] == 0
    ]

    # Among those, find the threshold with the highest
    # defect detection rate.
    #
    # This is descriptive analysis only.
    # It does NOT change the production threshold.
    # --------------------------------------------------------

    best_zero_normal = None

    if zero_normal_rejection:

        best_zero_normal = max(
            zero_normal_rejection,
            key=lambda x: (
                x["defect_detection_rate"],
                -x["threshold"],
            ),
        )

    # --------------------------------------------------------
    # Find maximum observed normal score
    # --------------------------------------------------------

    normal_max = normal_stats["maximum"]

    # --------------------------------------------------------
    # Find minimum observed defect score
    # --------------------------------------------------------

    defect_min = defect_stats["minimum"]

    # --------------------------------------------------------
    # Perfect separation check
    # --------------------------------------------------------

    perfect_separation = (
        normal_max < defect_min
    )

    separation_gap = (
        defect_min - normal_max
    )

    # --------------------------------------------------------
    # Save normal scores
    # --------------------------------------------------------

    normal_output = {
        "model": "PatchCore v3",
        "checkpoint": str(CHECKPOINT),
        "representation": (
            "detector-generated canonical ROI"
        ),
        "count": EXPECTED_NORMAL,
        "statistics": normal_stats,
        "results": normal_results,
    }

    with open(
        OUTPUT_DIR / "normal_scores.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            normal_output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save defect scores
    # --------------------------------------------------------

    defect_output = {
        "model": "PatchCore v3",
        "checkpoint": str(CHECKPOINT),
        "representation": (
            "detector-generated canonical ROI"
        ),
        "count": EXPECTED_DEFECT,
        "statistics": defect_stats,
        "results": defect_results,
    }

    with open(
        OUTPUT_DIR / "defect_scores.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            defect_output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save threshold analysis
    # --------------------------------------------------------

    threshold_output = {

        "model": "PatchCore v3",

        "checkpoint": str(CHECKPOINT),

        "normal_count": EXPECTED_NORMAL,

        "defect_count": EXPECTED_DEFECT,

        "normal_statistics": normal_stats,

        "defect_statistics": defect_stats,

        "perfect_separation": perfect_separation,

        "separation_gap": float(
            separation_gap
        ),

        "normal_max": float(
            normal_max
        ),

        "defect_min": float(
            defect_min
        ),

        "zero_normal_rejection_candidates": (
            zero_normal_rejection
        ),

        "best_zero_normal_rejection_candidate": (
            best_zero_normal
        ),

        "all_thresholds": threshold_results,
    }

    with open(
        OUTPUT_DIR / "threshold_analysis.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            threshold_output,
            f,
            indent=2,
        )

    # ========================================================
    # Console summary
    # ========================================================

    print("\n")
    print("=" * 80)
    print("PATCHCORE V3 — COMBINED RESULTS")
    print("=" * 80)

    print("\nNORMAL")
    print("-" * 80)

    print(
        f"Count  : {normal_stats['count']}"
    )

    print(
        f"Min    : {normal_stats['minimum']:.6f}"
    )

    print(
        f"Median : {normal_stats['median']:.6f}"
    )

    print(
        f"Mean   : {normal_stats['mean']:.6f}"
    )

    print(
        f"90%    : {normal_stats['p90']:.6f}"
    )

    print(
        f"95%    : {normal_stats['p95']:.6f}"
    )

    print(
        f"Max    : {normal_stats['maximum']:.6f}"
    )

    print("\nDEFECT")
    print("-" * 80)

    print(
        f"Count  : {defect_stats['count']}"
    )

    print(
        f"Min    : {defect_stats['minimum']:.6f}"
    )

    print(
        f"Median : {defect_stats['median']:.6f}"
    )

    print(
        f"Mean   : {defect_stats['mean']:.6f}"
    )

    print(
        f"90%    : {defect_stats['p90']:.6f}"
    )

    print(
        f"Max    : {defect_stats['maximum']:.6f}"
    )

    print("\nSEPARATION")
    print("-" * 80)

    print(
        f"Highest normal score : "
        f"{normal_max:.6f}"
    )

    print(
        f"Lowest defect score  : "
        f"{defect_min:.6f}"
    )

    print(
        f"Gap (defect - normal): "
        f"{separation_gap:.6f}"
    )

    print(
        f"Perfect separation   : "
        f"{perfect_separation}"
    )

    # --------------------------------------------------------
    # Zero-normal-rejection candidate
    # --------------------------------------------------------

    print("\nTHRESHOLD ANALYSIS")
    print("-" * 80)

    if best_zero_normal is not None:

        print(
            "\nHighest defect detection rate "
            "while rejecting ZERO known normal samples:"
        )

        print(
            f"Threshold             : "
            f"{best_zero_normal['threshold']:.6f}"
        )

        print(
            f"Normal rejected       : "
            f"{best_zero_normal['normal_fail']}/"
            f"{EXPECTED_NORMAL}"
        )

        print(
            f"Defects detected      : "
            f"{best_zero_normal['defect_fail']}/"
            f"{EXPECTED_DEFECT}"
        )

        print(
            f"Defect detection rate : "
            f"{best_zero_normal['defect_detection_rate'] * 100:.2f}%"
        )

        print(
            f"Defect miss rate       : "
            f"{best_zero_normal['defect_miss_rate'] * 100:.2f}%"
        )

    else:

        print(
            "No threshold produced zero "
            "normal false rejections."
        )

    # --------------------------------------------------------
    # Old threshold reminder
    # --------------------------------------------------------

    print("\nOLD PRODUCTION THRESHOLD")
    print("-" * 80)
    print("12.3785223961")
    print("DO NOT USE THIS THRESHOLD FOR V3.")

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    print("\nOUTPUT FILES")
    print("-" * 80)

    print(
        OUTPUT_DIR / "normal_scores.json"
    )

    print(
        OUTPUT_DIR / "defect_scores.json"
    )

    print(
        OUTPUT_DIR / "threshold_analysis.json"
    )

    print("\n" + "=" * 80)
    print("COMBINED V3 EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()