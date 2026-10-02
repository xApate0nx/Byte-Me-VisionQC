from pathlib import Path
import json
import random
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCORE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_v3"
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
    / "patchcore_v3_holdout_evaluation.json"
)

SEED = 42
HOLDOUT_RATIO = 0.30


# ============================================================
# LOAD PATCHCORE SCORES
# ============================================================

def load_scores(filename):
    path = SCORE_ROOT / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}

    for item in data["results"]:
        filename = item["filename"]
        name = Path(filename).stem

        results[name] = {
            "score": float(item["score"]),
            "filename": filename,
        }

    return results


# ============================================================
# SPATIAL FEATURES
# ============================================================

def spatial_features(path):
    heatmap = np.load(path).astype(np.float32)

    h, w = heatmap.shape

    # Normalize
    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))

    if maximum > minimum:
        normalized = (
            (heatmap - minimum)
            / (maximum - minimum)
        )
    else:
        normalized = np.zeros_like(heatmap)

    # Top 5% anomaly pixels
    p95 = np.percentile(heatmap, 95)

    mask = heatmap >= p95

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return {
            "central_fraction": 0.0,
            "border_fraction": 0.0,
            "centroid_x": 0.5,
            "centroid_y": 0.5,
            "centroid_distance": 0.0,
            "largest_region_fraction": 0.0,
            "anomaly_mass": 0.0,
        }

    cx = w / 2.0
    cy = h / 2.0

    # --------------------------------------------------------
    # Central 60%
    # --------------------------------------------------------

    central = (
        (np.abs(xs - cx) <= 0.30 * w)
        & (np.abs(ys - cy) <= 0.30 * h)
    )

    central_fraction = float(np.mean(central))

    # --------------------------------------------------------
    # Border 20%
    # --------------------------------------------------------

    border = (
        (xs < 0.20 * w)
        | (xs >= 0.80 * w)
        | (ys < 0.20 * h)
        | (ys >= 0.80 * h)
    )

    border_fraction = float(np.mean(border))

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    centroid_x = float(np.mean(xs) / w)
    centroid_y = float(np.mean(ys) / h)

    centroid_distance = float(
        np.sqrt(
            (centroid_x - 0.5) ** 2
            + (centroid_y - 0.5) ** 2
        )
    )

    # --------------------------------------------------------
    # Connected regions
    # --------------------------------------------------------

    import cv2

    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8,
        )
    )

    if num_labels > 1:
        component_sizes = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        largest_component = int(
            np.max(component_sizes)
        )
    else:
        largest_component = 0

    largest_region_fraction = (
        largest_component / len(xs)
    )

    # --------------------------------------------------------
    # Total normalized anomaly mass
    # --------------------------------------------------------

    anomaly_mass = float(
        np.mean(normalized)
    )

    return {
        "central_fraction": central_fraction,
        "border_fraction": border_fraction,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_distance": centroid_distance,
        "largest_region_fraction": largest_region_fraction,
        "anomaly_mass": anomaly_mass,
    }


def load_spatial(category):
    directory = HEATMAP_ROOT / category

    results = {}

    for path in sorted(directory.glob("*.npy")):

        name = path.stem.replace(
            "_anomaly_map",
            "",
        )

        results[name] = spatial_features(path)

    return results


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset():
    normal_scores = load_scores(
        "normal_scores.json"
    )

    defect_scores = load_scores(
        "defect_scores.json"
    )

    normal_spatial = load_spatial("normal")
    defect_spatial = load_spatial("defects")

    dataset = []

    for name, item in normal_scores.items():

        if name not in normal_spatial:
            continue

        dataset.append({
            "name": name,
            "filename": item["filename"],
            "label": 0,
            "label_name": "normal",
            "score": item["score"],
            **normal_spatial[name],
        })

    for name, item in defect_scores.items():

        if name not in defect_spatial:
            continue

        dataset.append({
            "name": name,
            "filename": item["filename"],
            "label": 1,
            "label_name": "defect",
            "score": item["score"],
            **defect_spatial[name],
        })

    return dataset


# ============================================================
# STRATIFIED HOLDOUT
# ============================================================

def split_dataset(dataset):

    random.seed(SEED)

    normal = [
        x for x in dataset
        if x["label"] == 0
    ]

    defects = [
        x for x in dataset
        if x["label"] == 1
    ]

    random.shuffle(normal)
    random.shuffle(defects)

    normal_holdout_count = max(
        1,
        round(len(normal) * HOLDOUT_RATIO)
    )

    defect_holdout_count = max(
        1,
        round(len(defects) * HOLDOUT_RATIO)
    )

    normal_holdout = normal[
        :normal_holdout_count
    ]

    defect_holdout = defects[
        :defect_holdout_count
    ]

    normal_train = normal[
        normal_holdout_count:
    ]

    defect_train = defects[
        defect_holdout_count:
    ]

    calibration = (
        normal_train
        + defect_train
    )

    holdout = (
        normal_holdout
        + defect_holdout
    )

    return calibration, holdout


# ============================================================
# METRICS
# ============================================================

def metrics(rows, predictor):

    normal = [
        r for r in rows
        if r["label"] == 0
    ]

    defects = [
        r for r in rows
        if r["label"] == 1
    ]

    normal_fail = sum(
        predictor(r)
        for r in normal
    )

    defect_detect = sum(
        predictor(r)
        for r in defects
    )

    normal_pass = (
        len(normal) - normal_fail
    )

    defect_miss = (
        len(defects) - defect_detect
    )

    total = len(normal) + len(defects)

    correct = (
        normal_pass
        + defect_detect
    )

    return {
        "normal_fail": normal_fail,
        "normal_total": len(normal),
        "defect_detect": defect_detect,
        "defect_total": len(defects),
        "false_reject_rate": (
            normal_fail / len(normal)
            if normal else 0
        ),
        "defect_miss_rate": (
            defect_miss / len(defects)
            if defects else 0
        ),
        "accuracy": correct / total
        if total else 0,
    }


# ============================================================
# CALIBRATION
# ============================================================

def calibrate(calibration):

    score_values = [
        r["score"]
        for r in calibration
    ]

    spatial_values = [
        r["central_fraction"]
        for r in calibration
    ]

    score_thresholds = np.arange(
        10.0,
        18.51,
        0.25,
    )

    spatial_thresholds = np.arange(
        0.30,
        0.86,
        0.05,
    )

    candidates = []

    for score_threshold in score_thresholds:

        for spatial_threshold in spatial_thresholds:

            def combined_or(row):
                return (
                    row["score"] >= score_threshold
                    or
                    row["central_fraction"]
                    >= spatial_threshold
                )

            result = metrics(
                calibration,
                combined_or,
            )

            result.update({
                "mode": "OR",
                "score_threshold": float(
                    score_threshold
                ),
                "spatial_threshold": float(
                    spatial_threshold
                ),
            })

            candidates.append(result)

    # Prefer:
    # 1. low false rejection
    # 2. high defect detection
    # 3. high accuracy
    candidates.sort(
        key=lambda x: (
            x["false_reject_rate"],
            -(
                x["defect_detect"]
                / x["defect_total"]
            ),
            -x["accuracy"],
        )
    )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("PATCHCORE V3 HOLDOUT EVALUATION")
    print("=" * 80)

    dataset = build_dataset()

    print()
    print(
        f"Total usable samples: {len(dataset)}"
    )

    calibration, holdout = split_dataset(
        dataset
    )

    print(
        f"Calibration samples : {len(calibration)}"
    )

    print(
        f"Holdout samples     : {len(holdout)}"
    )

    print()
    print("CALIBRATION")
    print("-" * 80)

    calibration_normal = sum(
        r["label"] == 0
        for r in calibration
    )

    calibration_defect = sum(
        r["label"] == 1
        for r in calibration
    )

    print(
        "Normal :",
        calibration_normal
    )

    print(
        "Defect :",
        calibration_defect
    )

    print()
    print("HOLDOUT")
    print("-" * 80)

    holdout_normal = sum(
        r["label"] == 0
        for r in holdout
    )

    holdout_defect = sum(
        r["label"] == 1
        for r in holdout
    )

    print(
        "Normal :",
        holdout_normal
    )

    print(
        "Defect :",
        holdout_defect
    )

    # --------------------------------------------------------
    # Baseline: PatchCore only
    # --------------------------------------------------------

    calibration_scores = sorted(
        r["score"]
        for r in calibration
    )

    # Test every midpoint between calibration scores
    score_thresholds = []

    for a, b in zip(
        calibration_scores[:-1],
        calibration_scores[1:],
    ):
        score_thresholds.append(
            (a + b) / 2
        )

    baseline_candidates = []

    for threshold in score_thresholds:

        def predictor(row):
            return row["score"] >= threshold

        calibration_result = metrics(
            calibration,
            predictor,
        )

        holdout_result = metrics(
            holdout,
            predictor,
        )

        baseline_candidates.append({
            "threshold": threshold,
            "calibration": calibration_result,
            "holdout": holdout_result,
        })

    # --------------------------------------------------------
    # Pick calibration rule:
    # minimum false rejection, then maximum
    # defect detection.
    # --------------------------------------------------------

    baseline_candidates.sort(
        key=lambda x: (
            x["calibration"]["false_reject_rate"],
            -(
                x["calibration"]["defect_detect"]
                / x["calibration"]["defect_total"]
            ),
        )
    )

    baseline = baseline_candidates[0]

    # --------------------------------------------------------
    # Combined calibration
    # --------------------------------------------------------

    combined_candidates = calibrate(
        calibration
    )

    combined = combined_candidates[0]

    score_threshold = combined[
        "score_threshold"
    ]

    spatial_threshold = combined[
        "spatial_threshold"
    ]

    def combined_predictor(row):
        return (
            row["score"] >= score_threshold
            or
            row["central_fraction"]
            >= spatial_threshold
        )

    combined_calibration_metrics = metrics(
        calibration,
        combined_predictor,
    )

    combined_holdout_metrics = metrics(
        holdout,
        combined_predictor,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PATCHCORE-ONLY HOLDOUT")
    print("=" * 80)

    print(
        "Calibration threshold:",
        f"{baseline['threshold']:.4f}"
    )

    print(
        "Calibration:",
        baseline["calibration"]
    )

    print(
        "HOLDOUT:",
        baseline["holdout"]
    )

    print()
    print("=" * 80)
    print("COMBINED SCORE + SPATIAL HOLDOUT")
    print("=" * 80)

    print(
        "Score threshold:",
        f"{score_threshold:.4f}"
    )

    print(
        "Spatial threshold:",
        f"{spatial_threshold:.2f}"
    )

    print()
    print("Calibration:")
    print(
        combined_calibration_metrics
    )

    print()
    print("HOLDOUT:")
    print(
        combined_holdout_metrics
    )

    # --------------------------------------------------------
    # Holdout sample list
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("HOLDOUT SAMPLES")
    print("=" * 80)

    for row in sorted(
        holdout,
        key=lambda x: (
            x["label"],
            x["filename"],
        ),
    ):

        print(
            f"{row['label_name']:7} | "
            f"score={row['score']:7.3f} | "
            f"central={row['central_fraction']:.3f} | "
            f"{row['filename']}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "seed": SEED,
        "holdout_ratio": HOLDOUT_RATIO,
        "total_samples": len(dataset),
        "calibration_count": len(calibration),
        "holdout_count": len(holdout),
        "baseline_patchcore_only": baseline,
        "combined_score_spatial": {
            "score_threshold": score_threshold,
            "spatial_threshold": spatial_threshold,
            "calibration": combined_calibration_metrics,
            "holdout": combined_holdout_metrics,
        },
        "calibration_samples": calibration,
        "holdout_samples": holdout,
    }

    OUTPUT.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("Saved:")
    print(OUTPUT)
    print("=" * 80)


if __name__ == "__main__":
    main()