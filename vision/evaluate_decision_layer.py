from __future__ import annotations

from pathlib import Path
import json
import random

import cv2
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

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
    / "decision_layer_evaluation.json"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42
HOLDOUT_RATIO = 0.30

FEATURE_NAMES = [
    "score",
    "central_fraction",
    "border_fraction",
    "centroid_distance",
    "largest_region_fraction",
    "anomaly_mass",
]


# ============================================================
# LOAD SCORES
# ============================================================

def load_scores(filename: str):

    path = SCORE_ROOT / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}

    for item in data["results"]:

        filename = item["filename"]

        name = Path(filename).stem

        results[name] = {
            "filename": filename,
            "score": float(item["score"]),
        }

    return results


# ============================================================
# EXTRACT HEATMAP FEATURES
# ============================================================

def extract_heatmap_features(path: Path):

    heatmap = np.load(path).astype(np.float32)

    h, w = heatmap.shape

    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))

    # --------------------------------------------------------
    # Normalize heatmap
    # --------------------------------------------------------

    if maximum > minimum:

        normalized = (
            (heatmap - minimum)
            / (maximum - minimum)
        )

    else:

        normalized = np.zeros_like(
            heatmap
        )

    # --------------------------------------------------------
    # Top 5% anomaly pixels
    # --------------------------------------------------------

    threshold_95 = np.percentile(
        heatmap,
        95,
    )

    mask = heatmap >= threshold_95

    ys, xs = np.where(mask)

    if len(xs) == 0:

        return {
            "central_fraction": 0.0,
            "border_fraction": 0.0,
            "centroid_distance": 0.0,
            "largest_region_fraction": 0.0,
            "anomaly_mass": 0.0,
        }

    # --------------------------------------------------------
    # Central region
    # --------------------------------------------------------

    center_x = w / 2.0
    center_y = h / 2.0

    central = (
        (np.abs(xs - center_x) <= 0.30 * w)
        &
        (np.abs(ys - center_y) <= 0.30 * h)
    )

    central_fraction = float(
        np.mean(central)
    )

    # --------------------------------------------------------
    # Border region
    # --------------------------------------------------------

    border = (
        (xs < 0.20 * w)
        |
        (xs >= 0.80 * w)
        |
        (ys < 0.20 * h)
        |
        (ys >= 0.80 * h)
    )

    border_fraction = float(
        np.mean(border)
    )

    # --------------------------------------------------------
    # Anomaly centroid
    # --------------------------------------------------------

    centroid_x = float(
        np.mean(xs) / w
    )

    centroid_y = float(
        np.mean(ys) / h
    )

    centroid_distance = float(
        np.sqrt(
            (centroid_x - 0.5) ** 2
            +
            (centroid_y - 0.5) ** 2
        )
    )

    # --------------------------------------------------------
    # Largest connected anomaly region
    # --------------------------------------------------------

    mask_uint8 = (
        mask.astype(np.uint8)
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8,
        )
    )

    if num_labels > 1:

        component_sizes = (
            stats[
                1:,
                cv2.CC_STAT_AREA,
            ]
        )

        largest_component = int(
            np.max(component_sizes)
        )

    else:

        largest_component = 0

    largest_region_fraction = (
        largest_component
        / len(xs)
    )

    # --------------------------------------------------------
    # Overall anomaly mass
    # --------------------------------------------------------

    anomaly_mass = float(
        np.mean(normalized)
    )

    return {
        "central_fraction": central_fraction,
        "border_fraction": border_fraction,
        "centroid_distance": centroid_distance,
        "largest_region_fraction": (
            largest_region_fraction
        ),
        "anomaly_mass": anomaly_mass,
    }


# ============================================================
# LOAD ALL 44 SAMPLES
# ============================================================

def build_dataset():

    normal_scores = load_scores(
        "normal_scores.json"
    )

    defect_scores = load_scores(
        "defect_scores.json"
    )

    dataset = []

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    normal_heatmaps = (
        HEATMAP_ROOT / "normal"
    )

    for name, item in normal_scores.items():

        heatmap_path = (
            normal_heatmaps
            / f"{name}_anomaly_map.npy"
        )

        if not heatmap_path.exists():

            print(
                f"WARNING: missing heatmap: "
                f"{heatmap_path.name}"
            )

            continue

        features = extract_heatmap_features(
            heatmap_path
        )

        dataset.append({
            "name": name,
            "filename": item["filename"],
            "label": 0,
            "label_name": "normal",
            "score": item["score"],
            **features,
        })

    # --------------------------------------------------------
    # DEFECT
    # --------------------------------------------------------

    defect_heatmaps = (
        HEATMAP_ROOT / "defects"
    )

    for name, item in defect_scores.items():

        heatmap_path = (
            defect_heatmaps
            / f"{name}_anomaly_map.npy"
        )

        if not heatmap_path.exists():

            print(
                f"WARNING: missing heatmap: "
                f"{heatmap_path.name}"
            )

            continue

        features = extract_heatmap_features(
            heatmap_path
        )

        dataset.append({
            "name": name,
            "filename": item["filename"],
            "label": 1,
            "label_name": "defect",
            "score": item["score"],
            **features,
        })

    return dataset


# ============================================================
# STRATIFIED SPLIT
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

    normal_holdout = max(
        1,
        round(
            len(normal)
            * HOLDOUT_RATIO
        ),
    )

    defect_holdout = max(
        1,
        round(
            len(defects)
            * HOLDOUT_RATIO
        ),
    )

    normal_test = normal[
        :normal_holdout
    ]

    defect_test = defects[
        :defect_holdout
    ]

    normal_train = normal[
        normal_holdout:
    ]

    defect_train = defects[
        defect_holdout:
    ]

    train = (
        normal_train
        + defect_train
    )

    test = (
        normal_test
        + defect_test
    )

    random.shuffle(train)
    random.shuffle(test)

    return train, test


# ============================================================
# MATRIX CREATION
# ============================================================

def make_matrix(rows):

    return np.array(
        [
            [
                row[name]
                for name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=np.float32,
    )


def make_labels(rows):

    return np.array(
        [
            row["label"]
            for row in rows
        ],
        dtype=np.int64,
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=SEED,
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ])


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),

        "normal_false_rejects": int(fp),
        "defect_misses": int(fn),

        "normal_total": int(
            tn + fp
        ),

        "defect_total": int(
            fn + tp
        ),

        "false_reject_rate": float(
            fp / (tn + fp)
            if (tn + fp) > 0
            else 0
        ),

        "defect_detection_rate": float(
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        ),

        "defect_miss_rate": float(
            fn / (tp + fn)
            if (tp + fn) > 0
            else 0
        ),

        "probability_min": float(
            np.min(probabilities)
        ),

        "probability_max": float(
            np.max(probabilities)
        ),
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_coefficients(model):

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    coefficients = (
        classifier.coef_[0]
    )

    result = {}

    for name, coefficient in zip(
        FEATURE_NAMES,
        coefficients,
    ):

        result[name] = float(
            coefficient
        )

    return result


# ============================================================
# PREDICTION DETAILS
# ============================================================

def prediction_details(
    model,
    rows,
):

    X = make_matrix(rows)

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    output = []

    for row, prediction, probability in zip(
        rows,
        predictions,
        probabilities,
    ):

        output.append({
            "filename": row["filename"],
            "actual": row["label_name"],
            "predicted": (
                "defect"
                if prediction == 1
                else "normal"
            ),
            "defect_probability": float(
                probability
            ),
            "score": row["score"],
            "central_fraction": (
                row["central_fraction"]
            ),
            "border_fraction": (
                row["border_fraction"]
            ),
            "centroid_distance": (
                row["centroid_distance"]
            ),
            "largest_region_fraction": (
                row[
                    "largest_region_fraction"
                ]
            ),
            "anomaly_mass": (
                row["anomaly_mass"]
            ),
        })

    return output


# ============================================================
# PATCHCORE BASELINE
# ============================================================

def evaluate_patchcore_baseline(
    train,
    test,
):

    train_scores = sorted(
        row["score"]
        for row in train
    )

    candidates = []

    for a, b in zip(
        train_scores[:-1],
        train_scores[1:],
    ):

        threshold = (
            a + b
        ) / 2.0

        train_predictions = np.array([
            int(
                row["score"]
                >= threshold
            )
            for row in train
        ])

        y_train = make_labels(
            train
        )

        fp = int(
            np.sum(
                (
                    y_train == 0
                )
                &
                (
                    train_predictions == 1
                )
            )
        )

        tp = int(
            np.sum(
                (
                    y_train == 1
                )
                &
                (
                    train_predictions == 1
                )
            )
        )

        normal_count = int(
            np.sum(
                y_train == 0
            )
        )

        defect_count = int(
            np.sum(
                y_train == 1
            )
        )

        false_reject = (
            fp / normal_count
            if normal_count
            else 0
        )

        detection = (
            tp / defect_count
            if defect_count
            else 0
        )

        candidates.append({
            "threshold": float(
                threshold
            ),
            "false_reject_rate": (
                false_reject
            ),
            "defect_detection_rate": (
                detection
            ),
        })

    # Prefer zero false rejection,
    # then highest training detection.
    candidates.sort(
        key=lambda x: (
            x["false_reject_rate"],
            -x["defect_detection_rate"],
        )
    )

    selected = candidates[0]

    threshold = selected[
        "threshold"
    ]

    y_test = make_labels(test)

    predictions = np.array([
        int(
            row["score"]
            >= threshold
        )
        for row in test
    ])

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(
            threshold
        ),
        "test_accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "test_false_reject_rate": float(
            fp / (tn + fp)
            if (tn + fp)
            else 0
        ),
        "test_defect_detection_rate": float(
            tp / (tp + fn)
            if (tp + fn)
            else 0
        ),
        "test_defect_miss_rate": float(
            fn / (tp + fn)
            if (tp + fn)
            else 0
        ),
        "test_normal_rejects": int(fp),
        "test_normal_total": int(
            tn + fp
        ),
        "test_defects_detected": int(tp),
        "test_defect_total": int(
            tp + fn
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("VISIONQC DECISION LAYER EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    dataset = build_dataset()

    print()
    print(
        "Usable samples:",
        len(dataset),
    )

    normal_count = sum(
        row["label"] == 0
        for row in dataset
    )

    defect_count = sum(
        row["label"] == 1
        for row in dataset
    )

    print(
        "Normal:",
        normal_count,
    )

    print(
        "Defect:",
        defect_count,
    )

    if len(dataset) < 20:

        raise RuntimeError(
            "Too few usable samples."
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, test = split_dataset(
        dataset
    )

    print()
    print(
        "Decision-layer training:",
        len(train),
    )

    print(
        "Decision-layer holdout:",
        len(test),
    )

    print()
    print(
        "Training normal:",
        sum(
            x["label"] == 0
            for x in train
        ),
    )

    print(
        "Training defects:",
        sum(
            x["label"] == 1
            for x in train
        ),
    )

    print()
    print(
        "Holdout normal:",
        sum(
            x["label"] == 0
            for x in test
        ),
    )

    print(
        "Holdout defects:",
        sum(
            x["label"] == 1
            for x in test
        ),
    )

    # --------------------------------------------------------
    # Train decision layer
    # --------------------------------------------------------

    X_train = make_matrix(
        train
    )

    y_train = make_labels(
        train
    )

    X_test = make_matrix(
        test
    )

    y_test = make_labels(
        test
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        X_train,
        y_train,
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    coefficients = get_coefficients(
        model
    )

    # --------------------------------------------------------
    # PatchCore baseline
    # --------------------------------------------------------

    baseline = evaluate_patchcore_baseline(
        train,
        test,
    )

    # --------------------------------------------------------
    # Print model results
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("DECISION LAYER — TRAINING")
    print("=" * 80)

    print(
        f"Accuracy : "
        f"{train_metrics['accuracy']:.3f}"
    )

    print(
        f"Precision: "
        f"{train_metrics['precision']:.3f}"
    )

    print(
        f"Recall   : "
        f"{train_metrics['recall']:.3f}"
    )

    print(
        f"F1       : "
        f"{train_metrics['f1']:.3f}"
    )

    print()
    print("=" * 80)
    print("DECISION LAYER — HOLDOUT")
    print("=" * 80)

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.3f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.3f}"
    )

    print(
        f"Recall   : "
        f"{test_metrics['recall']:.3f}"
    )

    print(
        f"F1       : "
        f"{test_metrics['f1']:.3f}"
    )

    print(
        f"False rejects: "
        f"{test_metrics['normal_false_rejects']}/"
        f"{test_metrics['normal_total']}"
    )

    print(
        f"Defects detected: "
        f"{test_metrics['true_positives']}/"
        f"{test_metrics['defect_total']}"
    )

    print(
        f"Defect miss rate: "
        f"{test_metrics['defect_miss_rate']:.3f}"
    )

    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("DECISION-LAYER FEATURE COEFFICIENTS")
    print("=" * 80)

    sorted_coefficients = sorted(
        coefficients.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    for name, value in sorted_coefficients:

        print(
            f"{name:28} "
            f"{value:+.5f}"
        )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PATCHCORE-ONLY BASELINE")
    print("=" * 80)

    print(
        f"Threshold: "
        f"{baseline['threshold']:.4f}"
    )

    print(
        f"Holdout accuracy: "
        f"{baseline['test_accuracy']:.3f}"
    )

    print(
        f"Holdout false rejects: "
        f"{baseline['test_normal_rejects']}/"
        f"{baseline['test_normal_total']}"
    )

    print(
        f"Holdout defects detected: "
        f"{baseline['test_defects_detected']}/"
        f"{baseline['test_defect_total']}"
    )

    print(
        f"Holdout defect detection rate: "
        f"{baseline['test_defect_detection_rate']:.3f}"
    )

    # --------------------------------------------------------
    # Prediction details
    # --------------------------------------------------------

    details = prediction_details(
        model,
        test,
    )

    print()
    print("=" * 80)
    print("HOLDOUT PREDICTIONS")
    print("=" * 80)

    for item in sorted(
        details,
        key=lambda x: x["actual"],
    ):

        print(
            f"{item['actual']:7} | "
            f"predicted={item['predicted']:7} | "
            f"prob={item['defect_probability']:.3f} | "
            f"score={item['score']:.3f} | "
            f"central={item['central_fraction']:.3f} | "
            f"{item['filename']}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "seed": SEED,
        "holdout_ratio": HOLDOUT_RATIO,

        "dataset": {
            "total": len(dataset),
            "normal": normal_count,
            "defect": defect_count,
        },

        "split": {
            "training": len(train),
            "holdout": len(test),
            "training_normal": sum(
                x["label"] == 0
                for x in train
            ),
            "training_defect": sum(
                x["label"] == 1
                for x in train
            ),
            "holdout_normal": sum(
                x["label"] == 0
                for x in test
            ),
            "holdout_defect": sum(
                x["label"] == 1
                for x in test
            ),
        },

        "features": FEATURE_NAMES,

        "decision_layer": {
            "model": (
                "StandardScaler + "
                "LogisticRegression"
            ),
            "training_metrics": (
                train_metrics
            ),
            "holdout_metrics": (
                test_metrics
            ),
            "coefficients": coefficients,
        },

        "patchcore_baseline": baseline,

        "holdout_predictions": details,
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
    print("RESULT SAVED")
    print("=" * 80)

    print(OUTPUT)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This experiment does NOT modify "
        "VisionQC production inference."
    )

    print(
        "The learned decision layer is "
        "experimental until independently validated."
    )

    print()


if __name__ == "__main__":
    main()