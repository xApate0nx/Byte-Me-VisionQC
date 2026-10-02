from __future__ import annotations

from pathlib import Path
import random
import json
import shutil

import numpy as np
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix


PROJECT = Path(__file__).resolve().parent.parent

NORMAL = PROJECT / "data/evaluation/roi_comparison/new_roi"
DEFECT = PROJECT / "data/evaluation/roi_comparison/defects_new_roi"

WORK = PROJECT / "data/evaluation/final_validation"
TRAIN = WORK / "train_normal"
TEST_NORMAL = WORK / "test_normal"
TEST_DEFECT = WORK / "test_defect"

MODEL_ROOT = PROJECT / "models/patchcore_water_cap_v4"

SEED = 42


# ---------------------------------------------------------------------
# DIRECTORY SETUP
# ---------------------------------------------------------------------

def reset_dirs():
    if WORK.exists():
        shutil.rmtree(WORK)

    for directory in [
        TRAIN,
        TEST_NORMAL,
        TEST_DEFECT,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ---------------------------------------------------------------------
# DATA SPLIT
# ---------------------------------------------------------------------

def split_data():

    normals = sorted(
        NORMAL.glob("*.png")
    )

    defects = sorted(
        DEFECT.glob("*.png")
    )

    if len(normals) != 21:
        raise RuntimeError(
            f"Expected 21 normal ROIs, found {len(normals)}"
        )

    if len(defects) != 23:
        raise RuntimeError(
            f"Expected 23 defect ROIs, found {len(defects)}"
        )

    random.seed(SEED)
    random.shuffle(normals)

    train_normals = normals[:15]
    test_normals = normals[15:]

    for path in train_normals:
        shutil.copy2(
            path,
            TRAIN / path.name,
        )

    for path in test_normals:
        shutil.copy2(
            path,
            TEST_NORMAL / path.name,
        )

    for path in defects:
        shutil.copy2(
            path,
            TEST_DEFECT / path.name,
        )

    print("=" * 80)
    print("FINAL MODEL VALIDATION SPLIT")
    print("=" * 80)

    print(
        f"Training normals : {len(train_normals)}"
    )

    print(
        f"Unseen normals   : {len(test_normals)}"
    )

    print(
        f"Unseen defects   : {len(defects)}"
    )


# ---------------------------------------------------------------------
# PATCHCORE V4 TRAINING
# ---------------------------------------------------------------------

def train_patchcore():

    print()
    print("=" * 80)
    print("TRAINING PATCHCORE V4")
    print("=" * 80)

    from anomalib.data import Folder
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    if MODEL_ROOT.exists():
        shutil.rmtree(MODEL_ROOT)

    datamodule = Folder(
        name="water_cap_v1_final_validation",
        root=str(WORK),
        normal_dir="train_normal",
        normal_split_ratio=0.0,
        abnormal_dir=None,
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )

    model = Patchcore(
        backbone="resnet18",
        layers=[
            "layer2",
            "layer3",
        ],
        pre_trained=True,
        num_neighbors=9,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir=str(
            MODEL_ROOT
        ),
    )

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print()
    print(
        "PATCHCORE V4 TRAINING COMPLETE"
    )

    print(MODEL_ROOT)


# ---------------------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------------------

def find_checkpoint():

    checkpoints = sorted(
        MODEL_ROOT.rglob("*.ckpt")
    )

    if not checkpoints:
        raise FileNotFoundError(
            "No PatchCore V4 checkpoint found under:\n"
            f"{MODEL_ROOT}"
        )

    checkpoint = checkpoints[0]

    print()
    print("=" * 80)
    print("PATCHCORE V4 CHECKPOINT")
    print("=" * 80)

    print(checkpoint)

    return checkpoint


# ---------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------

def run_inference(checkpoint):

    print()
    print("=" * 80)
    print("RUNNING FINAL UNSEEN-IMAGE INFERENCE")
    print("=" * 80)

    import torch
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    model = Patchcore(
        backbone="resnet18",
        layers=[
            "layer2",
            "layer3",
        ],
        pre_trained=True,
        num_neighbors=9,
    )

    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint_data.get(
        "state_dict",
        checkpoint_data,
    )

    model.load_state_dict(
        state,
        strict=False,
    )

    model.eval()

    # We want the raw PatchCore distance,
    # not Anomalib's automatic thresholding.
    model.post_processor = None

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    results = []

    for category, directory in [
        ("normal", TEST_NORMAL),
        ("defect", TEST_DEFECT),
    ]:

        images = sorted(
            directory.glob("*.png")
        )

        for image_path in images:

            print(
                f"Testing {category}: "
                f"{image_path.name}"
            )

            predictions = engine.predict(
                model=model,
                data_path=str(image_path),
            )

            if predictions is None:
                print(
                    "WARNING: no prediction returned"
                )
                continue

            if not isinstance(
                predictions,
                list,
            ):
                predictions = [
                    predictions
                ]

            prediction_found = False

            for output in predictions:

                if not hasattr(
                    output,
                    "pred_score",
                ):
                    continue

                score = float(
                    output.pred_score
                    .squeeze()
                    .item()
                )

                if not hasattr(
                    output,
                    "anomaly_map",
                ):
                    print(
                        "WARNING: no anomaly map:",
                        image_path.name,
                    )
                    continue

                anomaly_map = (
                    output.anomaly_map
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                map_path = (
                    WORK
                    / f"{category}_"
                    f"{image_path.stem}_map.npy"
                )

                np.save(
                    map_path,
                    anomaly_map,
                )

                results.append({
                    "category": category,
                    "filename": image_path.name,
                    "score": score,
                    "map": str(map_path),
                })

                prediction_found = True
                break

            if not prediction_found:
                print(
                    "WARNING: prediction could not be extracted:",
                    image_path.name,
                )

    print()
    print(
        f"Inference complete: "
        f"{len(results)} images"
    )

    return results


# ---------------------------------------------------------------------
# SPATIAL FEATURES
# ---------------------------------------------------------------------

def spatial_features(heatmap):

    h, w = heatmap.shape

    minimum = float(
        np.min(heatmap)
    )

    maximum = float(
        np.max(heatmap)
    )

    if maximum > minimum:

        norm = (
            heatmap - minimum
        ) / (
            maximum - minimum
        )

    else:

        norm = np.zeros_like(
            heatmap
        )

    threshold = np.percentile(
        heatmap,
        95,
    )

    mask = heatmap >= threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:

        return {
            "central": 0.0,
            "border": 0.0,
            "centroid_distance": 0.0,
            "largest_region": 0.0,
            "mass": 0.0,
        }

    cx = w / 2
    cy = h / 2

    central = (
        (np.abs(xs - cx) <= 0.30 * w)
        &
        (np.abs(ys - cy) <= 0.30 * h)
    )

    border = (
        (xs < 0.20 * w)
        |
        (xs >= 0.80 * w)
        |
        (ys < 0.20 * h)
        |
        (ys >= 0.80 * h)
    )

    centroid_x = (
        np.mean(xs) / w
    )

    centroid_y = (
        np.mean(ys) / h
    )

    distance = np.sqrt(
        (centroid_x - 0.5) ** 2
        +
        (centroid_y - 0.5) ** 2
    )

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            8,
        )
    )

    if n > 1:

        largest = np.max(
            stats[
                1:,
                cv2.CC_STAT_AREA,
            ]
        )

    else:

        largest = 0

    return {
        "central": float(
            np.mean(central)
        ),
        "border": float(
            np.mean(border)
        ),
        "centroid_distance": float(
            distance
        ),
        "largest_region": float(
            largest / len(xs)
        ),
        "mass": float(
            np.mean(norm)
        ),
    }


# ---------------------------------------------------------------------
# BUILD RESULTS
# ---------------------------------------------------------------------

def build_features(results):

    rows = []

    for result in results:

        heatmap = np.load(
            result["map"]
        )

        spatial = spatial_features(
            heatmap
        )

        rows.append({
            "filename": result["filename"],
            "category": result["category"],
            "label": (
                1
                if result["category"] == "defect"
                else 0
            ),
            "score": result["score"],
            **spatial,
        })

    return rows


# ---------------------------------------------------------------------
# DESCRIPTIVE VALIDATION
# ---------------------------------------------------------------------

def evaluate(rows):

    normals = [
        row
        for row in rows
        if row["label"] == 0
    ]

    defects = [
        row
        for row in rows
        if row["label"] == 1
    ]

    normal_scores = np.array([
        row["score"]
        for row in normals
    ])

    defect_scores = np.array([
        row["score"]
        for row in defects
    ])

    print()
    print("=" * 80)
    print("V4 RAW SCORE VALIDATION")
    print("=" * 80)

    print()
    print("UNSEEN NORMALS")

    print(
        f"Count  : {len(normal_scores)}"
    )

    print(
        f"Min    : {np.min(normal_scores):.6f}"
    )

    print(
        f"Median : {np.median(normal_scores):.6f}"
    )

    print(
        f"Mean   : {np.mean(normal_scores):.6f}"
    )

    print(
        f"Max    : {np.max(normal_scores):.6f}"
    )

    print()
    print("UNSEEN DEFECTS")

    print(
        f"Count  : {len(defect_scores)}"
    )

    print(
        f"Min    : {np.min(defect_scores):.6f}"
    )

    print(
        f"Median : {np.median(defect_scores):.6f}"
    )

    print(
        f"Mean   : {np.mean(defect_scores):.6f}"
    )

    print(
        f"Max    : {np.max(defect_scores):.6f}"
    )

    highest_normal = float(
        np.max(normal_scores)
    )

    lowest_defect = float(
        np.min(defect_scores)
    )

    gap = (
        lowest_defect
        - highest_normal
    )

    perfect = (
        lowest_defect
        > highest_normal
    )

    print()
    print("SEPARATION")

    print(
        f"Highest normal : "
        f"{highest_normal:.6f}"
    )

    print(
        f"Lowest defect  : "
        f"{lowest_defect:.6f}"
    )

    print(
        f"Gap            : "
        f"{gap:.6f}"
    )

    print(
        f"Perfect separation: "
        f"{perfect}"
    )

    # -----------------------------------------------------------------
    # Descriptive threshold sweep.
    #
    # This is NOT treated as a production threshold.
    # It simply shows the tradeoff on this small dataset.
    # -----------------------------------------------------------------

    candidate_thresholds = np.arange(
        np.floor(
            min(
                np.min(normal_scores),
                np.min(defect_scores),
            )
        ),
        np.ceil(
            max(
                np.max(normal_scores),
                np.max(defect_scores),
            )
        ) + 0.01,
        0.5,
    )

    threshold_results = []

    y = np.array([
        row["label"]
        for row in rows
    ])

    for threshold in candidate_thresholds:

        predictions = np.array([
            int(
                row["score"]
                >= threshold
            )
            for row in rows
        ])

        tn, fp, fn, tp = (
            confusion_matrix(
                y,
                predictions,
                labels=[0, 1],
            ).ravel()
        )

        threshold_results.append({
            "threshold": float(
                threshold
            ),
            "normal_false_rejects": int(fp),
            "defects_detected": int(tp),
            "defect_misses": int(fn),
            "false_reject_rate": float(
                fp / (tn + fp)
            ),
            "defect_detection_rate": float(
                tp / (tp + fn)
            ),
            "accuracy": float(
                accuracy_score(
                    y,
                    predictions,
                )
            ),
        })

    print()
    print("THRESHOLD TRADEOFF")

    print(
        "Threshold | Normal Fail | "
        "Defects Detected | Accuracy"
    )

    for item in threshold_results:

        print(
            f"{item['threshold']:9.2f} | "
            f"{item['normal_false_rejects']:3d} | "
            f"{item['defects_detected']:3d}/"
            f"{len(defects):3d} | "
            f"{item['accuracy']:.3f}"
        )

    return {
        "unseen_normal_count": len(
            normals
        ),
        "unseen_defect_count": len(
            defects
        ),
        "normal_min": float(
            np.min(normal_scores)
        ),
        "normal_median": float(
            np.median(normal_scores)
        ),
        "normal_mean": float(
            np.mean(normal_scores)
        ),
        "normal_max": float(
            np.max(normal_scores)
        ),
        "defect_min": float(
            np.min(defect_scores)
        ),
        "defect_median": float(
            np.median(defect_scores)
        ),
        "defect_mean": float(
            np.mean(defect_scores)
        ),
        "defect_max": float(
            np.max(defect_scores)
        ),
        "highest_normal": highest_normal,
        "lowest_defect": lowest_defect,
        "gap": gap,
        "perfect_separation": perfect,
        "threshold_sweep": threshold_results,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print()

    reset_dirs()

    split_data()

    train_patchcore()

    checkpoint = find_checkpoint()

    results = run_inference(
        checkpoint
    )

    if len(results) != 29:

        raise RuntimeError(
            f"Expected 29 inference results "
            f"(6 unseen normals + 23 defects), "
            f"but received {len(results)}."
        )

    rows = build_features(
        results
    )

    metrics = evaluate(
        rows
    )

    output = {
        "seed": SEED,
        "training_normal_count": 15,
        "unseen_normal_count": 6,
        "unseen_defect_count": 23,
        "checkpoint": str(
            checkpoint
        ),
        "metrics": metrics,
        "results": rows,
    }

    output_path = (
        PROJECT
        / "data"
        / "evaluation"
        / "final_model_validation.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL MODEL VALIDATION COMPLETE")
    print("=" * 80)

    print(
        f"Unseen normals : "
        f"{metrics['unseen_normal_count']}"
    )

    print(
        f"Unseen defects : "
        f"{metrics['unseen_defect_count']}"
    )

    print(
        f"Highest normal score : "
        f"{metrics['highest_normal']:.6f}"
    )

    print(
        f"Lowest defect score  : "
        f"{metrics['lowest_defect']:.6f}"
    )

    print(
        f"Score gap            : "
        f"{metrics['gap']:.6f}"
    )

    print(
        f"Perfect separation   : "
        f"{metrics['perfect_separation']}"
    )

    print()
    print("Saved:")
    print(output_path)


if __name__ == "__main__":
    main()