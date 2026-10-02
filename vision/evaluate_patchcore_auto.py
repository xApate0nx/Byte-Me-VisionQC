from pathlib import Path
import csv
import json
import sys

import numpy as np
from PIL import Image, ImageOps

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
    / "normal"
)

DEFECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
    / "defects"
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

TEMP_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "_auto_roi"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

THRESHOLD = 12.3785223961


def create_80_percent_roi(input_path, output_path):
    image = Image.open(input_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    canvas_size = 400

    scale = min(
        canvas_size / image.width,
        canvas_size / image.height,
    )

    new_width = round(image.width * scale)
    new_height = round(image.height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (canvas_size, canvas_size),
        (255, 255, 255),
    )

    x = (canvas_size - new_width) // 2
    y = (canvas_size - new_height) // 2

    canvas.paste(image, (x, y))

    crop_size = round(canvas_size * 0.80)

    left = (canvas_size - crop_size) // 2
    top = (canvas_size - crop_size) // 2

    right = left + crop_size
    bottom = top + crop_size

    roi = canvas.crop(
        (left, top, right, bottom)
    )

    roi = roi.resize(
        (256, 256),
        Image.Resampling.LANCZOS,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    roi.save(
        output_path,
        quality=95,
    )


def load_patchcore():

    print()
    print("Loading PatchCore v2...")

    import torch
    import anomalib

    torch.serialization.add_safe_globals(
        [anomalib.PrecisionType]
    )

    model = Patchcore.load_from_checkpoint(
        CHECKPOINT,
        weights_only=False,
    )

    # We need the raw PatchCore score.
    model.post_processor = None

    return model


def calculate_metrics(results):

    tp = sum(
        1
        for r in results
        if r["actual"] == "DEFECT"
        and r["decision"] == "FAIL"
    )

    tn = sum(
        1
        for r in results
        if r["actual"] == "NORMAL"
        and r["decision"] == "PASS"
    )

    fp = sum(
        1
        for r in results
        if r["actual"] == "NORMAL"
        and r["decision"] == "FAIL"
    )

    fn = sum(
        1
        for r in results
        if r["actual"] == "DEFECT"
        and r["decision"] == "PASS"
    )

    total = tp + tn + fp + fn

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def inspect_one(model, engine, image_path, actual):

    roi_path = (
        TEMP_DIR
        / actual.lower()
        / f"{image_path.stem}_roi80.jpg"
    )

    create_80_percent_roi(
        image_path,
        roi_path,
    )

    dataset = PredictDataset(
        path=str(roi_path),
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=None,
    )

    if not predictions:
        raise RuntimeError(
            f"No prediction returned for {image_path.name}"
        )

    prediction = predictions[0]

    score = float(
        prediction.pred_score.item()
    )

    decision = (
        "FAIL"
        if score >= THRESHOLD
        else "PASS"
    )

    return {
        "filename": image_path.name,
        "actual": actual,
        "anomaly_score": score,
        "threshold": THRESHOLD,
        "decision": decision,
    }


def main():

    print("=" * 70)
    print("VisionQC — FULL AUTOMATIC PATCHCORE EVALUATION")
    print("=" * 70)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    normal_images = sorted(
        list(NORMAL_DIR.glob("*.jpg"))
        + list(NORMAL_DIR.glob("*.jpeg"))
        + list(NORMAL_DIR.glob("*.png"))
    )

    defect_images = sorted(
        list(DEFECT_DIR.glob("*.jpg"))
        + list(DEFECT_DIR.glob("*.jpeg"))
        + list(DEFECT_DIR.glob("*.png"))
    )

    print()
    print(f"Normal images : {len(normal_images)}")
    print(f"Defect images : {len(defect_images)}")
    print(f"Total images  : {len(normal_images) + len(defect_images)}")

    if not normal_images:
        raise RuntimeError("No normal images found.")

    if not defect_images:
        raise RuntimeError("No defect images found.")

    model = load_patchcore()

    engine = Engine(
        default_root_dir=str(
            PROJECT_ROOT
            / "models"
            / "patchcore_water_cap_v2"
        ),
        accelerator="cpu",
        devices=1,
    )

    results = []

    print()
    print("-" * 70)
    print("EVALUATING NORMAL IMAGES")
    print("-" * 70)

    for index, image_path in enumerate(
        normal_images,
        start=1,
    ):

        result = inspect_one(
            model,
            engine,
            image_path,
            "NORMAL",
        )

        results.append(result)

        print(
            f"[NORMAL {index:02d}/{len(normal_images)}] "
            f"{result['anomaly_score']:.6f} "
            f"-> {result['decision']} "
            f"| {image_path.name}"
        )

    print()
    print("-" * 70)
    print("EVALUATING DEFECT IMAGES")
    print("-" * 70)

    for index, image_path in enumerate(
        defect_images,
        start=1,
    ):

        result = inspect_one(
            model,
            engine,
            image_path,
            "DEFECT",
        )

        results.append(result)

        print(
            f"[DEFECT {index:02d}/{len(defect_images)}] "
            f"{result['anomaly_score']:.6f} "
            f"-> {result['decision']} "
            f"| {image_path.name}"
        )

    metrics = calculate_metrics(results)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        REPORT_DIR
        / "patchcore_auto_evaluation.csv"
    )

    json_path = (
        REPORT_DIR
        / "patchcore_auto_evaluation.json"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "actual",
                "anomaly_score",
                "threshold",
                "decision",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    report = {
        "model": "PatchCore v2",
        "model_version": "patchcore_water_cap_v2",
        "threshold": THRESHOLD,
        "normal_samples": len(normal_images),
        "defect_samples": len(defect_images),
        "metrics": metrics,
        "results": results,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    print()
    print("=" * 70)
    print("VISIONQC EVALUATION RESULTS")
    print("=" * 70)

    print()
    print(f"Threshold   : {THRESHOLD:.6f}")

    print()
    print(f"TP          : {metrics['TP']}")
    print(f"TN          : {metrics['TN']}")
    print(f"FP          : {metrics['FP']}")
    print(f"FN          : {metrics['FN']}")

    print()
    print(
        f"Precision   : {metrics['precision']:.4f}"
    )

    print(
        f"Recall      : {metrics['recall']:.4f}"
    )

    print(
        f"Specificity : {metrics['specificity']:.4f}"
    )

    print(
        f"F1 Score    : {metrics['f1']:.4f}"
    )

    print(
        f"Accuracy    : {metrics['accuracy']:.4f}"
    )

    print()
    print(f"CSV report  : {csv_path}")
    print(f"JSON report : {json_path}")

    print()
    print("Evaluation completed.")


if __name__ == "__main__":
    main()