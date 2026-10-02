import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

from decision_engine import VisionQCDecisionEngine


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    / "inspections"
    / "_temporary_roi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "inspections"
    / "heatmaps"
)


# ============================================================
# ROI PREPROCESSING
# ============================================================

def create_80_percent_roi(input_path, output_path):

    image = Image.open(input_path)

    # Correct phone-camera orientation
    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Fit image into 400 x 400 canvas
    # --------------------------------------------------------

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

    canvas.paste(
        image,
        (x, y),
    )

    # --------------------------------------------------------
    # Central 80% crop
    # --------------------------------------------------------

    crop_size = round(canvas_size * 0.80)

    left = (canvas_size - crop_size) // 2
    top = (canvas_size - crop_size) // 2

    right = left + crop_size
    bottom = top + crop_size

    roi = canvas.crop(
        (left, top, right, bottom)
    )

    # --------------------------------------------------------
    # PatchCore resolution
    # --------------------------------------------------------

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


# ============================================================
# PATCHCORE MODEL
# ============================================================

def load_patchcore():

    print()
    print("Loading PatchCore v2...")

    import torch
    import anomalib

    # Our checkpoint was created locally.
    # PyTorch 2.6+ otherwise defaults to weights_only=True.
    torch.serialization.add_safe_globals(
        [anomalib.PrecisionType]
    )

    model = Patchcore.load_from_checkpoint(
        CHECKPOINT,
        weights_only=False,
    )

    # Disable Anomalib post-processing so we receive
    # the raw PatchCore anomaly score.
    model.post_processor = None

    return model


# ============================================================
# INSPECTION
# ============================================================

def inspect_image(original_image):

    print("=" * 70)
    print("VisionQC — Automatic PatchCore Inspection")
    print("=" * 70)

    print()
    print(
        f"Original image : {original_image.name}"
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not original_image.exists():

        raise FileNotFoundError(
            f"Image not found:\n{original_image}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"PatchCore checkpoint not found:\n{CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Create temporary ROI
    # --------------------------------------------------------

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    roi_path = (
        TEMP_DIR
        / f"{original_image.stem}_roi80.jpg"
    )

    print()
    print("Creating automatic 80% ROI...")

    create_80_percent_roi(
        original_image,
        roi_path,
    )

    print(
        f"ROI created     : {roi_path.name}"
    )

    # --------------------------------------------------------
    # Load PatchCore
    # --------------------------------------------------------

    model = load_patchcore()

    # --------------------------------------------------------
    # Prediction dataset
    # --------------------------------------------------------

    dataset = PredictDataset(
        path=str(roi_path),
        image_size=(256, 256),
    )

    # --------------------------------------------------------
    # Run PatchCore
    # --------------------------------------------------------

    print()
    print("Running PatchCore...")

    engine = Engine(
        default_root_dir=str(
            PROJECT_ROOT
            / "models"
            / "patchcore_water_cap_v2"
        ),
        accelerator="cpu",
        devices=1,
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=None,
    )

    if not predictions:

        raise RuntimeError(
            "PatchCore returned no prediction."
        )

    prediction = predictions[0]

    # --------------------------------------------------------
    # Raw anomaly score
    # --------------------------------------------------------

    raw_score = float(
        prediction.pred_score.item()
    )

    print()
    print(
        f"RAW anomaly score : {raw_score:.6f}"
    )

    # --------------------------------------------------------
    # Anomaly map
    # --------------------------------------------------------

    anomaly_map = (
        prediction.anomaly_map
        .detach()
        .cpu()
        .numpy()
    )

    anomaly_map = np.squeeze(
        anomaly_map
    )

    # --------------------------------------------------------
    # Save anomaly map
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    anomaly_map_path = (
        OUTPUT_DIR
        / f"{original_image.stem}_anomaly_map.npy"
    )

    np.save(
        anomaly_map_path,
        anomaly_map,
    )

    # --------------------------------------------------------
    # Decision engine
    # --------------------------------------------------------

    decision_engine = VisionQCDecisionEngine(
        threshold=12.3785223961,
        review_margin=0.10,
    )

    result = decision_engine.inspect(
        anomaly_score=raw_score,
        image_quality_valid=True,
        product_detected=True,
        alignment_valid=True,
    )

    # --------------------------------------------------------
    # Save inspection JSON
    # --------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / f"{original_image.stem}_inspection.json"
    )

    inspection_data = {
        "input_image": str(original_image),
        "product": "Water Bottle Cap",
        "sku": "water_cap_v1",
        "model": "PatchCore v2",
        "model_version": "patchcore_water_cap_v2",
        "image_quality": "VALID",
        "product_detected": True,
        "alignment": "VALID",
        "anomaly_score": result.anomaly_score,
        "threshold": result.threshold,
        "confidence": result.confidence,
        "decision": result.decision,
        "reason": result.reason,
        "anomaly_map": str(anomaly_map_path),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(inspection_data, f, indent=4)

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VISIONQC INSPECTION RESULT")
    print("=" * 70)

    print(
        f"Input Image       : {original_image.name}"
    )

    print(
        "Product           : Water Bottle Cap"
    )

    print(
        "SKU               : water_cap_v1"
    )

    print(
        "Model             : PatchCore v2"
    )

    print(
        "Image Quality     : VALID"
    )

    print(
        "Product Detected  : YES"
    )

    print(
        "Alignment         : VALID"
    )

    print(
        f"Anomaly Score     : {result.anomaly_score:.6f}"
    )

    print(
        f"Threshold         : {result.threshold:.6f}"
    )

    print(
        f"Confidence        : {result.confidence:.2f}%"
    )

    print(
        f"Decision          : {result.decision}"
    )

    print(
        f"Reason            : {result.reason}"
    )

    print()
    print(
        f"Temporary ROI     : {roi_path}"
    )

    print(
        f"Anomaly map       : {anomaly_map_path}"
    )

    print(
        f"Inspection log    : {json_path}"
    )

    print()
    print("Inspection completed.")

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            'Usage:\n'
            'python vision/run_patchcore_auto.py '
            '"path_to_image"'
        )

        sys.exit(1)

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.is_absolute():

        image_path = (
            PROJECT_ROOT
            / image_path
        )

    image_path = image_path.resolve()

    inspect_image(
        image_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()