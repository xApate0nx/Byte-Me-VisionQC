from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

import anomalib
from anomalib.models import Patchcore
from anomalib.data import PredictDataset
from anomalib.engine import Engine


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

TEMP_DIR = PROJECT_ROOT / "data" / "inspections" / "_backend_temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# VisionQC PatchCore configuration
# ---------------------------------------------------------

IMAGE_SIZE = 256

# Current prototype threshold.
# This was calibrated from the existing 44-image prototype
# evaluation and is NOT a production-calibrated threshold.
DEFAULT_THRESHOLD = 12.3785223961


# ---------------------------------------------------------
# PatchCore model holder
# ---------------------------------------------------------

_model = None
_engine = None


def _load_model():
    """
    Load PatchCore v2 once and keep it in memory.

    Loading the model for every inspection would be extremely
    slow, so the backend keeps one loaded model instance.
    """

    global _model, _engine

    if _model is not None and _engine is not None:
        return _model, _engine

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"PatchCore checkpoint not found: {CHECKPOINT}"
        )

    # Anomalib 2.6.2 checkpoint compatibility.
    torch.serialization.add_safe_globals(
        [anomalib.PrecisionType]
    )

    _model = Patchcore.load_from_checkpoint(
        CHECKPOINT,
        weights_only=False,
    )

    # We need the actual raw PatchCore distance,
    # not Anomalib's postprocessed 0/1 prediction.
    _model.post_processor = None

    _engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    return _model, _engine


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def prepare_roi(
    input_path: Path,
    output_path: Path,
) -> Path:
    """
    Convert the original inspection image into the same
    80% ROI representation used during PatchCore v2 evaluation.

    Pipeline:

        original image
             ↓
        EXIF correction
             ↓
        400x400 white canvas
             ↓
        central 80% crop
             ↓
        256x256
    """

    image = Image.open(input_path)

    # Correct phone-camera orientation.
    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    # -----------------------------------------------------
    # Fit image into a 400x400 canvas while preserving
    # aspect ratio.
    # -----------------------------------------------------

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
        "white",
    )

    x = (canvas_size - new_width) // 2
    y = (canvas_size - new_height) // 2

    canvas.paste(image, (x, y))

    # -----------------------------------------------------
    # Central 80% ROI
    # -----------------------------------------------------

    roi_size = round(canvas_size * 0.80)

    left = (canvas_size - roi_size) // 2
    top = (canvas_size - roi_size) // 2

    roi = canvas.crop(
        (
            left,
            top,
            left + roi_size,
            top + roi_size,
        )
    )

    # PatchCore input size.
    roi = roi.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    roi.save(output_path, quality=95)

    return output_path


# ---------------------------------------------------------
# PatchCore prediction
# ---------------------------------------------------------

def predict(image_path: Path):
    """
    Run PatchCore v2 on one prepared ROI image.
    """

    model, engine = _load_model()

    dataset = PredictDataset(
        path=image_path,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=None,
    )

    if predictions is None:
        raise RuntimeError(
            "PatchCore returned no prediction."
        )

    if not isinstance(predictions, list):
        predictions = list(predictions)

    if len(predictions) == 0:
        raise RuntimeError(
            "PatchCore returned an empty prediction list."
        )

    prediction = predictions[0]

    # -----------------------------------------------------
    # Extract raw anomaly score
    # -----------------------------------------------------

    if hasattr(prediction, "pred_score"):
        score_tensor = prediction.pred_score
    elif isinstance(prediction, dict) and "pred_score" in prediction:
        score_tensor = prediction["pred_score"]
    else:
        raise RuntimeError(
            "Could not find pred_score in PatchCore prediction."
        )

    if torch.is_tensor(score_tensor):
        score = float(score_tensor.detach().cpu().reshape(-1)[0])
    else:
        score = float(np.asarray(score_tensor).reshape(-1)[0])

    # -----------------------------------------------------
    # Extract anomaly map
    # -----------------------------------------------------

    anomaly_map = None

    if hasattr(prediction, "anomaly_map"):
        anomaly_map = prediction.anomaly_map
    elif isinstance(prediction, dict):
        anomaly_map = prediction.get("anomaly_map")

    if anomaly_map is not None:
        if torch.is_tensor(anomaly_map):
            anomaly_map = (
                anomaly_map.detach()
                .cpu()
                .numpy()
            )

        anomaly_map = np.asarray(anomaly_map)

        # Remove batch/channel dimensions.
        anomaly_map = np.squeeze(anomaly_map)

        if anomaly_map.ndim != 2:
            anomaly_map = None

    return {
        "anomaly_score": score,
        "anomaly_map": anomaly_map,
    }


# ---------------------------------------------------------
# Public inspection function
# ---------------------------------------------------------

def inspect_image(
    image_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
):
    """
    Complete ML inference pipeline.

    Returns:
        anomaly score
        threshold
        decision
        anomaly map
        ROI path
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    # Create a unique temporary ROI filename.
    roi_path = TEMP_DIR / (
        f"{image_path.stem}_roi80.png"
    )

    prepare_roi(
        image_path,
        roi_path,
    )

    result = predict(roi_path)

    score = result["anomaly_score"]

    # -----------------------------------------------------
    # Current prototype decision rule
    # -----------------------------------------------------

    if score >= threshold:
        decision = "FAIL"
    else:
        decision = "PASS"

    return {
        "anomaly_score": score,
        "threshold": threshold,
        "decision": decision,
        "roi_path": str(roi_path),
        "anomaly_map": result["anomaly_map"],
    }
# ---------------------------------------------------------
# Prepared canonical ROI inspection
# ---------------------------------------------------------

def inspect_prepared_roi(
    roi_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
):
    """
    Run the existing PatchCore v2 checkpoint on an already
    prepared canonical product ROI.

    IMPORTANT:
    This function intentionally does NOT call prepare_roi().

    The canonical ROI has already been localized and extracted
    by the product detector / ROI pipeline.

    Expected flow:

        original image
             ↓
        product detector
             ↓
        canonical ROI (224x224)
             ↓
        PredictDataset resize (256x256)
             ↓
        existing PatchCore v2
    """

    roi_path = Path(roi_path)

    if not roi_path.exists():
        raise FileNotFoundError(
            f"Prepared ROI not found: {roi_path}"
        )

    result = predict(roi_path)

    score = result["anomaly_score"]

    if score >= threshold:
        decision = "FAIL"
    else:
        decision = "PASS"

    return {
        "anomaly_score": score,
        "threshold": threshold,
        "decision": decision,
        "roi_path": str(roi_path),
        "anomaly_map": result["anomaly_map"],
    }