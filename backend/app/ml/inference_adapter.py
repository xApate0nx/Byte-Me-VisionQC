from __future__ import annotations
import cv2

import sys
from pathlib import Path

import numpy as np
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VISION_DIR = PROJECT_ROOT / "vision"

if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))

from product_detector import detect_product
from cap_roi import extract_canonical_roi
from spatial_features import extract_spatial_features


CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap_v4"
    / "Patchcore"
    / "water_cap_v1_final_validation"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

TEMP_DIR = (
    PROJECT_ROOT
    / "data"
    / "inspections"
    / "_backend_temp"
)

TEMP_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 14.0
REVIEW_MARGIN = 1.0

_model = None
_engine = None


def _load_model():
    global _model, _engine

    if _model is not None:
        return _model, _engine

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"PatchCore V4 checkpoint not found: {CHECKPOINT}"
        )

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )

    checkpoint_data = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint_data.get(
        "state_dict",
        checkpoint_data,
    )

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    model.eval()

    # Critical:
    # return raw PatchCore distance, not Anomalib 0-1 normalization.
    model.post_processor = None

    _model = model
    _engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    return _model, _engine


def _predict(roi_path: Path):
    model, engine = _load_model()

    predictions = engine.predict(
        model=model,
        data_path=str(roi_path),
    )

    if not predictions:
        raise RuntimeError(
            "PatchCore returned no prediction."
        )

    prediction = predictions[0]

    if hasattr(prediction, "pred_score"):
        score_value = prediction.pred_score
    elif isinstance(prediction, dict):
        score_value = prediction.get("pred_score")
    else:
        score_value = None

    if score_value is None:
        raise RuntimeError(
            "Could not extract PatchCore anomaly score."
        )

    if torch.is_tensor(score_value):
        score = float(
            score_value.detach()
            .cpu()
            .reshape(-1)[0]
        )
    else:
        score = float(
            np.asarray(score_value)
            .reshape(-1)[0]
        )

    if hasattr(prediction, "anomaly_map"):
        anomaly_map = prediction.anomaly_map
    elif isinstance(prediction, dict):
        anomaly_map = prediction.get("anomaly_map")
    else:
        anomaly_map = None

    if anomaly_map is None:
        raise RuntimeError(
            "Could not extract PatchCore anomaly map."
        )

    if torch.is_tensor(anomaly_map):
        anomaly_map = (
            anomaly_map.detach()
            .cpu()
            .numpy()
        )

    anomaly_map = np.asarray(anomaly_map)
    anomaly_map = np.squeeze(anomaly_map)

    if anomaly_map.ndim != 2:
        raise RuntimeError(
            f"Invalid anomaly map shape: {anomaly_map.shape}"
        )

    return score, anomaly_map


def _decide(score: float, threshold: float):
    if abs(score - threshold) <= REVIEW_MARGIN:
        return "REVIEW"

    if score >= threshold:
        return "FAIL"

    return "PASS"


def inspect_image(
    image_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    # ---------------------------------------------------------
    # 1. Product detection
    # ---------------------------------------------------------

    detection = detect_product(
        str(image_path)
    )

    if not detection.detected:
        return {
            "anomaly_score": None,
            "threshold": threshold,
            "decision": "INSPECTION_INVALID",
            "roi_path": None,
            "anomaly_map": None,
            "product_detected": False,
            "alignment_valid": False,
            "spatial_features": {},
            "reason": detection.reason,
        }

    # ---------------------------------------------------------
    # 2. Canonical ROI
    # ---------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    roi, roi_bbox = extract_canonical_roi(
        image,
        detection.bbox,
        output_size=IMAGE_SIZE,
        padding=0.30,
    )

    roi_path = (
        TEMP_DIR
        / f"{image_path.stem}_canonical_roi.png"
    )

    if not cv2.imwrite(
        str(roi_path),
        roi,
    ):
        raise RuntimeError(
            f"Could not save ROI: {roi_path}"
        )

    # ---------------------------------------------------------
    # 3. PatchCore V4
    # ---------------------------------------------------------

    score, anomaly_map = _predict(
        roi_path
    )

    # ---------------------------------------------------------
    # 4. Spatial anomaly features
    # ---------------------------------------------------------

    spatial = extract_spatial_features(
        anomaly_map
    )

    # ---------------------------------------------------------
    # 5. Decision
    # ---------------------------------------------------------

    decision = _decide(
        score,
        threshold,
    )

    return {
        "anomaly_score": score,
        "threshold": threshold,
        "decision": decision,
        "roi_path": str(roi_path),
        "anomaly_map": anomaly_map,
        "product_detected": True,
        "alignment_valid": True,
        "spatial_features": spatial,
        "reason": (
            "Within learned normal range."
            if decision == "PASS"
            else (
                "Near the operating threshold; supervisor review recommended."
                if decision == "REVIEW"
                else "Significant deviation from learned normal."
            )
        ),
    }

