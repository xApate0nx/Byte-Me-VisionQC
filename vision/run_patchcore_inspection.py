from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

from product_detector import detect_product
from cap_roi import extract_canonical_roi
from spatial_features import extract_spatial_features


# ============================================================
# VISIONQC - PRODUCTION PATCHCORE INSPECTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

OUTPUT_DIR = PROJECT_ROOT / "data" / "inspections"

ROI_SIZE = 224

# Configurable operating threshold.
# This is NOT claimed to be a scientifically final threshold.
DEFAULT_THRESHOLD = 14.0

REVIEW_MARGIN = 1.0


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def save_heatmap(
    anomaly_map: np.ndarray,
    output_path: Path,
) -> None:
    anomaly_map = np.asarray(
        anomaly_map,
        dtype=np.float32,
    )

    anomaly_map = np.squeeze(anomaly_map)
    anomaly_map = np.nan_to_num(anomaly_map)

    if anomaly_map.ndim != 2:
        raise ValueError(
            f"Expected 2D anomaly map, got shape {anomaly_map.shape}"
        )

    minimum = float(anomaly_map.min())
    maximum = float(anomaly_map.max())

    if maximum > minimum:
        normalized = (
            anomaly_map - minimum
        ) / (
            maximum - minimum
        )
    else:
        normalized = np.zeros_like(anomaly_map)

    normalized = (
        normalized * 255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )

    cv2.imwrite(
        str(output_path),
        heatmap,
    )


# ============================================================
# DECISION
# ============================================================

def decide(
    score: float,
    threshold: float,
    spatial: dict,
) -> str:
    """
    Current operating decision layer.

    PASS:
        score clearly below threshold.

    REVIEW:
        score is close to threshold or spatial evidence
        is unusually concentrated in the product center.

    FAIL:
        score reaches/exceeds threshold.

    This is an operating rule for the prototype, not a
    calibrated defect probability.
    """

    central = float(
        spatial.get(
            "central_anomaly_fraction",
            spatial.get("central", 0.0),
        )
    )

    if score >= threshold:
        return "FAIL"

    if score >= threshold - REVIEW_MARGIN:
        return "REVIEW"

    if central >= 0.80:
        return "REVIEW"

    return "PASS"


# ============================================================
# PATCHCORE
# ============================================================

def load_model() -> Patchcore:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"V4 checkpoint not found:\n{CHECKPOINT}"
        )

    model = Patchcore(
        backbone="resnet18",
        layers=[
            "layer2",
            "layer3",
        ],
        pre_trained=False,
        num_neighbors=9,
    )

    return model


def find_prediction_value(
    prediction,
    names: list[str],
):
    for name in names:
        if hasattr(prediction, name):
            value = getattr(
                prediction,
                name,
            )

            if value is not None:
                return value

    return None


def run_patchcore(roi_path: Path, model: Patchcore):
    import torch

    # V4 uses explicit checkpoint loading into the PatchCore model.
    # This preserves the trained PatchCore memory bank.
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

    # Match V4 validation: return the raw PatchCore distance
    # instead of Anomalib's automatic post-processing.
    model.post_processor = None

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    predictions = engine.predict(
        model=model,
        data_path=str(roi_path),
    )

    if not predictions:
        raise RuntimeError("PatchCore returned no prediction.")

    prediction = predictions[0]

    score = find_prediction_value(
        prediction,
        ["anomaly_score", "pred_score"],
    )

    anomaly_map = find_prediction_value(
        prediction,
        ["anomaly_map", "anomaly_maps"],
    )

    if score is None:
        raise RuntimeError("Could not extract PatchCore anomaly score.")

    if anomaly_map is None:
        raise RuntimeError("Could not extract PatchCore anomaly map.")

    score = float(np.asarray(score).squeeze())
    anomaly_map = np.asarray(anomaly_map)

    return score, anomaly_map

def inspect_image(
    image_path: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().isoformat()

    print("=" * 70)
    print("VISIONQC PRODUCTION INSPECTION")
    print("=" * 70)
    print(
        f"Image      : {image_path.name}"
    )
    print(
        "Checkpoint : PatchCore V4"
    )
    print(
        f"Threshold  : {threshold}"
    )
    print()

    # ========================================================
    # 1. PRODUCT DETECTION
    # ========================================================

    print("[1/6] Detecting product...")

    detection = detect_product(
        str(image_path)
    )

    if not detection.detected:

        result = {
            "timestamp": timestamp,
            "filename": image_path.name,
            "decision": "INSPECTION_INVALID",
            "reason": detection.reason,
            "product_detected": False,
            "detection_confidence": float(
                detection.confidence
            ),
        }

        output = (
            OUTPUT_DIR
            / f"{image_path.stem}_inspection.json"
        )

        output.write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "Product detection FAILED."
        )

        return result

    print(
        f"    detected=True "
        f"confidence={detection.confidence:.3f}"
    )

    # ========================================================
    # 2. CANONICAL ROI
    # ========================================================

    print(
        "[2/6] Extracting canonical ROI..."
    )

    image = load_image(
        image_path
    )

    bbox = detection.bbox

    roi, roi_bbox = extract_canonical_roi(
        image,
        bbox,
        output_size=ROI_SIZE,
        padding=0.30,
    )

    roi_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_roi.png"
    )

    success = cv2.imwrite(
        str(roi_path),
        roi,
    )

    if not success:
        raise RuntimeError(
            f"Could not save ROI: {roi_path}"
        )

    print(
        f"    ROI saved: {roi_path.name}"
    )

    # ========================================================
    # 3. LOAD V4
    # ========================================================

    print(
        "[3/6] Loading PatchCore V4..."
    )

    model = load_model()

    # ========================================================
    # 4. PATCHCORE INFERENCE
    # ========================================================

    print(
        "[4/6] Running PatchCore..."
    )

    score, anomaly_map = run_patchcore(
        roi_path,
        model,
    )

    print(
        f"    anomaly score = {score:.6f}"
    )

    # ========================================================
    # 5. SPATIAL FEATURES
    # ========================================================

    print(
        "[5/6] Extracting spatial anomaly features..."
    )

    spatial = extract_spatial_features(
        anomaly_map
    )

    # ========================================================
    # 6. DECISION
    # ========================================================

    print(
        "[6/6] Making decision..."
    )

    decision = decide(
        score,
        threshold,
        spatial,
    )

    # ========================================================
    # HEATMAP
    # ========================================================

    heatmap_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_heatmap.jpg"
    )

    save_heatmap(
        anomaly_map,
        heatmap_path,
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "timestamp": timestamp,

        "filename": image_path.name,

        "model": {
            "name": "PatchCore",
            "version": "V4",
            "backbone": "resnet18",
            "layers": [
                "layer2",
                "layer3",
            ],
            "num_neighbors": 9,
            "checkpoint": str(
                CHECKPOINT
            ),
        },

        "product_detection": {
            "detected": bool(
                detection.detected
            ),
            "confidence": float(
                detection.confidence
            ),
            "bbox": list(
                detection.bbox
            ),
            "occupancy": float(
                detection.occupancy
            ),
            "circularity": float(
                detection.circularity
            ),
            "aspect_ratio": float(
                detection.aspect_ratio
            ),
            "color_ratio": float(
                detection.color_ratio
            ),
           "alignment": detection.alignment,
            "reason": detection.reason,
        },

        "roi": {
            "size": ROI_SIZE,
            "padding": 0.30,
            "bbox": list(
                roi_bbox
            ),
            "path": str(
                roi_path
            ),
        },

        "patchcore": {
            "anomaly_score": score,
            "heatmap_path": str(
                heatmap_path
            ),
        },

        "spatial_features": spatial,

        "decision": {
            "result": decision,
            "threshold": threshold,
            "review_margin": REVIEW_MARGIN,
        },
    }

    output = (
        OUTPUT_DIR
        / f"{image_path.stem}_inspection.json"
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # CONSOLE RESULT
    # ========================================================

    print()
    print("=" * 70)
    print(
        f"DECISION: {decision}"
    )
    print(
        f"SCORE   : {score:.6f}"
    )
    print(
        f"HEATMAP : {heatmap_path}"
    )
    print(
        f"RESULT  : {output}"
    )
    print("=" * 70)

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage:"
        )
        print(
            "  python vision/run_patchcore_inspection.py "
            "<image_path> [threshold]"
        )
        sys.exit(1)

    image = sys.argv[1]

    threshold = DEFAULT_THRESHOLD

    if len(sys.argv) >= 3:
        threshold = float(
            sys.argv[2]
        )

    inspect_image(
        image,
        threshold=threshold,
    )

