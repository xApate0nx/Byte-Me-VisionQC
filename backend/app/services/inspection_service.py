import cv2
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from backend.app.database.models import Inspection
from backend.app.ml.inference_adapter import inspect_image
from backend.app.services.image_quality_service import check_image_quality
from backend.app.services.storage_service import (
    get_inspection_directory,
    save_original_image,
    save_roi_image,
)
from backend.app.services.threshold_service import get_current_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEMP_UPLOAD_DIR = PROJECT_ROOT / "data" / "inspections" / "_api_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STORAGE_ROOT = PROJECT_ROOT / "data" / "inspections"


def storage_url(path: Path | None) -> str | None:
    """Convert a stored inspection path into a frontend /storage URL."""
    if path is None:
        return None

    try:
        relative = Path(path).resolve().relative_to(STORAGE_ROOT.resolve())
    except ValueError:
        return None

    return "/storage/" + relative.as_posix()


def generate_heatmap(
    image_path: Path,
    anomaly_map: np.ndarray,
    output_path: Path,
):
    """
    Generate a true color anomaly heatmap over the canonical ROI.

    Low anomaly regions remain mostly transparent.
    High anomaly regions become yellow/red.
    """

    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

    if anomaly_map.ndim > 2:
        anomaly_map = np.squeeze(anomaly_map)

    if anomaly_map.ndim != 2:
        raise ValueError(
            f"Expected 2D anomaly map, got shape {anomaly_map.shape}"
        )

    roi = cv2.imread(str(image_path))

    if roi is None:
        raise ValueError(
            f"Could not read ROI image: {image_path}"
        )

    height, width = roi.shape[:2]

    # Resize anomaly map to the exact ROI dimensions.
    anomaly_map = cv2.resize(
        anomaly_map,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    # Robust normalization.
    minimum = float(np.percentile(anomaly_map, 1))
    maximum = float(np.percentile(anomaly_map, 99))

    if maximum > minimum:
        normalized = (
            (anomaly_map - minimum)
            / (maximum - minimum)
        )
    else:
        normalized = np.zeros_like(anomaly_map)

    normalized = np.clip(normalized, 0.0, 1.0)

    # Suppress very low anomaly values so the entire image
    # does not become colored.
    activation = np.clip(
        (normalized - 0.35) / 0.65,
        0.0,
        1.0,
    )

    # Smooth the anomaly field for a cleaner visualization.
    activation = cv2.GaussianBlur(
        activation,
        (0, 0),
        sigmaX=5,
    )

    # Convert to 8-bit for OpenCV colormap.
    heatmap_input = (
        activation * 255.0
    ).astype(np.uint8)

    # Blue -> cyan -> yellow -> red anomaly visualization.
    heatmap = cv2.applyColorMap(
        heatmap_input,
        cv2.COLORMAP_JET,
    )

    # Only overlay meaningful anomaly regions.
    alpha = (
        activation[..., None] * 0.75
    ).astype(np.float32)

    roi_float = roi.astype(np.float32)
    heatmap_float = heatmap.astype(np.float32)

    overlay = (
        roi_float * (1.0 - alpha)
        + heatmap_float * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(output_path),
        overlay,
    )

def calculate_provisional_confidence(
    score: float,
    threshold: float,
) -> float:
    """
    Distance from threshold expressed as a percentage.
    This is NOT a calibrated defect probability.
    """

    if threshold <= 0:
        return 0.0

    distance = abs(score - threshold)

    confidence = distance / threshold * 100.0

    return min(99.0, max(0.0, confidence))


def get_quality_reason(quality_result: dict) -> str:
    failed_checks = []

    checks = quality_result.get("checks", {})

    for check_name, check_data in checks.items():
        if isinstance(check_data, dict):
            if check_data.get("valid") is False:
                failed_checks.append(
                    check_data.get("reason", check_name)
                )

    if failed_checks:
        return (
            "Image quality / position issue: "
            + "; ".join(failed_checks)
        )

    return quality_result.get(
        "reason",
        "Image quality check failed.",
    )


def create_invalid_inspection(
    inspection_id: str,
    filename: str,
    original_path: Path,
    quality_result: dict,
    db: Session,
):
    inspection_dir = get_inspection_directory(inspection_id)

    reason = get_quality_reason(quality_result)

    inspection = Inspection(
        inspection_id=inspection_id,
        timestamp=datetime.utcnow(),
        filename=filename,
        product="Water Bottle Cap",
        sku="water_cap_v1",
        model_name="PatchCore",
        model_version="patchcore_water_cap_v4",
        anomaly_score=0.0,
        threshold=0.0,
        confidence=0.0,
        decision="INSPECTION_INVALID",
        reason=reason,
        image_quality="INVALID",
        alignment="NOT_EVALUATED",
        original_image_path=str(original_path),
        roi_image_path=None,
        heatmap_path=None,
    )

    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    inspection_json = {
        "inspection_id": inspection.inspection_id,
        "timestamp": inspection.timestamp.isoformat(),
        "filename": inspection.filename,
        "product": inspection.product,
        "sku": inspection.sku,
        "model_name": inspection.model_name,
        "model_version": inspection.model_version,
        "anomaly_score": None,
        "threshold": None,
        "confidence": 0.0,
        "confidence_type": "not_applicable",
        "decision": "INSPECTION_INVALID",
        "reason": reason,
        "image_quality": "INVALID",
        "alignment": "NOT_EVALUATED",
        "original_image_path": inspection.original_image_path,
        "roi_image_path": None,
        "heatmap_path": None,
        "image_url": storage_url(original_path),
        "heatmap_url": None,
        "product_detected": False,
        "alignment_valid": False,
        "quality_checks": quality_result.get("checks", {}),
    }

    (
        inspection_dir / "inspection.json"
    ).write_text(
        json.dumps(inspection_json, indent=2),
        encoding="utf-8",
    )

    return inspection_json


def run_inspection(
    image_bytes: bytes,
    filename: str,
    db: Session,
):
    inspection_id = str(uuid4())
    safe_filename = Path(filename).name

    input_path = (
        TEMP_UPLOAD_DIR
        / f"{inspection_id}_{safe_filename}"
    )

    input_path.write_bytes(image_bytes)

    try:
        # 1. Image quality
        quality_result = check_image_quality(input_path)

        # 2. Permanent inspection directory
        inspection_dir = get_inspection_directory(inspection_id)

        # 3. Save original
        original_path = save_original_image(
            inspection_id=inspection_id,
            source_path=input_path,
            filename=safe_filename,
        )

        # 4. Quality gate
        if not quality_result.get("valid", False):
            return create_invalid_inspection(
                inspection_id=inspection_id,
                filename=safe_filename,
                original_path=original_path,
                quality_result=quality_result,
                db=db,
            )

        # 5. Use the supervisor-configured threshold
        threshold = float(get_current_threshold())

        # 6. V4 PatchCore + product detection + canonical ROI
        result = inspect_image(
            input_path,
            threshold=threshold,
        )

        # Product detection failure
        if result.get("decision") == "INSPECTION_INVALID":
            reason = result.get(
                "reason",
                "Product could not be reliably detected.",
            )

            inspection = Inspection(
                inspection_id=inspection_id,
                timestamp=datetime.utcnow(),
                filename=safe_filename,
                product="Water Bottle Cap",
                sku="water_cap_v1",
                model_name="PatchCore",
                model_version="patchcore_water_cap_v4",
                anomaly_score=0.0,
                threshold=threshold,
                confidence=0.0,
                decision="INSPECTION_INVALID",
                reason=reason,
                image_quality="VALID",
                alignment="NOT_EVALUATED",
                original_image_path=str(original_path),
                roi_image_path=None,
                heatmap_path=None,
            )

            db.add(inspection)
            db.commit()
            db.refresh(inspection)

            inspection_json = {
                "inspection_id": inspection.inspection_id,
                "timestamp": inspection.timestamp.isoformat(),
                "filename": inspection.filename,
                "product": inspection.product,
                "sku": inspection.sku,
                "model_name": inspection.model_name,
                "model_version": inspection.model_version,
                "anomaly_score": None,
                "threshold": threshold,
                "confidence": 0.0,
                "confidence_type": "not_applicable",
                "decision": "INSPECTION_INVALID",
                "reason": reason,
                "image_quality": "VALID",
                "alignment": "NOT_EVALUATED",
                "original_image_path": str(original_path),
                "roi_image_path": None,
                "heatmap_path": None,
                "image_url": storage_url(original_path),
                "heatmap_url": None,
                "product_detected": False,
                "alignment_valid": False,
            }

            (
                inspection_dir / "inspection.json"
            ).write_text(
                json.dumps(inspection_json, indent=2),
                encoding="utf-8",
            )

            return inspection_json

        # 7. Read inference result
        anomaly_map = result.get("anomaly_map")
        score = float(result["anomaly_score"])
        decision = result["decision"]

        # 8. Provisional confidence
        confidence = calculate_provisional_confidence(
            score=score,
            threshold=threshold,
        )

        # 9. Save canonical ROI permanently
        roi_path = save_roi_image(
            inspection_id=inspection_id,
            source_path=Path(result["roi_path"]),
        )

        # 10. Generate heatmap
        heatmap_path = inspection_dir / "heatmap.png"

        generate_heatmap(
            image_path=roi_path,
            anomaly_map=anomaly_map,
            output_path=heatmap_path,
        )

        # 11. Human-readable reason
        if decision == "FAIL":
            reason = "Significant deviation from learned normal."
        elif decision == "REVIEW":
            reason = (
                "Anomaly score is near the operating threshold; "
                "supervisor review recommended."
            )
        else:
            reason = "Within learned normal range."

        # 12. Database record
        inspection = Inspection(
            inspection_id=inspection_id,
            timestamp=datetime.utcnow(),
            filename=safe_filename,
            product="Water Bottle Cap",
            sku="water_cap_v1",
            model_name="PatchCore",
            model_version="patchcore_water_cap_v4",
            anomaly_score=score,
            threshold=threshold,
            confidence=confidence,
            decision=decision,
            reason=reason,
            image_quality="VALID",
            alignment="VALID",
            original_image_path=str(original_path),
            roi_image_path=str(roi_path),
            heatmap_path=str(heatmap_path),
        )

        db.add(inspection)
        db.commit()
        db.refresh(inspection)

        # 13. API/UI URLs
        image_url = storage_url(original_path)
        heatmap_url = storage_url(heatmap_path)
        roi_url = storage_url(roi_path)

        # 14. Complete JSON record
        inspection_json = {
            "inspection_id": inspection.inspection_id,
            "timestamp": inspection.timestamp.isoformat(),
            "filename": inspection.filename,
            "product": inspection.product,
            "sku": inspection.sku,
            "model_name": inspection.model_name,
            "model_version": inspection.model_version,
            "anomaly_score": inspection.anomaly_score,
            "threshold": inspection.threshold,
            "confidence": inspection.confidence,
            "confidence_type": "provisional",
            "decision": inspection.decision,
            "reason": inspection.reason,
            "image_quality": inspection.image_quality,
            "alignment": inspection.alignment,
            "original_image_path": inspection.original_image_path,
            "roi_image_path": inspection.roi_image_path,
            "heatmap_path": inspection.heatmap_path,
            "image_url": image_url,
            "heatmap_url": heatmap_url,
            "roi_url": roi_url,
            "product_detected": True,
            "alignment_valid": True,
            "quality_checks": quality_result.get("checks", {}),
            "anomaly_map_shape": (
                list(anomaly_map.shape)
                if anomaly_map is not None
                else None
            ),
        }

        (
            inspection_dir / "inspection.json"
        ).write_text(
            json.dumps(inspection_json, indent=2),
            encoding="utf-8",
        )

        return inspection_json

    except Exception:
        db.rollback()
        raise

    finally:
        if input_path.exists():
            input_path.unlink()

