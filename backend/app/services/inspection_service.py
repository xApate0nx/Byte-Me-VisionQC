from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json


import numpy as np
from PIL import Image

from sqlalchemy.orm import Session

from backend.app.database.models import Inspection
from backend.app.ml.inference_adapter import inspect_image
from backend.app.services.image_quality_service import (
    check_image_quality,
)
from backend.app.services.storage_service import (
    get_inspection_directory,
    save_original_image,
    save_roi_image,
)
from backend.app.services.threshold_service import (
    get_current_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEMP_UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "inspections"
    / "_api_uploads"
)

TEMP_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def generate_heatmap(
    image_path: Path,
    anomaly_map: np.ndarray,
    output_path: Path,
):
    """
    Generate a visual anomaly heatmap and overlay it
    on the inspected ROI image.
    """

    anomaly_map = np.asarray(anomaly_map)

    if anomaly_map.ndim > 2:
        anomaly_map = np.squeeze(anomaly_map)

    if anomaly_map.ndim != 2:
        raise ValueError(
            f"Expected 2D anomaly map, got shape {anomaly_map.shape}"
        )

    minimum = float(anomaly_map.min())
    maximum = float(anomaly_map.max())

    if maximum > minimum:
        normalized = (
            (anomaly_map - minimum)
            / (maximum - minimum)
            * 255.0
        )
    else:
        normalized = np.zeros_like(
            anomaly_map,
            dtype=np.float32,
        )

    heatmap_gray = Image.fromarray(
        normalized.astype(np.uint8)
    )

    roi_image = Image.open(image_path).convert("RGB")

    heatmap_gray = heatmap_gray.resize(
        roi_image.size,
        Image.Resampling.BILINEAR,
    )

    heatmap_rgb = heatmap_gray.convert("RGB")

    overlay = Image.blend(
        roi_image,
        heatmap_rgb,
        alpha=0.45,
    )

    overlay.save(
        output_path,
        format="PNG",
    )


def calculate_provisional_confidence(
    score: float,
    threshold: float,
) -> float:
    """
    Calculate a provisional confidence-like score.

    IMPORTANT:
    This is NOT a calibrated probability.
    It represents distance from the decision threshold.
    """

    if threshold <= 0:
        return 0.0

    distance = abs(
        score - threshold
    )

    confidence = (
        distance
        / threshold
        * 100.0
    )

    return min(
        99.0,
        max(
            0.0,
            confidence,
        ),
    )


def get_quality_reason(
    quality_result: dict,
) -> str:
    """
    Convert image-quality failures into a readable
    inspection reason.
    """

    failed_checks = []

    checks = quality_result.get(
        "checks",
        {},
    )

    for check_name, check_data in checks.items():
        if isinstance(check_data, dict):
            if check_data.get("valid") is False:
                reason = check_data.get(
                    "reason",
                    check_name,
                )

                failed_checks.append(
                    reason
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
    """
    Create a database + JSON record for an image that
    cannot be reliably inspected.

    PatchCore is intentionally NOT executed.
    """

    inspection_dir = (
        get_inspection_directory(
            inspection_id
        )
    )

    reason = get_quality_reason(
        quality_result
    )

    inspection = Inspection(
        inspection_id=inspection_id,
        timestamp=datetime.utcnow(),
        filename=filename,
        product="Water Bottle Cap",
        sku="water_cap_v1",
        model_name="PatchCore",
        model_version="patchcore_water_cap_v2",
        anomaly_score=0.0,
        threshold=0.0,
        confidence=0.0,
        decision="INSPECTION_INVALID",
        reason=reason,
        image_quality="INVALID",
        alignment="NOT_EVALUATED",
        original_image_path=str(
            original_path
        ),
        roi_image_path=None,
        heatmap_path=None,
    )

    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    json_path = (
        inspection_dir
        / "inspection.json"
    )

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
        "quality_checks": quality_result.get(
            "checks",
            {},
        ),
    }

    json_path.write_text(
        json.dumps(
            inspection_json,
            indent=2,
        ),
        encoding="utf-8",
    )

    return inspection_json


def run_inspection(
    image_bytes: bytes,
    filename: str,
    db: Session,
):
    """
    Run one complete VisionQC inspection.

    Pipeline:

        Upload
          ↓
        Image Quality Gate
          ↓
        INVALID → save invalid inspection
          ↓
        VALID
          ↓
        PatchCore
          ↓
        Anomaly Score
          ↓
        Decision
          ↓
        Heatmap + Database + JSON
    """

    inspection_id = str(uuid4())

    safe_filename = Path(filename).name

    input_path = (
        TEMP_UPLOAD_DIR
        / f"{inspection_id}_{safe_filename}"
    )

    input_path.write_bytes(
        image_bytes
    )

    try:
        # --------------------------------------------------
        # 1. IMAGE QUALITY GATE
        # --------------------------------------------------

        quality_result = check_image_quality(
            input_path
        )

        # --------------------------------------------------
        # 2. SAVE ORIGINAL IMAGE
        # --------------------------------------------------

        inspection_dir = (
            get_inspection_directory(
                inspection_id
            )
        )

        original_path = save_original_image(
            inspection_id=inspection_id,
            source_path=input_path,
            filename=safe_filename,
        )

        # --------------------------------------------------
        # 3. STOP IF IMAGE QUALITY IS INVALID
        # --------------------------------------------------

        if not quality_result.get(
            "valid",
            False,
        ):
            return create_invalid_inspection(
                inspection_id=inspection_id,
                filename=safe_filename,
                original_path=original_path,
                quality_result=quality_result,
                db=db,
            )

        # --------------------------------------------------
        # 4. RUN PATCHCORE
        # --------------------------------------------------

        result = inspect_image(
            input_path
        )

        anomaly_map = result.get(
            "anomaly_map"
        )

        score = float(
            result["anomaly_score"]
        )

        threshold = float(
            result["threshold"]
        )

        decision = result["decision"]

        # --------------------------------------------------
        # 5. PROVISIONAL CONFIDENCE
        # --------------------------------------------------

        confidence = (
            calculate_provisional_confidence(
                score=score,
                threshold=threshold,
            )
        )

        # --------------------------------------------------
        # 6. SAVE NORMALIZED ROI
        # --------------------------------------------------

        roi_path = save_roi_image(
            inspection_id=inspection_id,
            source_path=Path(
                result["roi_path"]
            ),
        )

        # --------------------------------------------------
        # 7. GENERATE VISUAL HEATMAP
        # --------------------------------------------------

        heatmap_path = (
            inspection_dir
            / "heatmap.png"
        )

        generate_heatmap(
            image_path=roi_path,
            anomaly_map=anomaly_map,
            output_path=heatmap_path,
        )

        # --------------------------------------------------
        # 8. GENERATE REASON
        # --------------------------------------------------

        if decision == "FAIL":
            reason = (
                "Significant deviation from learned normal."
            )
        else:
            reason = (
                "Within learned normal range."
            )

        # --------------------------------------------------
        # 9. SAVE DATABASE RECORD
        # --------------------------------------------------

        inspection = Inspection(
            inspection_id=inspection_id,
            timestamp=datetime.utcnow(),
            filename=safe_filename,
            product="Water Bottle Cap",
            sku="water_cap_v1",
            model_name="PatchCore",
            model_version="patchcore_water_cap_v2",
            anomaly_score=score,
            threshold=threshold,
            confidence=confidence,
            decision=decision,
            reason=reason,
            image_quality="VALID",
            alignment="VALID",
            original_image_path=str(
                original_path
            ),
            roi_image_path=str(
                roi_path
            ),
            heatmap_path=str(
                heatmap_path
            ),
        )

        db.add(
            inspection
        )

        db.commit()

        db.refresh(
            inspection
        )

        # --------------------------------------------------
        # 10. SAVE COMPLETE JSON RECORD
        # --------------------------------------------------

        json_path = (
            inspection_dir
            / "inspection.json"
        )

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
            "quality_checks": quality_result.get(
                "checks",
                {},
            ),
            "anomaly_map_shape": (
                list(
                    anomaly_map.shape
                )
                if anomaly_map is not None
                else None
            ),
        }

        json_path.write_text(
            json.dumps(
                inspection_json,
                indent=2,
            ),
            encoding="utf-8",
        )

        return inspection_json

    except Exception:
        db.rollback()
        raise

    finally:
        # Temporary uploaded file is deleted.
        if input_path.exists():
            input_path.unlink()