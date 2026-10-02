from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.database.database import (
    create_inspection,
    get_setting,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_STORAGE_DIR = (
    PROJECT_ROOT / "storage" / "inspections" / "images"
)


def run_ml_inference(image_bytes: bytes) -> dict[str, Any]:
    """
    Temporary ML adapter.

    This will later call the real PatchCore inference
    implementation from the vision/ directory.
    """

    # TEMPORARY MOCK
    return {
        "anomaly_score": 12.8,
        "heatmap_path": None,
    }


def inspect_image(
    image_bytes: bytes,
    filename: str | None,
) -> dict[str, Any]:
    """
    Run one VisionQC inspection and persist the result.
    """

    if not image_bytes:
        raise ValueError("Image is empty.")

    # --------------------------------------------------
    # 1. Run ML
    # --------------------------------------------------

    ml_result = run_ml_inference(image_bytes)

    anomaly_score = float(
        ml_result["anomaly_score"]
    )

    heatmap_path = ml_result.get("heatmap_path")

    # --------------------------------------------------
    # 2. Get supervisor threshold
    # --------------------------------------------------

    threshold_value = get_setting("threshold")

    if threshold_value is None:
        raise RuntimeError(
            "Inspection threshold is not configured."
        )

    threshold = float(threshold_value)

    # --------------------------------------------------
    # 3. Decision
    # --------------------------------------------------

    result = (
        "FAIL"
        if anomaly_score >= threshold
        else "PASS"
    )

    # --------------------------------------------------
    # 4. Save inspected image
    # --------------------------------------------------

    IMAGE_STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(timezone.utc)

    inspection_filename = (
        f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        f"_{filename or 'inspection.jpg'}"
    )

    image_path = (
        IMAGE_STORAGE_DIR / inspection_filename
    )

    image_path.write_bytes(image_bytes)

    # --------------------------------------------------
    # 5. Save database record
    # --------------------------------------------------

    inspection_id = create_inspection(
        timestamp=timestamp.isoformat(),
        filename=filename,
        anomaly_score=anomaly_score,
        threshold=threshold,
        result=result,
        image_path=str(
            image_path.relative_to(PROJECT_ROOT)
        ),
        heatmap_path=heatmap_path,
    )

    # --------------------------------------------------
    # 6. Return application result
    # --------------------------------------------------

    return {
        "inspection_id": inspection_id,
        "filename": filename,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "result": result,
        "heatmap": heatmap_path,
        "timestamp": timestamp.isoformat(),
    }