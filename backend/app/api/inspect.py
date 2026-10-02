from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.services.inspection_service import run_inspection


router = APIRouter(
    tags=["Inspection"],
)


@router.post("/api/inspect")
async def inspect_upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing file content type.",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WebP images are supported.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    try:
        result = run_inspection(
            image_bytes=image_bytes,
            filename=file.filename or "inspection.jpg",
            db=db,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inspection failed: {exc}",
        ) from exc

    return {
        "inspection_id": result.get("inspection_id"),
        "filename": result.get("filename"),
        "anomaly_score": result.get("anomaly_score"),
        "threshold": result.get("threshold"),
        "result": result.get("decision"),
        "confidence": result.get("confidence", 0.0),
        "reason": result.get("reason"),
        "image_quality_valid": (
            result.get("image_quality") == "VALID"
        ),
        "quality_reason": result.get(
            "reason",
            "Image quality acceptable.",
        ),
        "product_detected": result.get(
            "product_detected",
            True,
        ),
        "alignment_valid": result.get(
            "alignment_valid",
            result.get("alignment") == "VALID",
        ),
        "image_url": result.get("image_url"),
        "heatmap_url": result.get("heatmap_url"),
        "timestamp": result.get("timestamp"),
    }