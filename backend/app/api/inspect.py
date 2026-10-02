from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import Inspection


router = APIRouter(
    prefix="/api/inspections",
    tags=["Inspection History"],
)


@router.get("")
def get_inspections(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    inspections = (
        db.query(Inspection)
        .order_by(Inspection.timestamp.desc())
        .limit(limit)
        .all()
    )

    return {
        "success": True,
        "count": len(inspections),
        "inspections": [
            {
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
                "decision": inspection.decision,
                "reason": inspection.reason,
                "image_quality": inspection.image_quality,
                "alignment": inspection.alignment,
                "original_image_path": inspection.original_image_path,
                "roi_image_path": inspection.roi_image_path,
                "heatmap_path": inspection.heatmap_path,
                "supervisor_verdict": inspection.supervisor_verdict,
                "supervisor_feedback": inspection.supervisor_feedback,
            }
            for inspection in inspections
        ],
    }