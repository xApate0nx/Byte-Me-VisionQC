from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import Inspection


router = APIRouter(
    prefix="/api/inspections",
    tags=["Inspection History"],
)


class SupervisorFeedback(BaseModel):
    verdict: str
    feedback: str | None = None


@router.get("")
def get_inspections(
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_db),
):
    inspections = (
        db.query(Inspection)
        .order_by(Inspection.timestamp.desc())
        .limit(limit)
        .all()
    )

    return {
        "inspections": [
            {
                "inspection_id": inspection.inspection_id,
                "id": inspection.id,
                "timestamp": inspection.timestamp.isoformat(),
                "filename": inspection.filename,
                "result": inspection.decision,
                "decision": inspection.decision,
                "anomaly_score": inspection.anomaly_score,
                "threshold": inspection.threshold,
                "confidence": inspection.confidence,
                "reason": inspection.reason,
                "image_quality_valid": (
                    inspection.image_quality == "VALID"
                ),
                "product_detected": (
                    inspection.decision != "INSPECTION_INVALID"
                ),
                "alignment_valid": (
                    inspection.alignment == "VALID"
                ),
                "image_url": inspection.original_image_path,
                "heatmap_url": inspection.heatmap_path,
                "roi_image_path": inspection.roi_image_path,
                "supervisor_verdict": inspection.supervisor_verdict,
                "supervisor_feedback": inspection.supervisor_feedback,
            }
            for inspection in inspections
        ]
    }


@router.post("/{inspection_id}/feedback")
def add_supervisor_feedback(
    inspection_id: str,
    feedback: SupervisorFeedback,
    db: Session = Depends(get_db),
):
    allowed_verdicts = {
        "PASS",
        "FAIL",
        "REVIEW",
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE",
    }

    verdict = feedback.verdict.upper().strip()

    if verdict not in allowed_verdicts:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid supervisor verdict. "
                "Use PASS, FAIL, REVIEW, "
                "FALSE_POSITIVE, or FALSE_NEGATIVE."
            ),
        )

    inspection = (
        db.query(Inspection)
        .filter(
            Inspection.inspection_id == inspection_id
        )
        .first()
    )

    if inspection is None:
        raise HTTPException(
            status_code=404,
            detail="Inspection not found.",
        )

    inspection.supervisor_verdict = verdict
    inspection.supervisor_feedback = feedback.feedback

    db.commit()
    db.refresh(inspection)

    return {
        "success": True,
        "message": "Supervisor feedback saved.",
        "inspection": {
            "inspection_id": inspection.inspection_id,
            "ai_decision": inspection.decision,
            "supervisor_verdict": inspection.supervisor_verdict,
            "supervisor_feedback": inspection.supervisor_feedback,
        },
    }