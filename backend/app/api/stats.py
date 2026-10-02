from datetime import datetime, time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import Inspection


router = APIRouter(
    prefix="/api/stats",
    tags=["Statistics"],
)


@router.get("/today")
def get_today_stats(db: Session = Depends(get_db)):
    today = datetime.utcnow().date()

    start_of_day = datetime.combine(today, time.min)
    start_of_next_day = datetime.combine(
        today,
        time.min,
    )

    # Move to the next calendar day.
    from datetime import timedelta
    start_of_next_day += timedelta(days=1)

    inspections = (
        db.query(Inspection)
        .filter(
            Inspection.timestamp >= start_of_day,
            Inspection.timestamp < start_of_next_day,
        )
        .all()
    )

    total = len(inspections)

    pass_count = sum(
        1 for inspection in inspections
        if inspection.decision == "PASS"
    )

    review_count = sum(
        1 for inspection in inspections
        if inspection.decision == "REVIEW"
    )

    fail_count = sum(
        1 for inspection in inspections
        if inspection.decision == "FAIL"
    )

    invalid_count = sum(
        1
        for inspection in inspections
        if inspection.decision == "INSPECTION_INVALID"
    )

    valid_inspections = [
        inspection
        for inspection in inspections
        if inspection.decision != "INSPECTION_INVALID"
    ]

    valid_count = len(valid_inspections)

    rejection_rate = (
        (fail_count / valid_count) * 100
        if valid_count > 0
        else 0.0
    )

    average_anomaly_score = (
        sum(
            inspection.anomaly_score
            for inspection in valid_inspections
        ) / valid_count
        if valid_count > 0
        else 0.0
    )

    average_confidence = (
        sum(
            inspection.confidence
            for inspection in valid_inspections
        ) / valid_count
        if valid_count > 0
        else 0.0
    )

    return {
        "success": True,
        "date": today.isoformat(),
        "total_inspections": total,
        "valid_inspections": valid_count,
        "invalid_inspections": invalid_count,
        "pass_count": pass_count,
        "review_count": review_count,
        "fail_count": fail_count,
        "rejection_rate_percent": round(rejection_rate, 2),
        "average_anomaly_score": round(average_anomaly_score, 4),
        "average_confidence": round(average_confidence, 2),
    }