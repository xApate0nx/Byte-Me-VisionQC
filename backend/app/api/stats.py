from datetime import datetime, time, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import Inspection


router = APIRouter(
    prefix="/api/stats",
    tags=["Statistics"],
)


@router.get("/today")
def get_today_stats(
    db: Session = Depends(get_db),
):
    today = datetime.utcnow().date()

    start_of_day = datetime.combine(
        today,
        time.min,
    )

    start_of_next_day = (
        start_of_day + timedelta(days=1)
    )

    inspections = (
        db.query(Inspection)
        .filter(
            Inspection.timestamp >= start_of_day,
            Inspection.timestamp < start_of_next_day,
        )
        .all()
    )

    total = len(inspections)

    passed = sum(
        1 for x in inspections
        if x.decision == "PASS"
    )

    failed = sum(
        1 for x in inspections
        if x.decision == "FAIL"
    )

    review = sum(
        1 for x in inspections
        if x.decision == "REVIEW"
    )

    invalid = sum(
        1 for x in inspections
        if x.decision == "INSPECTION_INVALID"
    )

    valid = total - invalid

    rejection_rate = (
        (failed / valid) * 100
        if valid > 0
        else 0.0
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "review": review,
        "invalid": invalid,
        "rejection_rate": round(
            rejection_rate,
            2,
        ),
    }