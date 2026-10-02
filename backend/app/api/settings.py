from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.database.database import get_setting, set_setting


router = APIRouter(
    prefix="/api/settings",
    tags=["settings"],
)


class ThresholdUpdate(BaseModel):
    threshold: float = Field(gt=0)


@router.get("/threshold")
async def get_threshold():
    value = get_setting("threshold")

    if value is None:
        raise HTTPException(
            status_code=500,
            detail="Threshold is not configured",
        )

    return {
        "threshold": float(value),
    }


@router.post("/threshold")
async def update_threshold(data: ThresholdUpdate):
    set_setting("threshold", data.threshold)

    return {
        "threshold": data.threshold,
    }