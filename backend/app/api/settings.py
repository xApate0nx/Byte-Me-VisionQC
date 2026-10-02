from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.threshold_service import (
    get_current_threshold,
    set_current_threshold,
)


router = APIRouter(
    prefix="/api/settings",
    tags=["Supervisor Settings"],
)


class ThresholdUpdate(BaseModel):
    threshold: float = Field(
        ...,
        gt=0,
    )


@router.get("/threshold")
def get_threshold():
    return {
        "threshold": get_current_threshold(),
    }


@router.put("/threshold")
def update_threshold(payload: ThresholdUpdate):
    try:
        threshold = set_current_threshold(
            payload.threshold
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "threshold": threshold,
    }


@router.get("")
def get_settings():
    return {
        "success": True,
        "threshold": get_current_threshold(),
        "threshold_type": "anomaly_score",
        "model_name": "PatchCore",
        "model_version": "patchcore_water_cap_v4",
    }