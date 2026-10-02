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
        description="PatchCore anomaly threshold used for PASS/FAIL decisions.",
    )


@router.get("")
def get_settings():
    return {
        "success": True,
        "threshold": get_current_threshold(),
        "threshold_type": "anomaly_score",
        "model_name": "PatchCore",
        "model_version": "patchcore_water_cap_v2",
    }


@router.post("/threshold")
def update_threshold(payload: ThresholdUpdate):
    try:
        new_threshold = set_current_threshold(payload.threshold)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "success": True,
        "message": "Supervisor threshold updated successfully.",
        "threshold": new_threshold,
        "threshold_type": "anomaly_score",
        "model_name": "PatchCore",
        "model_version": "patchcore_water_cap_v2",
    }