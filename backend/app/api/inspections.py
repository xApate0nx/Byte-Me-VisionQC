from fastapi import APIRouter, Query

from app.database.database import get_inspections


router = APIRouter(
    prefix="/api/inspections",
    tags=["inspections"],
)


@router.get("")
async def list_inspections(
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
):
    return {
        "inspections": get_inspections(limit)
    }