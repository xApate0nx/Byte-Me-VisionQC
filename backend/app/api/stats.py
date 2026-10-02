from fastapi import APIRouter

from app.database.database import get_today_stats


router = APIRouter(
    prefix="/api/stats",
    tags=["statistics"],
)


@router.get("/today")
async def today_stats():
    return get_today_stats()
    