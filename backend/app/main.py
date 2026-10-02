from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.inspect import router as inspect_router
from backend.app.api.inspections import router as inspections_router
from backend.app.api.settings import router as settings_router
from backend.app.api.stats import router as stats_router
from backend.app.database.database import Base, engine
from backend.app.database import models


Base.metadata.create_all(bind=engine)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "inspections"
)


app = FastAPI(
    title="VisionQC API",
    description="Visual Quality Control API",
    version="0.2.0",
)


app.include_router(inspect_router)
app.include_router(inspections_router)
app.include_router(settings_router)
app.include_router(stats_router)


app.mount(
    "/storage",
    StaticFiles(directory=STORAGE_ROOT),
    name="storage",
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
    }