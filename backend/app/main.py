from fastapi import FastAPI

from backend.app.api.inspect import router as inspect_router
from backend.app.api.inspections import router as inspections_router
from backend.app.api.settings import router as settings_router
from backend.app.api.stats import router as stats_router
from backend.app.database.database import Base, engine
from backend.app.database import models


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="VisionQC API",
    description="Visual Quality Control API for industrial inspection",
    version="0.1.0",
)


app.include_router(inspect_router)
app.include_router(inspections_router)
app.include_router(settings_router)
app.include_router(stats_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "VisionQC API",
        "version": "0.1.0",
    }