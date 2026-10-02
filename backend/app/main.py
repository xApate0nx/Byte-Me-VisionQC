from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.inspect import router as inspect_router
from app.api.inspections import router as inspections_router
from app.api.settings import router as settings_router
from app.api.stats import router as stats_router
from app.database.database import initialize_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="VisionQC API",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(inspect_router)
app.include_router(inspections_router)
app.include_router(settings_router)
app.include_router(stats_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}