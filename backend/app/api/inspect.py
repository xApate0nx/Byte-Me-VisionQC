from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.inspection_service import inspect_image


router = APIRouter(
    prefix="/api",
    tags=["inspection"],
)


@router.post("/inspect")
async def inspect(
    file: Annotated[UploadFile, File()],
):
    try:
        image_bytes = await file.read()

        result = inspect_image(
            image_bytes=image_bytes,
            filename=file.filename,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc