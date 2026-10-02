from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INSPECTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "inspections"
)

INSPECTIONS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_inspection_directory(inspection_id: str) -> Path:
    """
    Return and create the permanent directory
    for one inspection.
    """

    inspection_dir = INSPECTIONS_DIR / inspection_id
    inspection_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return inspection_dir


def save_original_image(
    inspection_id: str,
    source_path: Path,
    filename: str,
) -> Path:
    """
    Permanently save the original uploaded image.
    """

    inspection_dir = get_inspection_directory(
        inspection_id
    )

    safe_filename = Path(filename).name

    destination = (
        inspection_dir
        / f"original_{safe_filename}"
    )

    shutil.copy2(
        source_path,
        destination,
    )

    return destination


def save_roi_image(
    inspection_id: str,
    source_path: Path,
) -> Path:
    """
    Permanently save the normalized ROI image.
    """

    inspection_dir = get_inspection_directory(
        inspection_id
    )

    destination = (
        inspection_dir
        / "roi.png"
    )

    shutil.copy2(
        source_path,
        destination,
    )

    return destination