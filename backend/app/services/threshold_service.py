from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

THRESHOLD_FILE = (
    PROJECT_ROOT
    / "data"
    / "database"
    / "threshold.txt"
)

DEFAULT_THRESHOLD = 12.3785223961


def get_current_threshold() -> float:
    """
    Return the currently configured supervisor threshold.

    If no threshold has been configured yet, return the
    calibrated prototype default.
    """

    if not THRESHOLD_FILE.exists():
        return DEFAULT_THRESHOLD

    try:
        value = float(THRESHOLD_FILE.read_text().strip())

        if value <= 0:
            return DEFAULT_THRESHOLD

        return value

    except (ValueError, OSError):
        return DEFAULT_THRESHOLD


def set_current_threshold(threshold: float) -> float:
    """
    Save a new supervisor threshold.

    This changes only the decision boundary.
    It does NOT retrain or modify the PatchCore model.
    """

    threshold = float(threshold)

    if threshold <= 0:
        raise ValueError("Threshold must be greater than 0.")

    THRESHOLD_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    THRESHOLD_FILE.write_text(
        str(threshold),
        encoding="utf-8",
    )

    return threshold