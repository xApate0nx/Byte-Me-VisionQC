from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Prototype image-quality thresholds
# --------------------------------------------------

MIN_WIDTH = 400
MIN_HEIGHT = 400

# Laplacian variance.
# Calibrated against the current 21 normal images.
BLUR_THRESHOLD = 5.0

# Mean grayscale brightness.
DARK_THRESHOLD = 35.0
BRIGHT_THRESHOLD = 225.0

# Broad prototype product-visibility range.
MIN_PRODUCT_OCCUPANCY = 0.08
MAX_PRODUCT_OCCUPANCY = 0.97


# --------------------------------------------------
# Resolution
# --------------------------------------------------

def check_resolution(
    image: np.ndarray,
) -> tuple[bool, str]:
    """
    Check whether the uploaded image has sufficient resolution.
    """

    height, width = image.shape[:2]

    if (
        width < MIN_WIDTH
        or height < MIN_HEIGHT
    ):
        return (
            False,
            f"Image resolution too low: {width}x{height}.",
        )

    return (
        True,
        "Resolution acceptable.",
    )


# --------------------------------------------------
# Blur / sharpness
# --------------------------------------------------

def calculate_blur_score(
    image: np.ndarray,
) -> float:
    """
    Calculate Laplacian variance as a simple sharpness metric.

    Higher values generally indicate a sharper image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def check_blur(
    image: np.ndarray,
) -> tuple[bool, str, float]:
    """
    Check whether the image is severely blurred.
    """

    blur_score = calculate_blur_score(
        image
    )

    if blur_score < BLUR_THRESHOLD:
        return (
            False,
            (
                "Image appears too blurry: "
                f"score={blur_score:.2f}."
            ),
            blur_score,
        )

    return (
        True,
        (
            "Image sharpness acceptable: "
            f"score={blur_score:.2f}."
        ),
        blur_score,
    )


# --------------------------------------------------
# Exposure
# --------------------------------------------------

def check_exposure(
    image: np.ndarray,
) -> tuple[bool, str, float]:
    """
    Check whether the image is severely underexposed
    or overexposed.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    mean_brightness = float(
        gray.mean()
    )

    if mean_brightness < DARK_THRESHOLD:
        return (
            False,
            (
                "Image is too dark: "
                f"brightness={mean_brightness:.2f}."
            ),
            mean_brightness,
        )

    if mean_brightness > BRIGHT_THRESHOLD:
        return (
            False,
            (
                "Image is too bright: "
                f"brightness={mean_brightness:.2f}."
            ),
            mean_brightness,
        )

    return (
        True,
        (
            "Exposure acceptable: "
            f"brightness={mean_brightness:.2f}."
        ),
        mean_brightness,
    )


# --------------------------------------------------
# Product occupancy
# --------------------------------------------------

def calculate_product_occupancy(
    image: np.ndarray,
) -> float:
    """
    Estimate how much of the image contains
    non-background content.

    This is a broad prototype heuristic,
    not a trained object detector.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # Estimate background intensity
    # from the image borders.
    border_pixels = np.concatenate(
        [
            blurred[0, :],
            blurred[-1, :],
            blurred[:, 0],
            blurred[:, -1],
        ]
    )

    background_value = float(
        np.median(border_pixels)
    )

    difference = cv2.absdiff(
        blurred,
        np.full_like(
            blurred,
            int(background_value),
        ),
    )

    _, mask = cv2.threshold(
        difference,
        15,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    occupancy = float(
        np.count_nonzero(mask)
        / mask.size
    )

    return occupancy


def check_product_occupancy(
    image: np.ndarray,
) -> tuple[bool, str, float]:
    """
    Perform a broad product-position/visibility check.
    """

    occupancy = calculate_product_occupancy(
        image
    )

    if occupancy < MIN_PRODUCT_OCCUPANCY:
        return (
            False,
            (
                "Expected product region is not "
                "sufficiently visible: "
                f"occupancy={occupancy:.3f}."
            ),
            occupancy,
        )

    if occupancy > MAX_PRODUCT_OCCUPANCY:
        return (
            False,
            (
                "Image contains excessive "
                "non-background content: "
                f"occupancy={occupancy:.3f}."
            ),
            occupancy,
        )

    return (
        True,
        (
            "Product visibility acceptable: "
            f"occupancy={occupancy:.3f}."
        ),
        occupancy,
    )


# --------------------------------------------------
# Complete image-quality gate
# --------------------------------------------------

def check_image_quality(
    image_path: str | Path,
) -> dict:
    """
    Run all image-quality checks.

    Every check is evaluated before deciding whether
    the image is valid. This gives the inspection service
    a complete diagnostic result.
    """

    image_path = Path(
        image_path
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return {
            "valid": False,
            "status": "INVALID",
            "reason": "Could not read image file.",
            "checks": {},
        }

    checks = {}

    # --------------------------------------------------
    # Resolution
    # --------------------------------------------------

    resolution_valid, resolution_reason = (
        check_resolution(image)
    )

    checks["resolution"] = {
        "valid": resolution_valid,
        "reason": resolution_reason,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }

    # --------------------------------------------------
    # Blur
    # --------------------------------------------------

    blur_valid, blur_reason, blur_score = (
        check_blur(image)
    )

    checks["blur"] = {
        "valid": blur_valid,
        "reason": blur_reason,
        "score": blur_score,
        "threshold": BLUR_THRESHOLD,
    }

    # --------------------------------------------------
    # Exposure
    # --------------------------------------------------

    exposure_valid, exposure_reason, brightness = (
        check_exposure(image)
    )

    checks["exposure"] = {
        "valid": exposure_valid,
        "reason": exposure_reason,
        "mean_brightness": brightness,
        "dark_threshold": DARK_THRESHOLD,
        "bright_threshold": BRIGHT_THRESHOLD,
    }

    # --------------------------------------------------
    # Product visibility
    # --------------------------------------------------

    product_valid, product_reason, occupancy = (
        check_product_occupancy(image)
    )

    checks["product_visibility"] = {
        "valid": product_valid,
        "reason": product_reason,
        "occupancy": occupancy,
        "minimum": MIN_PRODUCT_OCCUPANCY,
        "maximum": MAX_PRODUCT_OCCUPANCY,
    }

    # --------------------------------------------------
    # Overall result
    # --------------------------------------------------

    all_valid = (
        resolution_valid
        and blur_valid
        and exposure_valid
        and product_valid
    )

    if not all_valid:

        failed_checks = []

        if not resolution_valid:
            failed_checks.append(
                "resolution"
            )

        if not blur_valid:
            failed_checks.append(
                "blur"
            )

        if not exposure_valid:
            failed_checks.append(
                "exposure"
            )

        if not product_valid:
            failed_checks.append(
                "product visibility"
            )

        reason = (
            "Image quality check failed: "
            + ", ".join(failed_checks)
            + "."
        )

        return {
            "valid": False,
            "status": "INVALID",
            "reason": reason,
            "checks": checks,
        }

    return {
        "valid": True,
        "status": "VALID",
        "reason": "Image quality checks passed.",
        "checks": checks,
    }