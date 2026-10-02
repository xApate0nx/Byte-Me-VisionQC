from pathlib import Path

import cv2
import numpy as np


# Minimum acceptable input dimensions.
MIN_WIDTH = 400
MIN_HEIGHT = 400

# Blur threshold.
# Lower variance generally means a blurrier image.
BLUR_THRESHOLD = 5.0

# Exposure thresholds.
DARK_THRESHOLD = 35.0
BRIGHT_THRESHOLD = 225.0

# Expected product occupancy inside the normalized 80% ROI.
# These are intentionally broad for the prototype.
MIN_PRODUCT_OCCUPANCY = 0.08
MAX_PRODUCT_OCCUPANCY = 0.97


def check_resolution(image: np.ndarray) -> tuple[bool, str]:
    """
    Check whether the uploaded image has sufficient resolution.
    """

    height, width = image.shape[:2]

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return (
            False,
            f"Image resolution too low: {width}x{height}.",
        )

    return True, "Resolution acceptable."


def calculate_blur_score(image: np.ndarray) -> float:
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


def check_blur(image: np.ndarray) -> tuple[bool, str, float]:
    """
    Check whether the image is severely blurred.
    """

    blur_score = calculate_blur_score(image)

    if blur_score < BLUR_THRESHOLD:
        return (
            False,
            f"Image appears too blurry: score={blur_score:.2f}.",
            blur_score,
        )

    return (
        True,
        f"Image sharpness acceptable: score={blur_score:.2f}.",
        blur_score,
    )


def check_exposure(image: np.ndarray) -> tuple[bool, str, float]:
    """
    Check whether the image is severely underexposed
    or overexposed.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    mean_brightness = float(gray.mean())

    if mean_brightness < DARK_THRESHOLD:
        return (
            False,
            f"Image is too dark: brightness={mean_brightness:.2f}.",
            mean_brightness,
        )

    if mean_brightness > BRIGHT_THRESHOLD:
        return (
            False,
            f"Image is too bright: brightness={mean_brightness:.2f}.",
            mean_brightness,
        )

    return (
        True,
        f"Exposure acceptable: brightness={mean_brightness:.2f}.",
        mean_brightness,
    )


def calculate_product_occupancy(image: np.ndarray) -> float:
    """
    Estimate how much of the image contains non-background content.

    This is deliberately a broad prototype heuristic, not a trained
    object detector.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Blur slightly to reduce small image noise.
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # Estimate background using border pixels.
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

    # Difference from estimated background.
    difference = cv2.absdiff(
        blurred,
        np.full_like(
            blurred,
            int(background_value),
        ),
    )

    # Threshold based on difference from background.
    _, mask = cv2.threshold(
        difference,
        15,
        255,
        cv2.THRESH_BINARY,
    )

    # Remove small noise.
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
                "Expected product region is not sufficiently "
                f"visible: occupancy={occupancy:.3f}."
            ),
            occupancy,
        )

    if occupancy > MAX_PRODUCT_OCCUPANCY:
        return (
            False,
            (
                "Image contains excessive non-background "
                f"content: occupancy={occupancy:.3f}."
            ),
            occupancy,
        )

    return (
        True,
        f"Product visibility acceptable: occupancy={occupancy:.3f}.",
        occupancy,
    )


def check_image_quality(
    image_path: str | Path,
) -> dict:
    """
    Run all image-quality checks.

    Returns a structured result that the inspection service
    can use before running PatchCore.
    """

    image_path = Path(image_path)

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

    # Resolution
    resolution_valid, resolution_reason = check_resolution(
        image
    )
def check_image_quality(
    image_path: str | Path,
) -> dict:
    """
    Run all image-quality checks.

    All checks are evaluated before deciding whether the image
    is valid. This allows the system to provide a complete
    diagnostic result.
    """

    image_path = Path(image_path)

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
    # 1. Resolution
    # --------------------------------------------------

    resolution_valid, resolution_reason = check_resolution(
        image
    )

    checks["resolution"] = {
        "valid": resolution_valid,
        "reason": resolution_reason,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }

    # --------------------------------------------------
    # 2. Blur
    # --------------------------------------------------

    blur_valid, blur_reason, blur_score = check_blur(
        image
    )

    checks["blur"] = {
        "valid": blur_valid,
        "reason": blur_reason,
        "score": blur_score,
        "threshold": BLUR_THRESHOLD,
    }

    # --------------------------------------------------
    # 3. Exposure
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
    # 4. Product visibility
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
    # Overall decision
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
            failed_checks.append("resolution")

        if not blur_valid:
            failed_checks.append("blur")

        if not exposure_valid:
            failed_checks.append("exposure")

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