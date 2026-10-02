from __future__ import annotations

import numpy as np


def extract_spatial_features(heatmap: np.ndarray) -> dict[str, float]:
    """
    Extract spatial characteristics from a PatchCore anomaly heatmap.

    Accepts:
        (H, W)
        (1, H, W)
        (H, W, 1)

    Returns normalized spatial features.
    """

    heatmap = np.asarray(heatmap, dtype=np.float32)

    # Normalize common PatchCore output shapes to 2D.
    if heatmap.ndim == 3:
        if heatmap.shape[0] == 1:
            heatmap = heatmap[0]
        elif heatmap.shape[-1] == 1:
            heatmap = heatmap[..., 0]
        else:
            raise ValueError(
                f"Expected single-channel heatmap, got {heatmap.shape}"
            )

    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {heatmap.shape}")

    if not np.all(np.isfinite(heatmap)):
        heatmap = np.nan_to_num(
            heatmap,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    h, w = heatmap.shape

    if h == 0 or w == 0:
        raise ValueError(f"Invalid heatmap shape: {heatmap.shape}")

    # ---------------------------------------------------------
    # Top 5% anomaly mask
    # ---------------------------------------------------------
    percentile = np.percentile(heatmap, 95.0)

    mask = heatmap >= percentile

    # ---------------------------------------------------------
    # Coordinate grids
    # ---------------------------------------------------------
    yy, xx = np.mgrid[0:h, 0:w]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # ---------------------------------------------------------
    # Central anomaly fraction
    # Within 30% of image center.
    # ---------------------------------------------------------
    dx = (xx - cx) / max(w / 2.0, 1.0)
    dy = (yy - cy) / max(h / 2.0, 1.0)

    radial_distance = np.sqrt(dx ** 2 + dy ** 2)

    central_mask = radial_distance <= 0.30

    # ---------------------------------------------------------
    # Border anomaly fraction
    # Outside the central 60% region.
    # ---------------------------------------------------------
    border_mask = (
        (xx < 0.20 * w)
        | (xx >= 0.80 * w)
        | (yy < 0.20 * h)
        | (yy >= 0.80 * h)
    )

    total_anomaly_pixels = max(int(mask.sum()), 1)

    central_anomaly_fraction = float(
        np.logical_and(mask, central_mask).sum()
        / total_anomaly_pixels
    )

    border_anomaly_fraction = float(
        np.logical_and(mask, border_mask).sum()
        / total_anomaly_pixels
    )

    # ---------------------------------------------------------
    # Weighted anomaly centroid
    # ---------------------------------------------------------
    positive_heatmap = np.maximum(heatmap, 0.0)
    mass = float(positive_heatmap.sum())

    if mass > 0:
        centroid_x = float((xx * positive_heatmap).sum() / mass)
        centroid_y = float((yy * positive_heatmap).sum() / mass)

        normalized_cx = (centroid_x - cx) / max(w / 2.0, 1.0)
        normalized_cy = (centroid_y - cy) / max(h / 2.0, 1.0)

        centroid_distance = float(
            np.sqrt(normalized_cx ** 2 + normalized_cy ** 2)
        )
    else:
        centroid_x = cx
        centroid_y = cy
        centroid_distance = 0.0

    # ---------------------------------------------------------
    # Largest connected-ish anomaly region approximation
    # ---------------------------------------------------------
    # Use connected components if OpenCV is available.
    largest_region_fraction = 0.0

    try:
        import cv2

        binary = mask.astype(np.uint8)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        if num_labels > 1:
            region_sizes = stats[1:, cv2.CC_STAT_AREA]

            if len(region_sizes):
                largest_region = float(np.max(region_sizes))
                largest_region_fraction = (
                    largest_region / total_anomaly_pixels
                )

    except Exception:
        # Safe fallback if OpenCV component analysis fails.
        largest_region_fraction = 0.0

    # ---------------------------------------------------------
    # Compactness
    # ---------------------------------------------------------
    anomaly_compactness = largest_region_fraction

    # ---------------------------------------------------------
    # Overall anomaly mass
    # Normalize by image size.
    # ---------------------------------------------------------
    anomaly_mass = float(positive_heatmap.mean())

    return {
        "central_anomaly_fraction": central_anomaly_fraction,
        "border_anomaly_fraction": border_anomaly_fraction,
        "anomaly_centroid_x": float(centroid_x / max(w, 1)),
        "anomaly_centroid_y": float(centroid_y / max(h, 1)),
        "centroid_distance": centroid_distance,
        "largest_region_fraction": largest_region_fraction,
        "anomaly_compactness": anomaly_compactness,
        "anomaly_mass": anomaly_mass,
    }