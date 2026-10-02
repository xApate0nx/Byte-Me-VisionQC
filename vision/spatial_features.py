from pathlib import Path
import numpy as np
import cv2


def extract_spatial_features(heatmap):
    """
    Extract spatial characteristics from a PatchCore anomaly map.

    Returns normalized, model-independent features.
    """

    heatmap = np.asarray(heatmap, dtype=np.float32)

    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {heatmap.shape}")

    h, w = heatmap.shape

    # ------------------------------------------------------------
    # Normalize anomaly map to 0..1
    # ------------------------------------------------------------

    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))

    if maximum > minimum:
        normalized = (heatmap - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(heatmap)

    # ------------------------------------------------------------
    # Top 5% anomaly region
    # ------------------------------------------------------------

    p95 = np.percentile(heatmap, 95)
    high_mask = (heatmap >= p95).astype(np.uint8)

    high_pixels = int(np.sum(high_mask))

    if high_pixels == 0:
        return {
            "central_anomaly_fraction": 0.0,
            "border_anomaly_fraction": 0.0,
            "anomaly_centroid_x": 0.5,
            "anomaly_centroid_y": 0.5,
            "centroid_distance": 0.0,
            "largest_region_fraction": 0.0,
            "anomaly_compactness": 0.0,
            "anomaly_mass": 0.0,
        }

    ys, xs = np.where(high_mask > 0)

    # ------------------------------------------------------------
    # Central anomaly fraction
    # ------------------------------------------------------------

    cx = w / 2.0
    cy = h / 2.0

    central_mask = (
        (np.abs(xs - cx) <= 0.30 * w)
        & (np.abs(ys - cy) <= 0.30 * h)
    )

    central_fraction = float(np.mean(central_mask))

    # ------------------------------------------------------------
    # Border anomaly fraction
    # ------------------------------------------------------------

    border_mask = (
        (xs < 0.20 * w)
        | (xs >= 0.80 * w)
        | (ys < 0.20 * h)
        | (ys >= 0.80 * h)
    )

    border_fraction = float(np.mean(border_mask))

    # ------------------------------------------------------------
    # Anomaly centroid
    # ------------------------------------------------------------

    centroid_x = float(np.mean(xs) / w)
    centroid_y = float(np.mean(ys) / h)

    centroid_distance = float(
        np.sqrt(
            (centroid_x - 0.5) ** 2
            + (centroid_y - 0.5) ** 2
        )
    )

    # ------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        high_mask,
        connectivity=8,
    )

    component_sizes = stats[1:, cv2.CC_STAT_AREA]

    if len(component_sizes) > 0:
        largest_component = int(np.max(component_sizes))
    else:
        largest_component = 0

    largest_region_fraction = (
        largest_component / high_pixels
    )

    # ------------------------------------------------------------
    # Compactness
    # ------------------------------------------------------------

    # A concentrated anomaly occupies fewer pixels.
    # A scattered anomaly produces many disconnected pixels.
    compactness = float(largest_region_fraction)

    # ------------------------------------------------------------
    # Weighted anomaly mass
    # ------------------------------------------------------------

    anomaly_mass = float(np.mean(normalized))

    return {
        "central_anomaly_fraction": central_fraction,
        "border_anomaly_fraction": border_fraction,
        "anomaly_centroid_x": centroid_x,
        "anomaly_centroid_y": centroid_y,
        "centroid_distance": centroid_distance,
        "largest_region_fraction": largest_region_fraction,
        "anomaly_compactness": compactness,
        "anomaly_mass": anomaly_mass,
    }


def extract_from_file(path):
    heatmap = np.load(path)
    return extract_spatial_features(heatmap)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    HEATMAP_ROOT = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "patchcore_v3_heatmaps"
    )

    for category in ["normal", "defects"]:

        directory = HEATMAP_ROOT / category

        print()
        print("=" * 70)
        print(category.upper())
        print("=" * 70)

        for path in sorted(directory.glob("*.npy")):

            features = extract_from_file(path)

            print()
            print(path.name)

            for key, value in features.items():
                print(f"  {key:30} {value:.4f}")