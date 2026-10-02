from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD ANOMALY MAP
# ============================================================

def load_anomaly_map(path: Path):

    anomaly_map = np.load(path)

    # Remove unnecessary dimensions
    anomaly_map = np.squeeze(anomaly_map)

    # Make sure we have a 2D map
    if anomaly_map.ndim != 2:
        raise ValueError(
            f"Expected 2D anomaly map, got shape "
            f"{anomaly_map.shape}"
        )

    return anomaly_map


# ============================================================
# NORMALIZE MAP
# ============================================================

def normalize_map(anomaly_map):

    anomaly_map = anomaly_map.astype(np.float32)

    min_value = float(np.min(anomaly_map))
    max_value = float(np.max(anomaly_map))

    if max_value - min_value < 1e-8:
        return np.zeros_like(anomaly_map)

    normalized = (
        anomaly_map - min_value
    ) / (
        max_value - min_value
    )

    return normalized


# ============================================================
# FIND PEAK REGION
# ============================================================

def find_peak_region(normalized_map):

    peak_y, peak_x = np.unravel_index(
        np.argmax(normalized_map),
        normalized_map.shape,
    )

    return peak_x, peak_y


# ============================================================
# GENERATE HEATMAP
# ============================================================

def generate_heatmap(
    image_path: Path,
    anomaly_map_path: Path,
):

    image = Image.open(image_path).convert("RGB")

    anomaly_map = load_anomaly_map(
        anomaly_map_path
    )

    normalized = normalize_map(
        anomaly_map
    )

    peak_x, peak_y = find_peak_region(
        normalized
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 5)
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    ax1 = fig.add_subplot(1, 2, 1)

    ax1.imshow(image)

    ax1.set_title(
        "Inspection Image"
    )

    ax1.axis("off")

    # --------------------------------------------------------
    # Heatmap overlay
    # --------------------------------------------------------

    ax2 = fig.add_subplot(1, 2, 2)

    ax2.imshow(image)

    heatmap = ax2.imshow(
        normalized,
        cmap="jet",
        alpha=0.55,
        interpolation="bilinear",
    )

    ax2.set_title(
        "PatchCore Deviation Heatmap"
    )

    ax2.axis("off")

    fig.colorbar(
        heatmap,
        ax=ax2,
        fraction=0.046,
        pad=0.04,
        label="Relative anomaly intensity",
    )

    fig.suptitle(
        "VisionQC — Inspection Analysis",
        fontsize=14,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "inspections"
        / "heatmaps"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{image_path.stem}_visual_heatmap.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path, peak_x, peak_y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "VisionQC — PatchCore Visual Heatmap"
    )
    print("=" * 70)

    if len(sys.argv) < 3:

        print()
        print("Usage:")

        print(
            'python vision/generate_inspection_heatmap.py '
            '"image_path" "anomaly_map.npy"'
        )

        sys.exit(1)

    image_path = Path(
        sys.argv[1]
    )

    anomaly_map_path = Path(
        sys.argv[2]
    )

    if not image_path.is_absolute():
        image_path = (
            PROJECT_ROOT / image_path
        )

    if not anomaly_map_path.is_absolute():
        anomaly_map_path = (
            PROJECT_ROOT / anomaly_map_path
        )

    image_path = image_path.resolve()
    anomaly_map_path = anomaly_map_path.resolve()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not anomaly_map_path.exists():

        raise FileNotFoundError(
            f"Anomaly map not found:\n"
            f"{anomaly_map_path}"
        )

    print()
    print(
        f"Image       : {image_path.name}"
    )

    print(
        f"Anomaly map : {anomaly_map_path.name}"
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    output_path, peak_x, peak_y = (
        generate_heatmap(
            image_path,
            anomaly_map_path,
        )
    )

    print()
    print(
        f"Peak anomaly X : {peak_x}"
    )

    print(
        f"Peak anomaly Y : {peak_y}"
    )

    print()
    print(
        "Visual heatmap saved:"
    )

    print(
        output_path.relative_to(
            PROJECT_ROOT
        )
    )

    print()
    print("Heatmap generation completed.")


if __name__ == "__main__":
    main()