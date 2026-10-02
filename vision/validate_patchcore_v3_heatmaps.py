from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HEATMAP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_v3_heatmaps"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_v3_heatmap_validation.json"
)


def region(y, x, h, w):
    """Classify a point into a 3x3 image region."""
    row = min(2, int(y / (h / 3)))
    col = min(2, int(x / (w / 3)))

    names = [
        ["TOP_LEFT", "TOP_CENTER", "TOP_RIGHT"],
        ["MIDDLE_LEFT", "CENTER", "MIDDLE_RIGHT"],
        ["BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"],
    ]

    return names[row][col]


def analyze_file(path):
    heatmap = np.load(path)

    h, w = heatmap.shape

    # Highest anomaly locations
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    max_y, max_x = max_idx
    max_value = float(heatmap[max_idx])

    # Top 5% anomaly pixels
    percentile_95 = np.percentile(heatmap, 95)
    high = heatmap >= percentile_95

    ys, xs = np.where(high)

    # Image center
    cy = h / 2
    cx = w / 2

    distances = np.sqrt(
        ((ys - cy) / h) ** 2 +
        ((xs - cx) / w) ** 2
    )

    # Inner central 60% area
    inner = (
        (np.abs(xs - cx) <= 0.30 * w)
        & (np.abs(ys - cy) <= 0.30 * h)
    )

    central_high_fraction = float(np.mean(inner))

    # Border 20% area
    border = (
        (xs < 0.20 * w)
        | (xs >= 0.80 * w)
        | (ys < 0.20 * h)
        | (ys >= 0.80 * h)
    )

    border_high_fraction = float(np.mean(border))

    return {
        "filename": path.stem,
        "shape": [h, w],
        "max_value": max_value,
        "max_y": int(max_y),
        "max_x": int(max_x),
        "max_region": region(max_y, max_x, h, w),
        "p95": float(percentile_95),
        "central_high_fraction": central_high_fraction,
        "border_high_fraction": border_high_fraction,
        "high_pixel_count": int(len(xs)),
    }


def summarize(results):
    if not results:
        return {}

    return {
        "count": len(results),
        "mean_max_value": float(
            np.mean([r["max_value"] for r in results])
        ),
        "mean_central_high_fraction": float(
            np.mean([r["central_high_fraction"] for r in results])
        ),
        "mean_border_high_fraction": float(
            np.mean([r["border_high_fraction"] for r in results])
        ),
        "max_region_counts": {
            region_name: sum(
                r["max_region"] == region_name
                for r in results
            )
            for region_name in [
                "TOP_LEFT",
                "TOP_CENTER",
                "TOP_RIGHT",
                "MIDDLE_LEFT",
                "CENTER",
                "MIDDLE_RIGHT",
                "BOTTOM_LEFT",
                "BOTTOM_CENTER",
                "BOTTOM_RIGHT",
            ]
        },
    }


def collect(category):
    directory = HEATMAP_ROOT / category

    results = []

    for path in sorted(directory.glob("*.npy")):
        results.append(analyze_file(path))

    return results


def main():
    normal = collect("normal")
    defects = collect("defects")

    output = {
        "model": "PatchCore v3",
        "normal": {
            "summary": summarize(normal),
            "images": normal,
        },
        "defects": {
            "summary": summarize(defects),
            "images": defects,
        },
    }

    OUTPUT.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("PATCHCORE V3 HEATMAP VALIDATION")
    print("=" * 70)

    print()
    print("NORMAL")
    print("-" * 70)
    print("Images:", len(normal))
    print(
        "Mean central high-anomaly fraction:",
        f"{output['normal']['summary']['mean_central_high_fraction']:.4f}"
    )
    print(
        "Mean border high-anomaly fraction:",
        f"{output['normal']['summary']['mean_border_high_fraction']:.4f}"
    )

    print()
    print("DEFECTS")
    print("-" * 70)
    print("Images:", len(defects))
    print(
        "Mean central high-anomaly fraction:",
        f"{output['defects']['summary']['mean_central_high_fraction']:.4f}"
    )
    print(
        "Mean border high-anomaly fraction:",
        f"{output['defects']['summary']['mean_border_high_fraction']:.4f}"
    )

    print()
    print("MAX-ANOMALY REGION — NORMAL")
    print("-" * 70)
    for k, v in output["normal"]["summary"]["max_region_counts"].items():
        print(f"{k:15} {v}")

    print()
    print("MAX-ANOMALY REGION — DEFECTS")
    print("-" * 70)
    for k, v in output["defects"]["summary"]["max_region_counts"].items():
        print(f"{k:15} {v}")

    print()
    print("Saved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()