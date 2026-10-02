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

SCORE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_v3"
)


def load_scores(filename):
    path = SCORE_ROOT / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}

    for item in data["results"]:
        name = Path(item["filename"]).stem
        results[name] = float(item["score"])

    return results


def spatial_signal(path):
    heatmap = np.load(path)

    h, w = heatmap.shape

    # Top 5% most anomalous pixels
    p95 = np.percentile(heatmap, 95)
    high = heatmap >= p95

    ys, xs = np.where(high)

    if len(xs) == 0:
        return 0.0

    cx = w / 2
    cy = h / 2

    # Central 60% of the ROI
    central = (
        (np.abs(xs - cx) <= 0.30 * w)
        & (np.abs(ys - cy) <= 0.30 * h)
    )

    return float(np.mean(central))


def collect(category):
    directory = HEATMAP_ROOT / category

    results = {}

    for path in sorted(directory.glob("*.npy")):
        name = path.stem.replace("_anomaly_map", "")
        results[name] = spatial_signal(path)

    return results


def evaluate(
    normal_scores,
    defect_scores,
    normal_spatial,
    defect_spatial,
):

    print()
    print("=" * 70)
    print("PATCHCORE V3 + SPATIAL SIGNAL ANALYSIS")
    print("=" * 70)

    print()
    print("NORMAL")
    print("-" * 70)

    for name in sorted(normal_scores):

        spatial = normal_spatial.get(name)

        if spatial is not None:
            print(
                f"{name[:45]:45} "
                f"score={normal_scores[name]:7.3f} "
                f"central={spatial:.3f}"
            )

    print()
    print("DEFECTS")
    print("-" * 70)

    for name in sorted(defect_scores):

        spatial = defect_spatial.get(name)

        if spatial is not None:
            print(
                f"{name[:45]:45} "
                f"score={defect_scores[name]:7.3f} "
                f"central={spatial:.3f}"
            )

    normal_values = list(normal_spatial.values())
    defect_values = list(defect_spatial.values())

    print()
    print("=" * 70)
    print("SPATIAL SUMMARY")
    print("=" * 70)

    print()
    print(
        "Normal mean central fraction :",
        f"{np.mean(normal_values):.4f}"
    )

    print(
        "Defect mean central fraction :",
        f"{np.mean(defect_values):.4f}"
    )

    print(
        "Normal median                :",
        f"{np.median(normal_values):.4f}"
    )

    print(
        "Defect median                :",
        f"{np.median(defect_values):.4f}"
    )

    print()
    print("=" * 70)
    print("SPATIAL THRESHOLD TEST")
    print("=" * 70)

    print()
    print(
        f"{'Threshold':<12}"
        f"{'Normal flagged':<18}"
        f"{'Defects detected':<20}"
    )

    for threshold in np.arange(0.30, 0.81, 0.05):

        normal_flagged = sum(
            x >= threshold for x in normal_values
        )

        defect_detected = sum(
            x >= threshold for x in defect_values
        )

        print(
            f"{threshold:<12.2f}"
            f"{normal_flagged}/{len(normal_values):<17}"
            f"{defect_detected}/{len(defect_values):<20}"
        )


def main():

    normal_scores = load_scores("normal_scores.json")
    defect_scores = load_scores("defect_scores.json")

    normal_spatial = collect("normal")
    defect_spatial = collect("defects")

    print(
        f"Loaded normal scores : {len(normal_scores)}"
    )

    print(
        f"Loaded defect scores : {len(defect_scores)}"
    )

    print(
        f"Normal heatmaps      : {len(normal_spatial)}"
    )

    print(
        f"Defect heatmaps      : {len(defect_spatial)}"
    )

    evaluate(
        normal_scores,
        defect_scores,
        normal_spatial,
        defect_spatial,
    )


if __name__ == "__main__":
    main()