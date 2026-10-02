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
    with open(SCORE_ROOT / filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        Path(item["filename"]).stem: float(item["score"])
        for item in data["results"]
    }


def spatial_signal(path):
    heatmap = np.load(path)

    h, w = heatmap.shape

    p95 = np.percentile(heatmap, 95)
    high = heatmap >= p95

    ys, xs = np.where(high)

    if len(xs) == 0:
        return 0.0

    cx = w / 2
    cy = h / 2

    central = (
        (np.abs(xs - cx) <= 0.30 * w)
        & (np.abs(ys - cy) <= 0.30 * h)
    )

    return float(np.mean(central))


def load_spatial(category):
    directory = HEATMAP_ROOT / category

    return {
        path.stem.replace("_anomaly_map", ""): spatial_signal(path)
        for path in sorted(directory.glob("*.npy"))
    }


def evaluate_rule(
    normal_scores,
    defect_scores,
    normal_spatial,
    defect_spatial,
    score_threshold,
    spatial_threshold,
    mode,
):
    def flagged(score, spatial):
        score_flag = score >= score_threshold
        spatial_flag = spatial >= spatial_threshold

        if mode == "OR":
            return score_flag or spatial_flag

        return score_flag and spatial_flag

    normal_flags = [
        flagged(score, normal_spatial[name])
        for name, score in normal_scores.items()
        if name in normal_spatial
    ]

    defect_flags = [
        flagged(score, defect_spatial[name])
        for name, score in defect_scores.items()
        if name in defect_spatial
    ]

    normal_fail = sum(normal_flags)
    defect_detect = sum(defect_flags)

    false_reject = normal_fail / len(normal_flags)
    miss_rate = 1 - (defect_detect / len(defect_flags))

    # Overall accuracy
    correct = (
        (len(normal_flags) - normal_fail)
        + defect_detect
    )

    accuracy = correct / (
        len(normal_flags) + len(defect_flags)
    )

    return {
        "score_threshold": score_threshold,
        "spatial_threshold": spatial_threshold,
        "mode": mode,
        "normal_fail": normal_fail,
        "normal_total": len(normal_flags),
        "defect_detect": defect_detect,
        "defect_total": len(defect_flags),
        "false_reject": false_reject,
        "miss_rate": miss_rate,
        "accuracy": accuracy,
    }


def main():
    normal_scores = load_scores("normal_scores.json")
    defect_scores = load_scores("defect_scores.json")

    normal_spatial = load_spatial("normal")
    defect_spatial = load_spatial("defects")

    print()
    print("=" * 80)
    print("PATCHCORE V3 + SPATIAL SIGNAL — COMBINED ANALYSIS")
    print("=" * 80)

    print()
    print("Loaded:")
    print("Normal :", len(normal_scores))
    print("Defects:", len(defect_scores))

    results = []

    # Score thresholds based on the observed v3 score range.
    score_thresholds = np.arange(11.0, 18.01, 0.25)

    # Spatial thresholds.
    spatial_thresholds = np.arange(0.30, 0.81, 0.05)

    for mode in ["OR", "AND"]:

        for score_threshold in score_thresholds:

            for spatial_threshold in spatial_thresholds:

                result = evaluate_rule(
                    normal_scores,
                    defect_scores,
                    normal_spatial,
                    defect_spatial,
                    float(score_threshold),
                    float(spatial_threshold),
                    mode,
                )

                results.append(result)

    # ------------------------------------------------------------------
    # Find configurations with ZERO known normal false rejects.
    # Among those, show the highest defect detection.
    # ------------------------------------------------------------------

    zero_false_reject = [
        r for r in results
        if r["normal_fail"] == 0
    ]

    zero_false_reject.sort(
        key=lambda r: (
            -r["defect_detect"],
            r["miss_rate"],
            r["score_threshold"],
            r["spatial_threshold"],
        )
    )

    print()
    print("=" * 80)
    print("BEST CONFIGURATIONS WITH ZERO KNOWN NORMAL FALSE REJECTS")
    print("=" * 80)

    print()

    shown = 0

    for r in zero_false_reject:

        print(
            f"{r['mode']:3} | "
            f"score >= {r['score_threshold']:5.2f} | "
            f"spatial >= {r['spatial_threshold']:.2f} | "
            f"normal {r['normal_fail']}/{r['normal_total']} | "
            f"defects {r['defect_detect']}/{r['defect_total']} | "
            f"accuracy {r['accuracy']:.3f}"
        )

        shown += 1

        if shown >= 20:
            break

    # ------------------------------------------------------------------
    # Show best configurations by accuracy.
    # ------------------------------------------------------------------

    best_accuracy = sorted(
        results,
        key=lambda r: (
            -r["accuracy"],
            r["false_reject"],
            r["miss_rate"],
        )
    )

    print()
    print("=" * 80)
    print("HIGHEST ACCURACY CONFIGURATIONS")
    print("=" * 80)

    print()

    for r in best_accuracy[:20]:

        print(
            f"{r['mode']:3} | "
            f"score >= {r['score_threshold']:5.2f} | "
            f"spatial >= {r['spatial_threshold']:.2f} | "
            f"normal {r['normal_fail']}/{r['normal_total']} | "
            f"defects {r['defect_detect']}/{r['defect_total']} | "
            f"false reject {r['false_reject']:.3f} | "
            f"miss {r['miss_rate']:.3f} | "
            f"accuracy {r['accuracy']:.3f}"
        )

    # ------------------------------------------------------------------
    # Save complete experiment.
    # ------------------------------------------------------------------

    output = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "patchcore_v3_combined_signal.json"
    )

    output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("Saved:")
    print(output)
    print("=" * 80)


if __name__ == "__main__":
    main()