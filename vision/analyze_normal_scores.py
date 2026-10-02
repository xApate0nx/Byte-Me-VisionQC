from pathlib import Path
import json
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT = PROJECT_ROOT / "data" / "evaluation" / "patchcore_auto_evaluation.json"


def main():
    print("=" * 70)
    print("VisionQC — Normal Score Distribution Analysis")
    print("=" * 70)

    if not REPORT.exists():
        print("Evaluation report not found:")
        print(REPORT)
        return

    with open(REPORT, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either a list directly or a dictionary containing results.
    if isinstance(data, dict):
        results = data.get("results", data.get("evaluations", []))
    else:
        results = data

    normals = []

    for item in results:
        label = str(item.get("label", item.get("actual", ""))).lower()

        if label in {"normal", "good", "0"}:
            normals.append(item)

    if not normals:
        print("Could not find normal samples in JSON.")
        print("JSON keys:", data.keys() if isinstance(data, dict) else "list")
        return

    scores = []

    for item in normals:
        score = float(
            item.get(
                "score",
                item.get(
                    "anomaly_score",
                    item.get("pred_score", 0)
                )
            )
        )

        name = item.get("filename", item.get("image", "unknown"))

        scores.append((score, name))

    scores.sort()

    values = np.array([x[0] for x in scores])

    print(f"\nNormal images found: {len(values)}")

    print("\nSorted normal scores:")
    print("-" * 70)

    for i, (score, name) in enumerate(scores, 1):
        print(f"{i:02d}. {score:10.6f}   {name}")

    print("\nStatistics")
    print("-" * 70)

    print(f"Minimum       : {values.min():.6f}")
    print(f"Maximum       : {values.max():.6f}")
    print(f"Mean          : {values.mean():.6f}")
    print(f"Median        : {np.median(values):.6f}")
    print(f"Std deviation : {values.std():.6f}")

    print("\nPercentiles")
    print("-" * 70)

    for p in [50, 75, 90, 95, 100]:
        print(f"{p:3d}th percentile : {np.percentile(values, p):.6f}")

    print("\nPotential high-score normal images")
    print("-" * 70)

    mean = values.mean()
    std = values.std()

    for score, name in scores:
        z = (score - mean) / std if std > 0 else 0

        if z > 2:
            print(f"{score:.6f}   z={z:.2f}   {name}")

    print("\nThreshold comparison")
    print("-" * 70)

    threshold = 12.3785223961

    print(f"Current threshold : {threshold:.6f}")
    print(f"Normals below     : {(values < threshold).sum()}")
    print(f"Normals above     : {(values >= threshold).sum()}")


if __name__ == "__main__":
    main()