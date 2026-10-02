from pathlib import Path
import json

from backend.app.ml.inference_adapter import inspect_prepared_roi


PROJECT_ROOT = Path(__file__).resolve().parent

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi"
    / "normal"
)

DEFECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi"
    / "defects"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_prepared_roi"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def evaluate_category(directory, category):
    results = []

    paths = sorted(directory.glob("*_roi.png"))

    print()
    print("=" * 60)
    print(category.upper())
    print("=" * 60)

    for index, path in enumerate(paths, start=1):
        print(
            f"[{index:02d}/{len(paths):02d}] "
            f"{path.name}",
            end=" ... ",
        )

        try:
            result = inspect_prepared_roi(path)

            score = result["anomaly_score"]
            decision = result["decision"]

            print(
                f"score={score:.6f} "
                f"decision={decision}"
            )

            results.append(
                {
                    "category": category,
                    "filename": path.name,
                    "anomaly_score": score,
                    "decision": decision,
                    "status": "OK",
                }
            )

        except Exception as exc:
            print(f"ERROR: {exc}")

            results.append(
                {
                    "category": category,
                    "filename": path.name,
                    "status": "ERROR",
                    "error": str(exc),
                }
            )

    return results


def main():
    normal_results = evaluate_category(
        NORMAL_DIR,
        "normal",
    )

    defect_results = evaluate_category(
        DEFECT_DIR,
        "defect",
    )

    results = normal_results + defect_results

    output_file = OUTPUT_DIR / "prepared_roi_results.json"

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    valid_scores = [
        item["anomaly_score"]
        for item in results
        if item.get("status") == "OK"
    ]

    normal_scores = [
        item["anomaly_score"]
        for item in results
        if (
            item.get("status") == "OK"
            and item.get("category") == "normal"
        )
    ]

    defect_scores = [
        item["anomaly_score"]
        for item in results
        if (
            item.get("status") == "OK"
            and item.get("category") == "defect"
        )
    ]

    print()
    print("=" * 60)
    print("PREPARED ROI PATCHCORE EVALUATION")
    print("=" * 60)

    print(f"Normal images: {len(normal_scores)}")
    print(f"Defect images: {len(defect_scores)}")
    print(f"Successful: {len(valid_scores)}")
    print(f"Results: {output_file}")

    if normal_scores:
        print()
        print("NORMAL SCORES")
        print(
            f"Min:  {min(normal_scores):.6f}"
        )
        print(
            f"Max:  {max(normal_scores):.6f}"
        )
        print(
            f"Mean: {sum(normal_scores) / len(normal_scores):.6f}"
        )

    if defect_scores:
        print()
        print("DEFECT SCORES")
        print(
            f"Min:  {min(defect_scores):.6f}"
        )
        print(
            f"Max:  {max(defect_scores):.6f}"
        )
        print(
            f"Mean: {sum(defect_scores) / len(defect_scores):.6f}"
        )


if __name__ == "__main__":
    main()