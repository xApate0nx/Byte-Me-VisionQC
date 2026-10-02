from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap_v3"
    / "Patchcore"
    / "water_cap_v1_canonical_roi"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "new_roi"
)

DEFECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_comparison"
    / "defects_new_roi"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patchcore_v3_heatmaps"
)

MODEL_NAME = "PatchCore v3"
MODEL_VERSION = "patchcore_water_cap_v3"


def load_model():
    print("Loading PatchCore v3...")

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        num_neighbors=9,
    )

    # We want the raw PatchCore anomaly distance.
    model.post_processor = None

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    return model, engine


def run_patchcore(model, engine, image_path):
    dataset = PredictDataset(
        path=str(image_path),
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT),
    )

    if not predictions:
        raise RuntimeError("PatchCore returned no predictions.")

    prediction = predictions[0]

    score = prediction.pred_score

    if hasattr(score, "item"):
        score = score.item()

    anomaly_map = prediction.anomaly_map

    if hasattr(anomaly_map, "detach"):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

    # Remove unnecessary dimensions.
    anomaly_map = np.squeeze(anomaly_map)

    return float(score), anomaly_map


def save_heatmap(anomaly_map, output_path):
    normalized = cv2.normalize(
        anomaly_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    normalized = normalized.astype(np.uint8)

    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )

    cv2.imwrite(
        str(output_path),
        heatmap,
    )


def analyze_map(anomaly_map):
    h, w = anomaly_map.shape[:2]

    max_y, max_x = np.unravel_index(
        np.argmax(anomaly_map),
        anomaly_map.shape,
    )

    threshold_95 = np.percentile(anomaly_map, 95)
    hot_mask = anomaly_map >= threshold_95

    ys, xs = np.where(hot_mask)

    if len(xs) > 0:
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        hot_center_x = float((x_min + x_max) / 2)
        hot_center_y = float((y_min + y_max) / 2)
    else:
        x_min = x_max = int(max_x)
        y_min = y_max = int(max_y)

        hot_center_x = float(max_x)
        hot_center_y = float(max_y)

    return {
        "map_width": int(w),
        "map_height": int(h),
        "min": float(anomaly_map.min()),
        "max": float(anomaly_map.max()),
        "mean": float(anomaly_map.mean()),
        "median": float(np.median(anomaly_map)),
        "p90": float(np.percentile(anomaly_map, 90)),
        "p95": float(np.percentile(anomaly_map, 95)),
        "p99": float(np.percentile(anomaly_map, 99)),
        "max_x": int(max_x),
        "max_y": int(max_y),
        "hot_pixels_95": int(len(xs)),
        "hot_percent_95": float(len(xs) / (w * h) * 100),
        "hot_bbox": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "hot_center": {
            "x": hot_center_x,
            "y": hot_center_y,
        },
    }


def process_category(
    category,
    input_dir,
    model,
    engine,
):
    output_dir = OUTPUT_ROOT / category
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob("*.png"))

    print("\n" + "=" * 80)
    print(category.upper())
    print("=" * 80)

    print(f"Images: {len(image_files)}")

    results = []

    for index, image_path in enumerate(image_files, start=1):
        print(
            f"\n[{index}/{len(image_files)}] "
            f"{image_path.name}"
        )

        try:
            score, anomaly_map = run_patchcore(
                model,
                engine,
                image_path,
            )

            print(f"  score : {score:.6f}")

            map_stats = analyze_map(anomaly_map)

            print(
                f"  map   : "
                f"min={map_stats['min']:.4f}, "
                f"max={map_stats['max']:.4f}, "
                f"mean={map_stats['mean']:.4f}"
            )

            print(
                f"  max   : "
                f"({map_stats['max_x']}, "
                f"{map_stats['max_y']})"
            )

            print(
                f"  hot95 : "
                f"{map_stats['hot_percent_95']:.2f}%"
            )

            stem = image_path.stem

            npy_path = output_dir / f"{stem}_anomaly_map.npy"
            heatmap_path = output_dir / f"{stem}_heatmap.png"
            json_path = output_dir / f"{stem}_analysis.json"

            np.save(
                npy_path,
                anomaly_map,
            )

            save_heatmap(
                anomaly_map,
                heatmap_path,
            )

            result = {
                "category": category,
                "filename": image_path.name,
                "model": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "checkpoint": str(CHECKPOINT),
                "score": score,
                "anomaly_map": str(npy_path),
                "heatmap": str(heatmap_path),
                **map_stats,
            }

            with open(
                json_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    result,
                    f,
                    indent=2,
                )

            results.append(result)

        except Exception as exc:
            print(f"  ERROR: {exc}")

    summary_path = (
        OUTPUT_ROOT
        / f"{category}_results.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    return results


def main():
    print("=" * 80)
    print("PATCHCORE V3 — HEATMAP VALIDATION")
    print("=" * 80)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, engine = load_model()

    normal_results = process_category(
        "normal",
        NORMAL_DIR,
        model,
        engine,
    )

    defect_results = process_category(
        "defects",
        DEFECT_DIR,
        model,
        engine,
    )

    summary = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "normal_count": len(normal_results),
        "defect_count": len(defect_results),
        "output_root": str(OUTPUT_ROOT),
    }

    with open(
        OUTPUT_ROOT / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("PATCHCORE V3 HEATMAP GENERATION COMPLETE")
    print("=" * 80)

    print(f"Normal heatmaps : {len(normal_results)}")
    print(f"Defect heatmaps : {len(defect_results)}")
    print(f"Output          : {OUTPUT_ROOT}")

    print("\nNext step:")
    print("Open the generated normal and defect heatmaps.")
    print("We need to verify whether high-anomaly regions correspond")
    print("to the actual product/defect rather than ROI boundaries.")
    print("=" * 80)


if __name__ == "__main__":
    main()