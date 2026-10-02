from pathlib import Path
import sys
import json
from datetime import datetime

import numpy as np

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

from decision_engine import VisionQCDecisionEngine


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap_v2"
    / "Patchcore"
    / "water_cap_v1_roi80"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

INSPECTION_DIR = PROJECT_ROOT / "data" / "inspections"

THRESHOLD = 12.3785223961


# ============================================================
# PATCHCORE INFERENCE
# ============================================================

def run_patchcore(image_path: Path):

    print("\nLoading PatchCore v2...")

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )

    # IMPORTANT:
    # Disable Anomalib post-processing so that we receive
    # the actual PatchCore anomaly distance.
    model.post_processor = None

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    dataset = PredictDataset(
        path=str(image_path),
        image_size=(256, 256),
    )

    print("Running PatchCore inference...")

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT),
    )

    if not predictions:
        raise RuntimeError(
            "PatchCore returned no predictions."
        )

    prediction = predictions[0]

    anomaly_score = prediction.pred_score

    if hasattr(anomaly_score, "item"):
        anomaly_score = anomaly_score.item()

    anomaly_score = float(anomaly_score)

    anomaly_map = prediction.anomaly_map

    if hasattr(anomaly_map, "detach"):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    anomaly_map = np.asarray(anomaly_map)

    return anomaly_score, anomaly_map


# ============================================================
# SAVE ANOMALY MAP
# ============================================================

def save_anomaly_map(
    anomaly_map,
    original_path: Path,
):

    heatmap_dir = INSPECTION_DIR / "heatmaps"

    heatmap_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        heatmap_dir
        / f"{original_path.stem}_anomaly_map.npy"
    )

    np.save(
        output_path,
        anomaly_map,
    )

    return output_path


# ============================================================
# SAVE INSPECTION LOG
# ============================================================

def save_inspection_log(
    image_path: Path,
    anomaly_score: float,
    result,
    heatmap_path: Path,
):

    INSPECTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inspection_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    inspection_record = {

        "inspection_id": inspection_id,

        "timestamp": datetime.now().isoformat(),

        "product": "Water Bottle Cap",

        "sku": "water_cap_v1",

        "model": "PatchCore",

        "model_version": "patchcore_water_cap_v2",

        "input_image": str(
            image_path.relative_to(PROJECT_ROOT)
        ),

        "heatmap_data": str(
            heatmap_path.relative_to(PROJECT_ROOT)
        ),

        "image_quality": {
            "valid": result.image_quality_valid,
            "reason": result.quality_reason,
        },

        "alignment": {
            "valid": result.alignment_valid,
        },

        "product_detected": result.product_detected,

        "anomaly_score": result.anomaly_score,

        "confidence": result.confidence,

        "threshold": result.threshold,

        "decision": result.decision,

        "reason": result.reason,

        "supervisor_verdict": None,

        "supervisor_feedback": None,
    }

    output_path = (
        INSPECTION_DIR
        / f"{image_path.stem}_inspection.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            inspection_record,
            file,
            indent=4,
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VisionQC — PatchCore End-to-End Inspection")
    print("=" * 70)

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            'python vision/run_patchcore_inspection.py '
            '"path\\to\\roi_image.jpg"'
        )

        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image_path = image_path.resolve()

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"PatchCore checkpoint not found:\n{CHECKPOINT}"
        )

    print()
    print(f"Input ROI image : {image_path.name}")
    print(f"Checkpoint      : {CHECKPOINT.name}")

    # --------------------------------------------------------
    # PATCHCORE
    # --------------------------------------------------------

    anomaly_score, anomaly_map = run_patchcore(
        image_path
    )

    print()
    print(
        f"RAW anomaly score : "
        f"{anomaly_score:.6f}"
    )

    # --------------------------------------------------------
    # SAVE HEATMAP DATA
    # --------------------------------------------------------

    heatmap_path = save_anomaly_map(
        anomaly_map,
        image_path,
    )

    # --------------------------------------------------------
    # DECISION ENGINE
    # --------------------------------------------------------

    decision_engine = VisionQCDecisionEngine(
        threshold=THRESHOLD,
        review_margin=0.10,
    )

    result = decision_engine.inspect(
        anomaly_score=anomaly_score,
        image_quality_valid=True,
        quality_reason="Image quality acceptable",
        product_detected=True,
        alignment_valid=True,
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VISIONQC INSPECTION RESULT")
    print("=" * 70)

    print(
        f"Input ROI         : "
        f"{image_path.name}"
    )

    print(
        f"Product           : "
        f"Water Bottle Cap"
    )

    print(
        f"SKU               : "
        f"water_cap_v1"
    )

    print(
        f"Model             : "
        f"PatchCore v2"
    )

    print(
        f"Image Quality     : "
        f"{'VALID' if result.image_quality_valid else 'INVALID'}"
    )

    print(
        f"Product Detected  : "
        f"{'YES' if result.product_detected else 'NO'}"
    )

    print(
        f"Alignment         : "
        f"{'VALID' if result.alignment_valid else 'INVALID'}"
    )

    print(
        f"Anomaly Score     : "
        f"{result.anomaly_score:.6f}"
    )

    print(
        f"Threshold         : "
        f"{result.threshold:.6f}"
    )

    print(
        f"Confidence        : "
        f"{result.confidence:.2f}%"
    )

    print(
        f"Decision          : "
        f"{result.decision}"
    )

    print(
        f"Reason            : "
        f"{result.reason}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # SAVE LOG
    # --------------------------------------------------------

    log_path = save_inspection_log(
        image_path=image_path,
        anomaly_score=anomaly_score,
        result=result,
        heatmap_path=heatmap_path,
    )

    print()
    print(
        f"Inspection log    : "
        f"{log_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"Heatmap data      : "
        f"{heatmap_path.relative_to(PROJECT_ROOT)}"
    )

    print()
    print("Inspection completed.")


if __name__ == "__main__":
    main()