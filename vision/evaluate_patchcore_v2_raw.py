from pathlib import Path
import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_candidates"
    / "large"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap_v2"
)


def main():
    print("=" * 70)
    print("VisionQC — PatchCore v2 RAW SCORE EVALUATION")
    print("=" * 70)

    datamodule = Folder(
        name="water_cap_v1_roi80_raw",
        root=str(DATA_DIR),
        normal_dir="normal",
        abnormal_dir="defects",
        normal_split_ratio=0.0,
        test_split_mode="from_dir",
        test_split_ratio=0.0,
        val_split_mode="none",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=str(MODEL_DIR),
        accelerator="cpu",
        devices=1,
    )

    checkpoints = list(MODEL_DIR.rglob("*.ckpt"))

    if not checkpoints:
        raise RuntimeError("No PatchCore v2 checkpoint found.")

    checkpoint = checkpoints[0]

    print("\nCheckpoint:")
    print(checkpoint)

    # Disable the post-processing normalization that turns
    # anomaly scores into 0/1 predictions.
    model.post_processor = None

    print("\nRunning RAW predictions...\n")

    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(checkpoint),
    )

    results = []

    for batch in predictions:
        paths = batch.image_path

        if isinstance(paths, str):
            paths = [paths]

        # Try to retrieve the actual anomaly score tensor.
        if hasattr(batch, "pred_score"):
            scores = (
                batch.pred_score
                .detach()
                .cpu()
                .flatten()
                .tolist()
            )
        else:
            raise RuntimeError(
                "Prediction batch does not contain pred_score."
            )

        labels = (
            batch.gt_label
            .detach()
            .cpu()
            .flatten()
            .tolist()
            if hasattr(batch, "gt_label")
            and batch.gt_label is not None
            else [-1] * len(paths)
        )

        for i, path in enumerate(paths):
            results.append(
                (
                    Path(path).name,
                    float(scores[i]),
                    int(labels[i]),
                )
            )

    print("=" * 70)
    print("RAW RESULTS")
    print("=" * 70)

    for name, score, label in results:
        label_name = (
            "DEFECT"
            if label == 1
            else "NORMAL"
        )

        print(
            f"{label_name:7s} | "
            f"{score:.10f} | "
            f"{name}"
        )

    normal = [
        score
        for _, score, label in results
        if label == 0
    ]

    defects = [
        score
        for _, score, label in results
        if label == 1
    ]

    print("\n" + "=" * 70)
    print("RAW SCORE DISTRIBUTIONS")
    print("=" * 70)

    if normal:
        print("\nNORMAL")
        print(f"Count : {len(normal)}")
        print(f"Min   : {min(normal):.10f}")
        print(f"Max   : {max(normal):.10f}")
        print(f"Mean  : {sum(normal)/len(normal):.10f}")

    if defects:
        print("\nDEFECT")
        print(f"Count : {len(defects)}")
        print(f"Min   : {min(defects):.10f}")
        print(f"Max   : {max(defects):.10f}")
        print(f"Mean  : {sum(defects)/len(defects):.10f}")

    print("\n" + "=" * 70)
    print("SORTED BY RAW SCORE")
    print("=" * 70)

    for name, score, label in sorted(
        results,
        key=lambda x: x[1],
        reverse=True,
    ):
        label_name = (
            "DEFECT"
            if label == 1
            else "NORMAL"
        )

        print(
            f"{score:.10f} | "
            f"{label_name:7s} | "
            f"{name}"
        )


if __name__ == "__main__":
    main()