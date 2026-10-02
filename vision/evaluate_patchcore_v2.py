from pathlib import Path
import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_candidates"
    / "large"
    / "normal"
)

DEFECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "roi_candidates"
    / "large"
    / "defects"
)

MODEL_DIR = PROJECT_ROOT / "models" / "patchcore_water_cap_v2"


def main():
    print("=" * 70)
    print("VisionQC — PatchCore v2 Evaluation")
    print("=" * 70)

    print("\nNormal images:", len(list(NORMAL_DIR.glob("*"))))
    print("Defect images:", len(list(DEFECT_DIR.glob("*"))))

    # Evaluation dataset.
    # Anomalib will create the same type of held-out normal split
    # that we used for the original PatchCore experiment.
    datamodule = Folder(
        name="water_cap_v1_roi80_evaluation",
        root=str(PROJECT_ROOT / "data" / "evaluation" / "roi_candidates" / "large"),
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

    # Locate checkpoint.
    checkpoints = list(MODEL_DIR.rglob("*.ckpt"))

    if not checkpoints:
        raise RuntimeError(
            f"No PatchCore v2 checkpoint found under {MODEL_DIR}"
        )

    checkpoint = checkpoints[0]

    print("\nCheckpoint:")
    print(checkpoint)

    print("\nRunning predictions...\n")

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

        scores = batch.pred_score.detach().cpu().flatten().tolist()

        labels = None
        if hasattr(batch, "gt_label") and batch.gt_label is not None:
            labels = (
                batch.gt_label.detach()
                .cpu()
                .flatten()
                .tolist()
            )

        for i, path in enumerate(paths):
            score = float(scores[i])

            if labels is not None:
                label = int(labels[i])
            else:
                label = -1

            results.append(
                {
                    "path": str(path),
                    "score": score,
                    "label": label,
                }
            )

    print("=" * 70)
    print("INDIVIDUAL RESULTS")
    print("=" * 70)

    for r in results:
        filename = Path(r["path"]).name

        label_name = (
            "DEFECT"
            if r["label"] == 1
            else "NORMAL"
            if r["label"] == 0
            else "UNKNOWN"
        )

        print(
            f"{label_name:7s} | "
            f"{r['score']:.6f} | "
            f"{filename}"
        )

    normal = [
        r["score"]
        for r in results
        if r["label"] == 0
    ]

    defects = [
        r["score"]
        for r in results
        if r["label"] == 1
    ]

    print("\n" + "=" * 70)
    print("SCORE SUMMARY")
    print("=" * 70)

    if normal:
        print("\nNORMAL")
        print(f"Count : {len(normal)}")
        print(f"Min   : {min(normal):.6f}")
        print(f"Max   : {max(normal):.6f}")
        print(f"Mean  : {sum(normal) / len(normal):.6f}")

    if defects:
        print("\nDEFECT")
        print(f"Count : {len(defects)}")
        print(f"Min   : {min(defects):.6f}")
        print(f"Max   : {max(defects):.6f}")
        print(f"Mean  : {sum(defects) / len(defects):.6f}")

    print("\n" + "=" * 70)
    print("TOP SCORES")
    print("=" * 70)

    for r in sorted(
        results,
        key=lambda x: x["score"],
        reverse=True,
    )[:15]:

        filename = Path(r["path"]).name

        label_name = (
            "DEFECT"
            if r["label"] == 1
            else "NORMAL"
        )

        print(
            f"{r['score']:.6f} | "
            f"{label_name:7s} | "
            f"{filename}"
        )


if __name__ == "__main__":
    main()