from pathlib import Path
import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "patchcore_water_cap"
    / "Patchcore"
    / "water_cap_v1"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)


def main():

    print("=" * 70)
    print("VisionQC — PatchCore Internal Diagnostic")
    print("=" * 70)

    datamodule = Folder(
        name="water_cap_evaluation",
        root=str(DATA_ROOT),
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

    datamodule.setup()

    print("\nTrain images:", len(datamodule.train_data))
    print("Test images :", len(datamodule.test_data))

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    print("\nLoading checkpoint...")
    
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT),
    )

    print("\nPrediction batches:", len(predictions))

    print("\n" + "=" * 70)
    print("PATCHCORE MODEL INFORMATION")
    print("=" * 70)

    print("\nModel:")
    print(model)

    print("\n" + "=" * 70)
    print("MEMORY BANK INFORMATION")
    print("=" * 70)

    # Search for PatchCore memory bank related attributes
    for name in dir(model):

        if any(
            keyword in name.lower()
            for keyword in [
                "memory",
                "coreset",
                "neighbor",
                "sampler",
            ]
        ):

            try:

                value = getattr(model, name)

                print(
                    f"\n{name}: "
                    f"type={type(value)}"
                )

                if torch.is_tensor(value):

                    print(
                        f"shape={tuple(value.shape)} "
                        f"min={value.min().item():.6f} "
                        f"max={value.max().item():.6f}"
                    )

                elif hasattr(value, "shape"):

                    print(
                        f"shape={value.shape}"
                    )

            except Exception as e:

                print(
                    f"\n{name}: "
                    f"<could not inspect: {e}>"
                )

    print("\n" + "=" * 70)
    print("PREDICTION SCORE + MAP DIAGNOSTIC")
    print("=" * 70)

    sample_number = 0

    for batch_index, prediction in enumerate(predictions):

        paths = prediction.image_path
        scores = prediction.pred_score
        maps = prediction.anomaly_map

        if torch.is_tensor(scores):

            scores_np = scores.detach().cpu().numpy()

        else:

            scores_np = scores

        if torch.is_tensor(maps):

            maps_np = maps.detach().cpu().numpy()

        else:

            maps_np = maps

        for i, path in enumerate(paths):

            score = float(scores_np[i])

            print(
                f"\n[{sample_number:02d}] "
                f"{Path(path).parent.name}"
            )

            print(
                f"File: {Path(path).name}"
            )

            print(
                f"Score: {score:.8f}"
            )

            if maps_np is not None:

                current_map = maps_np[i]

                print(
                    f"Map shape: "
                    f"{current_map.shape}"
                )

                print(
                    f"Map min: "
                    f"{current_map.min():.8f}"
                )

                print(
                    f"Map max: "
                    f"{current_map.max():.8f}"
                )

                print(
                    f"Map mean: "
                    f"{current_map.mean():.8f}"
                )

                print(
                    f"Map std: "
                    f"{current_map.std():.8f}"
                )

            sample_number += 1

            # Only inspect first 10 images in detail
            if sample_number >= 10:
                break

        if sample_number >= 10:
            break

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()