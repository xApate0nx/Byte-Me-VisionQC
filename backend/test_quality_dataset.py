from pathlib import Path

from backend.app.services.image_quality_service import (
    check_image_quality,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NORMAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
    / "normal"
)

DEFECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "products"
    / "water_cap_v1"
    / "defects"
)


def get_images(directory: Path):
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        ]
    )


def analyze_dataset(
    name: str,
    directory: Path,
):
    images = get_images(directory)

    print("=" * 70)
    print(f"{name.upper()} DATASET")
    print("=" * 70)
    print(f"Images found: {len(images)}")
    print()

    results = []

    for image_path in images:

        result = check_image_quality(
            image_path
        )

        checks = result["checks"]

        blur = checks["blur"]["score"]

        brightness = checks[
            "exposure"
        ]["mean_brightness"]

        occupancy = checks[
            "product_visibility"
        ]["occupancy"]

        print(image_path.name)
        print(
            f"  status:     {result['status']}"
        )
        print(
            f"  blur:       {blur:.2f}"
        )
        print(
            f"  brightness: {brightness:.2f}"
        )
        print(
            f"  occupancy:  {occupancy:.3f}"
        )

        if result["status"] == "INVALID":
            print(
                f"  reason:     {result['reason']}"
            )

        print()

        results.append(
            {
                "name": image_path.name,
                "status": result["status"],
                "blur": blur,
                "brightness": brightness,
                "occupancy": occupancy,
            }
        )

    valid_count = sum(
        result["status"] == "VALID"
        for result in results
    )

    invalid_count = (
        len(results) - valid_count
    )

    print("-" * 70)
    print(f"{name} SUMMARY")
    print("-" * 70)
    print(f"VALID   : {valid_count}")
    print(f"INVALID : {invalid_count}")

    if results:

        blurs = [
            result["blur"]
            for result in results
        ]

        brightness_values = [
            result["brightness"]
            for result in results
        ]

        occupancies = [
            result["occupancy"]
            for result in results
        ]

        print(
            f"Blur    : min={min(blurs):.2f}, "
            f"max={max(blurs):.2f}, "
            f"mean={sum(blurs) / len(blurs):.2f}"
        )

        print(
            f"Bright  : min={min(brightness_values):.2f}, "
            f"max={max(brightness_values):.2f}, "
            f"mean={sum(brightness_values) / len(brightness_values):.2f}"
        )

        print(
            f"Occup.  : min={min(occupancies):.3f}, "
            f"max={max(occupancies):.3f}, "
            f"mean={sum(occupancies) / len(occupancies):.3f}"
        )

    print()

    return results


def main():

    print("=" * 70)
    print("VisionQC Image Quality Dataset Analysis")
    print("=" * 70)
    print()

    normal_results = analyze_dataset(
        "Normal",
        NORMAL_DIR,
    )

    defect_results = analyze_dataset(
        "Defect",
        DEFECT_DIR,
    )

    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    normal_valid = sum(
        result["status"] == "VALID"
        for result in normal_results
    )

    defect_valid = sum(
        result["status"] == "VALID"
        for result in defect_results
    )

    print(
        f"Normal : {normal_valid}/{len(normal_results)} "
        "quality-valid"
    )

    print(
        f"Defect : {defect_valid}/{len(defect_results)} "
        "quality-valid"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()