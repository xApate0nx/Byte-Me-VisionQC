from pathlib import Path
import cv2
import numpy as np
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NORMAL_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "normal"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "normalized_contact_sheet"

IMAGE_SIZE = 400
COLUMNS = 4
PADDING = 20


def load_and_normalize(path):
    image = cv2.imread(str(path))

    if image is None:
        print(f"[SKIP] Could not read: {path.name}")
        return None

    original_h, original_w = image.shape[:2]

    # Convert portrait images to landscape.
    if original_h > original_w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Resize while preserving aspect ratio.
    h, w = image.shape[:2]
    scale = IMAGE_SIZE / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Place the resized image in a fixed square canvas.
    canvas = np.ones(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    ) * 230

    x = (IMAGE_SIZE - new_w) // 2
    y = (IMAGE_SIZE - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def add_label(image, label):
    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (IMAGE_SIZE - 1, IMAGE_SIZE - 1),
        (80, 80, 80),
        2
    )

    cv2.rectangle(
        output,
        (0, IMAGE_SIZE - 38),
        (IMAGE_SIZE, IMAGE_SIZE),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        output,
        label,
        (8, IMAGE_SIZE - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return output


def main():
    print("=" * 70)
    print("VisionQC — Normalized Dataset Contact Sheet")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(NORMAL_DIR.glob("*.jpg")) +
        list(NORMAL_DIR.glob("*.jpeg")) +
        list(NORMAL_DIR.glob("*.png"))
    )

    print(f"\nNormal images found: {len(files)}")

    if not files:
        raise RuntimeError("No normal images found.")

    normalized_images = []

    for index, path in enumerate(files):
        image = load_and_normalize(path)

        if image is None:
            continue

        label = f"{index + 1:02d}  {path.name}"

        image = add_label(image, label)

        normalized_images.append(image)

    rows = math.ceil(len(normalized_images) / COLUMNS)

    sheet_width = (
        COLUMNS * IMAGE_SIZE +
        (COLUMNS + 1) * PADDING
    )

    sheet_height = (
        rows * IMAGE_SIZE +
        (rows + 1) * PADDING
    )

    sheet = np.ones(
        (sheet_height, sheet_width, 3),
        dtype=np.uint8
    ) * 245

    for i, image in enumerate(normalized_images):

        row = i // COLUMNS
        col = i % COLUMNS

        x = PADDING + col * (IMAGE_SIZE + PADDING)
        y = PADDING + row * (IMAGE_SIZE + PADDING)

        sheet[
            y:y + IMAGE_SIZE,
            x:x + IMAGE_SIZE
        ] = image

    output_path = OUTPUT_DIR / "normal_contact_sheet.jpg"

    cv2.imwrite(
        str(output_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    print("\nGenerated:")
    print(f"  {output_path}")

    print("\nNormalization:")
    print("  • Portrait images rotated to landscape")
    print("  • Aspect ratio preserved")
    print("  • Images centered on 400x400 canvas")
    print("  • Original images NOT modified")
    print("  • Dataset filenames NOT changed")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()