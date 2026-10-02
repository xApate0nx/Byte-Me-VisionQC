from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "normal"
DEFECT_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "defects"

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "roi_candidates"

# Candidate central ROI sizes as a fraction of the normalized image.
# We will compare these visually before choosing one.
CANDIDATES = [
    ("large", 0.80),
    ("medium", 0.70),
    ("small", 0.60),
]

CANVAS_SIZE = 400
THUMB_SIZE = 220


def load_normalized(path):
    """Load image, apply EXIF orientation, then fit into a square canvas."""
    img = Image.open(path).convert("RGB")

    # Respect camera/WhatsApp EXIF orientation.
    img = ImageOps.exif_transpose(img)

    # Resize while preserving aspect ratio.
    img.thumbnail((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")

    x = (CANVAS_SIZE - img.width) // 2
    y = (CANVAS_SIZE - img.height) // 2

    canvas.paste(img, (x, y))

    return canvas


def central_crop(img, fraction):
    """Crop a centered square ROI."""
    size = int(CANVAS_SIZE * fraction)

    left = (CANVAS_SIZE - size) // 2
    top = (CANVAS_SIZE - size) // 2
    right = left + size
    bottom = top + size

    return img.crop((left, top, right, bottom))


def make_contact_sheet(images, output_path, title):
    """Create a labeled contact sheet."""
    columns = 4
    rows = (len(images) + columns - 1) // columns

    sheet_width = columns * THUMB_SIZE
    sheet_height = rows * (THUMB_SIZE + 35) + 50

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((10, 10), title, fill="black")

    for i, (name, img) in enumerate(images):
        row = i // columns
        col = i % columns

        x = col * THUMB_SIZE
        y = 50 + row * (THUMB_SIZE + 35)

        thumb = img.copy()
        thumb.thumbnail((THUMB_SIZE - 10, THUMB_SIZE - 10))

        px = x + (THUMB_SIZE - thumb.width) // 2
        py = y + (THUMB_SIZE - thumb.height) // 2

        sheet.paste(thumb, (px, py))

        label = name[:28]
        draw.text((x + 5, y + THUMB_SIZE), label, fill="black")

    sheet.save(output_path, quality=95)


def process_group(input_dir, group_name):
    images = []

    files = sorted(
        list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg"))
        + list(input_dir.glob("*.png"))
    )

    print(f"\n{group_name.upper()}")
    print(f"Images found: {len(files)}")

    for path in files:
        normalized = load_normalized(path)

        for candidate_name, fraction in CANDIDATES:
            output_dir = OUTPUT_DIR / candidate_name / group_name
            output_dir.mkdir(parents=True, exist_ok=True)

            cropped = central_crop(normalized, fraction)

            # Resize every candidate to the same model input size.
            cropped = cropped.resize(
                (256, 256),
                Image.Resampling.LANCZOS,
            )

            output_path = output_dir / path.name
            cropped.save(output_path, quality=95)

        images.append((path.name, normalized))

    # Normalized contact sheet.
    make_contact_sheet(
        images,
        OUTPUT_DIR / f"{group_name}_normalized_contact_sheet.jpg",
        f"{group_name.upper()} — normalized images",
    )


def main():
    print("=" * 70)
    print("VisionQC — ROI Candidate Generator")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    process_group(NORMAL_DIR, "normal")
    process_group(DEFECT_DIR, "defects")

    print("\n" + "=" * 70)
    print("ROI candidate generation complete.")
    print("=" * 70)

    print("\nGenerated:")
    for candidate_name, fraction in CANDIDATES:
        print(
            f"  {candidate_name:8s} = {int(fraction * 100)}% "
            f"central square ROI"
        )

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()