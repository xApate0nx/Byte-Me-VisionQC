from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NORMAL_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "normal"
DEFECT_DIR = PROJECT_ROOT / "data" / "products" / "water_cap_v1" / "defects"

OUTPUT_DIR = PROJECT_ROOT / "data" / "inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THUMBNAIL_SIZE = (220, 180)
LABEL_HEIGHT = 35
COLUMNS = 4


def get_images(directory):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in extensions
    )


def create_contact_sheet(paths, title, output_path):

    rows = (len(paths) + COLUMNS - 1) // COLUMNS

    sheet_width = COLUMNS * THUMBNAIL_SIZE[0]
    sheet_height = 50 + rows * (THUMBNAIL_SIZE[1] + LABEL_HEIGHT)

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((10, 10), title, fill="black")

    for index, path in enumerate(paths):

        image = Image.open(path).convert("RGB")
        image.thumbnail(THUMBNAIL_SIZE)

        x = (index % COLUMNS) * THUMBNAIL_SIZE[0]
        y = 50 + (index // COLUMNS) * (THUMBNAIL_SIZE[1] + LABEL_HEIGHT)

        image_x = x + (THUMBNAIL_SIZE[0] - image.width) // 2
        image_y = y + (THUMBNAIL_SIZE[1] - image.height) // 2

        sheet.paste(image, (image_x, image_y))

        filename = path.name

        # Keep label readable
        if len(filename) > 27:
            filename = filename[:24] + "..."

        draw.text(
            (x + 5, y + THUMBNAIL_SIZE[1]),
            f"{index + 1}: {filename}",
            fill="black"
        )

    sheet.save(output_path, quality=95)

    print(f"Saved: {output_path}")


normal_paths = get_images(NORMAL_DIR)
defect_paths = get_images(DEFECT_DIR)

print(f"Normal images: {len(normal_paths)}")
print(f"Defect images: {len(defect_paths)}")

create_contact_sheet(
    normal_paths,
    "VISIONQC — NORMAL DATASET",
    OUTPUT_DIR / "normal_contact_sheet.jpg"
)

create_contact_sheet(
    defect_paths,
    "VISIONQC — DEFECT DATASET",
    OUTPUT_DIR / "defect_contact_sheet.jpg"
)

print("\nDataset inspection sheets created successfully.")