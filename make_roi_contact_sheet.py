from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/evaluation/roi")
OUTPUT = ROOT / "roi_contact_sheet.jpg"

TITLE_HEIGHT = 50
THUMBNAIL_SIZE = 180
COLUMNS = 5
LABEL_HEIGHT = 25
CELL_HEIGHT = THUMBNAIL_SIZE + LABEL_HEIGHT

def load_roi_images(category):
    folder = ROOT / category

    if not folder.exists():
        print(f"ERROR: folder not found: {folder}")
        return []

    paths = sorted(folder.glob("*_roi.png"))

    if not paths:
        print(f"ERROR: no ROI images found in: {folder}")
        return []

    images = []

    for path in paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"WARNING: could not read: {path}")
            continue

        images.append((category, path, image))

    return images

def create_thumbnail(image):
    height, width = image.shape[:2]

    if height == 0 or width == 0:
        return None

    scale = min(
        THUMBNAIL_SIZE / width,
        THUMBNAIL_SIZE / height,
    )

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            THUMBNAIL_SIZE,
            THUMBNAIL_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    x_offset = (THUMBNAIL_SIZE - new_width) // 2
    y_offset = (THUMBNAIL_SIZE - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    return canvas

def add_label(cell, category, filename):
    label = category.upper()

    cv2.rectangle(
        cell,
        (0, THUMBNAIL_SIZE),
        (THUMBNAIL_SIZE, CELL_HEIGHT),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        cell,
        label,
        (5, THUMBNAIL_SIZE + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return cell

def main():
    normal_images = load_roi_images("normal")
    defect_images = load_roi_images("defects")

    images = normal_images + defect_images

    normal_count = len(normal_images)
    defect_count = len(defect_images)
    total_count = len(images)

    if total_count == 0:
        print("ERROR: no ROI images were found.")
        return

    rows = (
        total_count + COLUMNS - 1
    ) // COLUMNS

    sheet = np.zeros(
        (
            TITLE_HEIGHT + rows * CELL_HEIGHT,
            COLUMNS * THUMBNAIL_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    sheet[:] = (35, 35, 35)

    cv2.putText(
        sheet,
        "VisionQC Canonical ROI Validation",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for index, (category, path, image) in enumerate(images):
        thumbnail = create_thumbnail(image)

        if thumbnail is None:
            continue

        cell = np.zeros(
            (
                CELL_HEIGHT,
                THUMBNAIL_SIZE,
                3,
            ),
            dtype=np.uint8,
        )

        cell[:THUMBNAIL_SIZE, :] = thumbnail

        add_label(
            cell,
            category,
            path.name,
        )

        row = index // COLUMNS
        column = index % COLUMNS

        y1 = TITLE_HEIGHT + row * CELL_HEIGHT
        y2 = y1 + CELL_HEIGHT

        x1 = column * THUMBNAIL_SIZE
        x2 = x1 + THUMBNAIL_SIZE

        sheet[y1:y2, x1:x2] = cell

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(OUTPUT),
        sheet,
    )

    print()
    print("=" * 60)
    print("ROI CONTACT SHEET")
    print("=" * 60)
    print(f"Normal ROIs: {normal_count}")
    print(f"Defect ROIs: {defect_count}")
    print(f"Total ROIs: {total_count}")
    print(f"Output: {OUTPUT}")

if __name__ == "__main__":
    main()
