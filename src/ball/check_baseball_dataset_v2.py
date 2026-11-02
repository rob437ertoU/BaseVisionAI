from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "baseball_v2"
)


def count_split(split_name):

    images_dir = (
        DATASET_DIR
        / split_name
        / "images"
    )

    labels_dir = (
        DATASET_DIR
        / split_name
        / "labels"
    )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    images = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    positives = 0
    negatives = 0
    total_boxes = 0

    for image_path in images:

        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            negatives += 1
            continue

        content = (
            label_path
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if not content:
            negatives += 1
            continue

        lines = [
            line
            for line in content.splitlines()
            if line.strip()
        ]

        if len(lines) == 0:
            negatives += 1
        else:
            positives += 1
            total_boxes += len(lines)

    return {
        "images": len(images),
        "positives": positives,
        "negatives": negatives,
        "boxes": total_boxes,
    }


def main():

    print()
    print("=" * 70)
    print("BaseVisionAI - Baseball Dataset V2 Check")
    print("=" * 70)

    total_images = 0
    total_positives = 0
    total_negatives = 0
    total_boxes = 0

    for split in [
        "train",
        "valid",
        "test"
    ]:

        stats = count_split(split)

        total_images += stats["images"]
        total_positives += stats["positives"]
        total_negatives += stats["negatives"]
        total_boxes += stats["boxes"]

        positive_pct = (
            100.0
            * stats["positives"]
            / stats["images"]
            if stats["images"] > 0
            else 0.0
        )

        print()
        print(split.upper())
        print("-" * 70)

        print(
            f"Images        : {stats['images']}"
        )

        print(
            f"Positive      : {stats['positives']}"
        )

        print(
            f"Negative      : {stats['negatives']}"
        )

        print(
            f"Bounding boxes: {stats['boxes']}"
        )

        print(
            f"Positive %    : {positive_pct:.1f}%"
        )

    print()
    print("=" * 70)
    print("TOTAL")
    print("=" * 70)

    print(
        f"Images        : {total_images}"
    )

    print(
        f"Positive      : {total_positives}"
    )

    print(
        f"Negative      : {total_negatives}"
    )

    print(
        f"Bounding boxes: {total_boxes}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()