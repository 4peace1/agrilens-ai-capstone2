"""
Reorganizes the raw Kaggle "Cassava Leaf Disease Classification"
competition download into the per-class subfolder layout
`training/dataset.py` expects.

The competition ships as:
    train_images/*.jpg          (flat, ~21k files)
    train.csv                   (image_id, label as an integer 0-4)
    label_num_to_disease_map.json

This script reads the CSV + label map and copies (or symlinks) each
image into `<output_dir>/<disease_name>/<image_id>`.

Usage:
    python -m training.prepare_cassava_kaggle \
        --raw-dir data/raw/cassava \
        --output-dir data/cassava
"""
import argparse
import json
import pathlib
import shutil

# Kaggle's numeric labels map to these disease codes (matches
# training/synthetic_data.py and inference/model_server's default
# labels so folder names line up everywhere in the repo).
LABEL_NAME_OVERRIDE = {
    "Cassava Bacterial Blight (CBB)": "cbb",
    "Cassava Brown Streak Disease (CBSD)": "cbsd",
    "Cassava Green Mottle (CGM)": "cgm",
    "Cassava Mosaic Disease (CMD)": "cmd",
    "Healthy": "healthy",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True,
                         help="Folder containing train_images/, train.csv, "
                              "and label_num_to_disease_map.json as downloaded from Kaggle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symlink", action="store_true",
                         help="Symlink instead of copy — faster, saves disk space, "
                              "but the raw-dir must not be deleted afterward")
    parser.add_argument("--limit-per-class", type=int, default=None,
                         help="Optional cap per class, useful for a quick first "
                              "training run before committing to the full ~21k images")
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    images_dir = raw_dir / "train_images"
    csv_path = raw_dir / "train.csv"
    label_map_path = raw_dir / "label_num_to_disease_map.json"

    for required in (images_dir, csv_path, label_map_path):
        if not required.exists():
            raise FileNotFoundError(
                f"expected {required} — check the Kaggle download unzipped "
                "into --raw-dir correctly (see training/README.md)"
            )

    label_map = json.loads(label_map_path.read_text())
    # label_map looks like {"0": "Cassava Bacterial Blight (CBB)", ...}
    label_name_by_num = {
        num: LABEL_NAME_OVERRIDE.get(name, name) for num, name in label_map.items()
    }

    output_dir = pathlib.Path(args.output_dir)
    per_class_count: dict[str, int] = {}

    with open(csv_path) as f:
        next(f)  # header: image_id,label
        rows = [line.strip().split(",") for line in f if line.strip()]

    for image_id, label_num in rows:
        label_name = label_name_by_num[label_num]
        count = per_class_count.get(label_name, 0)
        if args.limit_per_class is not None and count >= args.limit_per_class:
            continue

        class_dir = output_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)
        src = images_dir / image_id
        dst = class_dir / image_id

        if not src.exists():
            continue  # a handful of CSV rows can reference missing files; skip safely

        if args.symlink:
            if not dst.exists():
                dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

        per_class_count[label_name] = count + 1

    total = sum(per_class_count.values())
    print(f"organized {total} images into {output_dir}")
    for name, count in sorted(per_class_count.items()):
        print(f"  {name}: {count}")
    print(
        "\nReminder: this dataset is imbalanced (cmd is the majority class) — "
        "training/train.py applies inverse-frequency class weighting "
        "automatically, no extra step needed here."
    )


if __name__ == "__main__":
    main()
