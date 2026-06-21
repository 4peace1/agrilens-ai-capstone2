"""
Synthetic leaf-image generator.

This is NOT a substitute for real training data — a model trained only
on this will not generalize to real photos and must never be shipped.
Its purpose is narrower: let you run the *entire* pipeline (dataset
loading -> train.py -> quantize.py -> model_server) end-to-end in
minutes, on a laptop, with zero downloads, to catch wiring bugs before
you spend hours downloading and training on the real ~20k-image
datasets referenced in training/README.md.

It procedurally draws a leaf-like blob with class-specific "lesion"
patterns (different colors/shapes per disease) so the classes are at
least trivially separable — enough to prove the pipeline produces a
checkpoint, computes metrics, and exports correctly.
"""
import argparse
import pathlib
import random

from PIL import Image, ImageDraw

# Mirrors the real public datasets referenced in training/README.md, so
# folder names line up with what you'd get from the actual Kaggle
# downloads later.
CROP_CLASSES = {
    "cassava": ["healthy", "cmd", "cbsd", "cbb", "cgm"],
    "cocoa": ["healthy", "black_pod_rot", "pod_borer"],
    "maize": ["healthy", "common_rust", "gray_leaf_spot", "blight"],
}

LEAF_GREEN = (60, 130, 60)
BACKGROUND = (210, 200, 170)  # dirt/soil-ish background, like a field photo

# Deterministic per-class lesion color/pattern so the synthetic classes
# are actually distinguishable (a real model trained on noise that
# *isn't* separable would be a useless smoke test).
LESION_STYLE = {
    "healthy": None,
    "cmd": {"color": (210, 210, 90), "pattern": "mosaic"},
    "cbsd": {"color": (120, 80, 40), "pattern": "streak"},
    "cbb": {"color": (40, 40, 30), "pattern": "spots"},
    "cgm": {"color": (180, 150, 60), "pattern": "speckle"},
    "black_pod_rot": {"color": (20, 20, 20), "pattern": "blotch"},
    "pod_borer": {"color": (90, 50, 20), "pattern": "holes"},
    "common_rust": {"color": (170, 90, 40), "pattern": "speckle"},
    "gray_leaf_spot": {"color": (140, 140, 140), "pattern": "spots"},
    "blight": {"color": (90, 70, 30), "pattern": "blotch"},
}


def _draw_pattern(draw: ImageDraw.ImageDraw, size: int, style: dict, rng: random.Random) -> None:
    color = style["color"]
    pattern = style["pattern"]
    if pattern == "spots":
        for _ in range(rng.randint(8, 20)):
            x, y = rng.randint(0, size), rng.randint(0, size)
            r = rng.randint(3, 8)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    elif pattern == "streak":
        for _ in range(rng.randint(3, 6)):
            x1 = rng.randint(0, size)
            y1 = rng.randint(0, size)
            length = rng.randint(size // 4, size // 2)
            angle_dx = rng.choice([-1, 1]) * length
            draw.line([x1, y1, x1 + angle_dx, y1 + length], fill=color, width=4)
    elif pattern == "mosaic":
        step = size // 8
        for gx in range(0, size, step):
            for gy in range(0, size, step):
                if rng.random() < 0.35:
                    draw.rectangle([gx, gy, gx + step, gy + step], fill=color)
    elif pattern == "speckle":
        for _ in range(rng.randint(40, 80)):
            x, y = rng.randint(0, size), rng.randint(0, size)
            draw.point((x, y), fill=color)
    elif pattern == "blotch":
        x, y = rng.randint(size // 4, 3 * size // 4), rng.randint(size // 4, 3 * size // 4)
        r = rng.randint(size // 6, size // 3)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    elif pattern == "holes":
        for _ in range(rng.randint(5, 10)):
            x, y = rng.randint(0, size), rng.randint(0, size)
            r = rng.randint(2, 5)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=BACKGROUND)


def generate_image(crop: str, label: str, size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(img)

    # Base leaf/pod shape — an irregular ellipse, roughly centered, with
    # per-image jitter so images within a class aren't pixel-identical.
    cx, cy = size // 2 + rng.randint(-10, 10), size // 2 + rng.randint(-10, 10)
    rx, ry = int(size * 0.38), int(size * 0.45)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=LEAF_GREEN)

    style = LESION_STYLE.get(label)
    if style is not None:
        _draw_pattern(draw, size, style, rng)

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", required=True, choices=CROP_CLASSES.keys())
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images-per-class", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    classes = CROP_CLASSES[args.crop]
    out_root = pathlib.Path(args.output_dir)
    counter = 0
    for label in classes:
        class_dir = out_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.images_per_class):
            img = generate_image(args.crop, label, args.image_size, seed=counter)
            img.save(class_dir / f"{label}_{i:04d}.jpg", quality=90)
            counter += 1

    total = len(classes) * args.images_per_class
    print(f"wrote {total} synthetic images across {len(classes)} classes to {out_root}")
    print("Reminder: this is for pipeline smoke-testing only, not for a "
          "model you'd ever ship — train on real data before deploying.")


if __name__ == "__main__":
    main()
