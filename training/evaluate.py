"""
Standalone evaluation — run a trained checkpoint against a held-out test
set that was never seen during training/validation (a true generalization
check, not just the val split train.py already reports each epoch).

Usage:
    python -m training.evaluate \
        --checkpoint checkpoints/cassava_resnet18.pt \
        --test-dir data/cassava_test \
        --image-size 224
"""
import argparse
import json

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from training.dataset import build_transforms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--f1-floor", type=float, default=0.85)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    transform = build_transforms(args.image_size, train=False)
    test_ds = datasets.ImageFolder(args.test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds, target_names=test_ds.classes, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"test set: {len(test_ds)} images across {len(test_ds.classes)} classes\n")
    print(report)
    print("confusion matrix (rows=true, cols=predicted):")
    print(json.dumps(cm.tolist(), indent=2))
    print(f"\nmacro-F1 on held-out test set: {macro_f1:.4f}")

    if macro_f1 >= args.f1_floor:
        print(f"PASS — meets the {args.f1_floor:.0%} F1 floor from the brief.")
    else:
        print(f"FAIL — below the {args.f1_floor:.0%} F1 floor. Not ready to ship.")


if __name__ == "__main__":
    main()
