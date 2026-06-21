"""
Training script — transfer learning on a ResNet18 backbone.

Usage (synthetic smoke test, no internet/pretrained weights needed):
    python -m training.synthetic_data --crop cassava --output-dir data/synthetic/cassava
    python -m training.train \
        --crop cassava \
        --data-dir data/synthetic/cassava \
        --output checkpoints/cassava_resnet18.pt \
        --epochs 2 --no-pretrained

Usage (real data, see training/README.md for dataset download):
    python -m training.train \
        --crop cassava \
        --data-dir data/cassava \
        --output checkpoints/cassava_resnet18.pt \
        --epochs 15

Outputs:
    <output>                       — torch.save'd model, loadable directly
                                      by inference/quantize.py
    <output>.labels.json           — class-index -> label-name mapping
    <output>.metrics.json          — best val macro-F1, per-class F1,
                                      confusion matrix

The brief requires a minimum 85% F1-score across the target crops — this
script reports macro-F1 every epoch specifically so that requirement is
visible during training, not just discovered after the fact in
production.
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import models

from training.dataset import build_dataloaders, compute_class_weights, save_labels


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """ResNet18 is the deliberate choice here, not ResNet50/EfficientNet:
    it's the smallest backbone that still transfer-learns well on
    leaf-texture tasks, which keeps the post-quantization TFLite model
    small enough for the brief's edge/low-bandwidth constraints and the
    <2.0s CPU inference SLA."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device, class_names) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    return {"macro_f1": macro_f1, "report": report, "confusion_matrix": cm}


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--f1-floor", type=float, default=0.85,
                         help="Matches the brief's minimum F1 requirement; "
                              "used only for the pass/fail summary at the end.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                         help="Use random init instead of ImageNet weights — needed if "
                              "you don't have internet access to download torchvision "
                              "pretrained weights (e.g. air-gapped environments).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} pretrained={args.pretrained}")

    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir, args.image_size, args.batch_size
    )
    print(f"classes={class_names} train_n={len(train_loader.dataset)} val_n={len(val_loader.dataset)}")

    weights = compute_class_weights(args.data_dir, class_names).to(device)
    print(f"class_weights={weights.tolist()}")

    model = build_model(len(class_names), args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device, class_names)
        scheduler.step()
        elapsed = time.monotonic() - start
        print(
            f"epoch={epoch}/{args.epochs} loss={train_loss:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} ({elapsed:.1f}s)"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics
            torch.save(model, args.output)

    save_labels(class_names, f"{args.output}.labels.json")
    with open(f"{args.output}.metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\nbest val macro-F1: {best_f1:.4f}")
    if best_f1 >= args.f1_floor:
        print(f"PASS — meets the {args.f1_floor:.0%} F1 floor from the brief.")
    else:
        print(
            f"BELOW FLOOR — {best_f1:.0%} < {args.f1_floor:.0%}. "
            "More real training data, more epochs, or a larger backbone "
            "needed before this is shippable. Do not promote to the "
            "stable A/B routing slot in inference/model_server/ab_router.py."
        )
    print(f"checkpoint: {args.output}")
    print(f"labels:     {args.output}.labels.json")
    print(f"metrics:    {args.output}.metrics.json")


if __name__ == "__main__":
    main()
