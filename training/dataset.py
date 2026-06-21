"""
Dataset loading utilities, shared by train.py and evaluate.py.

Expects an ImageFolder-style layout (works for both the synthetic
generator output and the real public datasets once unpacked):

    data/cassava/
        healthy/*.jpg
        cmd/*.jpg
        cbsd/*.jpg
        cbb/*.jpg
        cgm/*.jpg

Class names are discovered from the folder names — never hardcoded —
so this works unmodified whether you're using the 5-class Kaggle cassava
set, the 4-class maize set, or the synthetic smoke-test data.
"""
import json
import pathlib

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Training augmentation deliberately mimics the brief's stated
    constraint: 'varying resolutions and aspect ratios from diverse
    budget smartphone cameras.' Random resized crop + color jitter +
    rotation simulate that variability so the model doesn't overfit to
    clean, centered, well-lit training photos."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_datasets(data_dir: str, image_size: int, val_split: float = 0.2, seed: int = 42):
    """Returns (train_dataset, val_dataset, class_names).

    Train and val need *different* transforms (augmentation only on
    train), so we load the folder twice with torchvision's ImageFolder
    and then split indices identically across both — this avoids
    leaking augmented duplicates of validation images into training.
    """
    base = datasets.ImageFolder(data_dir)
    class_names = base.classes

    n_val = int(len(base) * val_split)
    n_train = len(base) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(base)), [n_train, n_val], generator=generator
    )

    train_ds = datasets.ImageFolder(data_dir, transform=build_transforms(image_size, train=True))
    val_ds = datasets.ImageFolder(data_dir, transform=build_transforms(image_size, train=False))

    train_subset = torch.utils.data.Subset(train_ds, list(train_indices))
    val_subset = torch.utils.data.Subset(val_ds, list(val_indices))

    return train_subset, val_subset, class_names


def build_dataloaders(data_dir: str, image_size: int, batch_size: int, val_split: float = 0.2,
                       num_workers: int = 1):
    train_ds, val_ds, class_names = load_datasets(data_dir, image_size, val_split)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, class_names


def compute_class_weights(data_dir: str, class_names: list[str]) -> torch.Tensor:
    """Inverse-frequency class weights for the loss function. The real
    cassava dataset is heavily imbalanced (~62% of images are a single
    disease class — see training/README.md), and training on it
    unweighted will produce a model that just predicts the majority
    class and looks deceptively accurate while having poor per-class
    F1 — exactly what the brief's 85% F1 floor is meant to catch."""
    counts = []
    for name in class_names:
        class_dir = pathlib.Path(data_dir) / name
        counts.append(len(list(class_dir.glob("*"))))
    counts_t = torch.tensor(counts, dtype=torch.float32)
    weights = counts_t.sum() / (len(counts_t) * counts_t.clamp(min=1))
    return weights


def save_labels(class_names: list[str], output_path: str) -> None:
    """Writes a `{version}.labels.json` sidecar next to a checkpoint /
    .tflite artifact. inference/model_server/app.py reads this at
    startup so the serving labels are always exactly what the model was
    actually trained on — no risk of a hand-maintained label list in
    app.py silently drifting out of sync with the model."""
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_names, indent=2))
