# Training pipeline

Two modes:

1. **Synthetic smoke test** — runs in minutes, no downloads, proves the
   pipeline (data loading -> training -> labels sidecar -> quantization
   -> serving) is wired correctly end-to-end. **Never ship a model
   trained only on this.**
2. **Real data** — uses the public datasets below, which is what
   produces a model actually worth deploying.

## Quick start: synthetic smoke test

```bash
pip install -r training/requirements.txt

python -m training.synthetic_data \
    --crop cassava --output-dir data/synthetic/cassava --images-per-class 40

python -m training.train \
    --crop cassava \
    --data-dir data/synthetic/cassava \
    --output checkpoints/cassava_resnet18.pt \
    --epochs 2 --no-pretrained
```

This was run as part of building this repo (2 epochs, random-init
weights, 150 synthetic images) and correctly completed end-to-end,
producing a checkpoint, a `.labels.json` sidecar, and a `.metrics.json`
file — and correctly reported a macro-F1 well under the 85% floor, since
that's exactly what undertrained model on synthetic shapes should do.
That's the pipeline working as intended, not a bug.

`--no-pretrained` is required in network-restricted environments (no
access to download ImageNet weights from `download.pytorch.org`). On
your own machine, drop that flag to use transfer learning, which you'll
want for real training — see below.

## Real data: verified public datasets

| Crop | Dataset | Classes | Size | Link |
|---|---|---|---|---|
| Cassava | Cassava Leaf Disease Classification (Kaggle / Makerere AI Lab) | healthy, cmd (mosaic), cbsd (brown streak), cbb (bacterial blight), cgm (green mottle) | 21,367 images | kaggle.com/competitions/cassava-leaf-disease-classification |
| Maize | Corn or Maize Leaf Disease Dataset (Kaggle, built from PlantVillage + PlantDoc) | healthy, common_rust, gray_leaf_spot, blight | 4,188 images | kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset |
| Cocoa | Cacao Disease (Kaggle) | healthy, black_pod_rot, pod_borer | ~4,300 images | search "Cacao Disease" on Kaggle — listing changes more often than the other two, verify before downloading |

**Known data quality issue to handle, not ignore:** the cassava dataset
is heavily imbalanced — about 62% of images are the CMD (mosaic disease)
class. `training/dataset.py::compute_class_weights` applies inverse-
frequency weighting to the loss specifically because of this; training
without it will produce a model that looks accurate but has poor
per-class F1 on the minority disease classes, which is exactly the
85%-macro-F1 floor in the brief is designed to catch (macro-F1 — not
accuracy — penalizes that).

### Downloading via the Kaggle API

```bash
pip install kaggle
# Get an API token from kaggle.com/settings -> API -> Create New Token,
# save it to ~/.kaggle/kaggle.json

kaggle competitions download -c cassava-leaf-disease-classification -p data/raw/cassava
kaggle datasets download -d smaranjitghose/corn-or-maize-leaf-disease-dataset -p data/raw/maize
kaggle datasets download -d <cacao-dataset-slug> -p data/raw/cocoa  # verify current slug on Kaggle first
```

Unzip each into the `data/<crop>/<class_name>/*.jpg` layout
`training/dataset.py` expects (an `ImageFolder` structure — one
subfolder per class). The Kaggle competition download in particular
ships as a flat CSV + image folder rather than pre-sorted by class; a
short reshuffling script (not included here, since the exact CSV schema
can change between competition re-uploads) will be needed to sort images
into class subfolders before training.

### Real training run

```bash
python -m training.train \
    --crop cassava \
    --data-dir data/cassava \
    --output checkpoints/cassava_resnet18.pt \
    --epochs 15 --batch-size 32 --lr 1e-4
```

Drop `--no-pretrained` here — transfer learning from ImageNet weights is
what gets a ResNet18 to reasonable accuracy on ~20k images in 15 epochs;
training from random init on a dataset this size will undertrain badly
the way the synthetic smoke test deliberately does.

### Verify against a held-out test set

```bash
python -m training.evaluate \
    --checkpoint checkpoints/cassava_resnet18.pt \
    --test-dir data/cassava_test \
    --f1-floor 0.85
```

Keep a true test split (not just train.py's internal val split) that the
model never sees during training or hyperparameter tuning — this is what
actually validates the 85% F1 floor rather than an optimistic number
from data the model indirectly influenced.

### Quantize and deploy

```bash
python -m inference.quantize \
    --checkpoint checkpoints/cassava_resnet18.pt \
    --calibration-dir data/cassava/healthy \
    --output models/cassava_v3_int8.tflite
```

The `.labels.json` sidecar travels automatically from the checkpoint to
the `.tflite` output (see `inference/quantize.py`), so
`inference/model_server/app.py` serves the exact class list the model
was trained on without needing a manual edit.

## Why ResNet18 and not something bigger

`training/train.py` uses ResNet18 deliberately, not ResNet50 or an
EfficientNet variant: it's the smallest backbone that still transfer-
learns well on leaf-texture classification, and a smaller backbone
quantizes to a smaller, faster `.tflite` file — which is what makes the
brief's <2.0s CPU-only inference SLA realistic on commodity GKE nodes
without GPUs. If accuracy on the real dataset comes in short of 85% F1
after reasonable tuning, the next lever to pull is more/better data and
augmentation before reaching for a bigger backbone, since model size
directly trades off against the latency SLA.
