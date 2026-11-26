# Phone Defect Classification

Detecting defects on smartphone screens — **cracked vs. clean** — with five convolutional
networks implemented in PyTorch, from LeNet-5 (1998) to ConvNeXt-Tiny (2022, ImageNet-pretrained).

The dataset ships as **YOLO object-detection annotations** (three defect types with bounding
boxes); this project collapses them to an image-level label so that classic classifiers can be
compared on identical data.

---

## Results

Held-out test set, **n = 308 images**, ground truth derived from the YOLO annotations.
No test image was used for training or model selection.

| Model | Year | Accuracy | Macro-F1 | AUC | mAP | Errors | Params | Pretrained |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LeNet-5 | 1998 | 0.9123 | 0.9118 | 0.9825 | 0.9833 | 27 | 0.06 M | no |
| AlexNet | 2012 | 0.9610 | 0.9607 | 0.9761 | 0.9765 | 12 | 58.29 M | no |
| ResNet-50 | 2015 | 0.9838 | 0.9836 | 0.9974 | 0.9975 | 5 | 23.51 M | no |
| GoogLeNet | 2014 | 0.9740 | 0.9737 | **0.9994** | **0.9994** | 8 | 5.98 M | no |
| **ConvNeXt-Tiny** | 2022 | **0.9903** | **0.9902** | 0.9971 | 0.9973 | **3** | 27.82 M | **yes** |

Per-class precision / recall:

| Model | P(clean) | R(clean) | P(defective) | R(defective) |
|---|---:|---:|---:|---:|
| LeNet-5 | 0.908 | 0.901 | 0.916 | 0.922 |
| AlexNet | 0.971 | 0.944 | 0.953 | 0.976 |
| ResNet-50 | 0.993 | 0.972 | 0.976 | 0.994 |
| GoogLeNet | 1.000 | 0.944 | 0.954 | **1.000** |
| ConvNeXt-Tiny | 0.986 | 0.993 | 0.994 | 0.988 |

**Reading the results**

- **ConvNeXt-Tiny has the lowest error count** (3/308), but it is pretrained on ImageNet, so this
  is not a fair architecture-to-architecture comparison — it is a demonstration of transfer learning.
- **GoogLeNet ranks best** (AUC/mAP 0.9994) and **never misses a defect** (166/166 recall), at the
  cost of 8 false alarms. For screening, that is the safer failure mode.
- All four from-scratch models separate the classes well (AUC ≥ 0.976); the gap between them is
  largely threshold placement, not representation quality.

> ⚠️ With 308 test images each error is **0.32 percentage points**. Differences of 1–3 images are
> noise, not a ranking. Treat the top three as tied at ≈98%.

Charts: [`reports/confusion_all.png`](reports/confusion_all.png),
[`reports/curves_all.png`](reports/curves_all.png), raw numbers in
[`reports/metrics.json`](reports/metrics.json).

---

## Dataset

YOLO format — one `.txt` per image, `class cx cy w h` normalised to [0, 1].

```
data/
├── train/  images/ 10,017   labels/ 10,017
├── val/    images/    306   labels/    306
└── test/   images/    308   labels/    308
```

| Split | Images | Clean (empty label) | Defective | Boxes |
|---|---:|---:|---:|---:|
| train | 10,017 | 4,467 (44.6%) | 5,550 (55.4%) | 13,719 |
| val | 306 | 131 (42.8%) | 175 (57.2%) | 472 |
| test | 308 | 142 (46.1%) | 166 (53.9%) | 430 |

**Classes:** `0 = dot`, `1 = scratch`, `2 = crack` — in train: 5,283 / 6,532 / 1,904 boxes.
Every defective image contains exactly **one** defect type, so 3-class classification is possible.

**Important properties**

- **~15 GB on disk**, 17 distinct resolutions, median 1024×1024 (max 6000×8000).
- **86% of the training set is synthetic augmentation** — 10,017 files from only **1,431 unique
  source photos** (7 augmented copies each). Val and test are *not* augmented.
- **No leakage**: the three splits share **zero** source photos (`train ∩ val = train ∩ test = 0`).
- `data/` is git-ignored; nothing in this repo depends on it being committed.

---

## Project structure

```
.
├── main.py                        # CLI entry point
├── train_all.sh                   # train several architectures in sequence
├── scripts/
│   ├── configs.py                 # per-architecture image size / lr / batch, defaults
│   ├── data.py                    # YOLO + ImageFolder loaders, transforms, splitting
│   ├── train.py                   # training loop, AMP, scheduler, resume, early stop
│   ├── evaluations.py             # confusion matrix, P/R/F1, macro-F1
│   ├── prepare_cache.py           # optional: pre-resize the dataset once
│   └── architectures/
│       ├── leenet.py              # LeNet-5        32×32    0.06 M
│       ├── alexnet.py             # AlexNet       224×224  58.29 M
│       ├── googlenet.py           # GoogLeNet     224×224   5.98 M
│       ├── resnet.py              # ResNet-50     224×224  23.51 M
│       └── convnext.py            # ConvNeXt-Tiny 224×224  27.82 M (pretrained)
├── models/                        # checkpoints (git-ignored)
├── logs/                          # one CSV per training run
└── reports/                       # metrics, confusion matrices, ROC/PR curves
```

---

## Installation

```bash
pip install torch torchvision pillow
# optional, only for regenerating reports/
pip install matplotlib
```

Verified with **Python 3.13.13**, **PyTorch 2.15.0+cu132**, **torchvision 0.29.2**, CUDA 13.2,
on an NVIDIA RTX 3060.

**For ConvNeXt-Tiny** (the only pretrained model) the ImageNet weights are downloaded on first use:

```bash
export TORCH_HOME="$PWD/.torchcache"     # keeps the 110 MB cache inside the project
```

If your network uses a TLS-intercepting proxy, `urllib` may fail with
`CERTIFICATE_VERIFY_FAILED`. Download the weights with `curl` instead:

```bash
mkdir -p .torchcache/hub/checkpoints
curl -L -o .torchcache/hub/checkpoints/convnext_tiny-983f1562.pth \
  https://download.pytorch.org/models/convnext_tiny-983f1562.pth
```

---

## Usage

```bash
# train (defaults are per-architecture: image size, lr, batch)
python main.py --arch resnet --device cuda --epochs 30 --test

# all five architectures, one after another
./train_all.sh

# continue an interrupted run
python main.py --arch resnet --device cuda --epochs 30 --resume models/resnet.state.pt

# fast smoke test — writes to <arch>_smoke.pt, never touches real weights
python main.py --arch lenet --limit 128 --epochs 1 --workers 0

# 4-class mode: clean / dot / scratch / crack
python main.py --arch convnext --labels type --device cuda
```

Key flags: `--data`, `--arch`, `--labels {binary,type}`, `--epochs`, `--batch-size`, `--lr`,
`--device {auto,cpu,cuda}`, `--limit`, `--workers`, `--patience`, `--resume`, `--test`,
`--compile`, `--no-amp`, `--no-scheduler`, `--validate`.

**Outputs**

| Path | Contents |
|---|---|
| `models/<arch>.pt` | best-epoch weights + class names |
| `models/<arch>.state.pt` | + AdamW moments and scheduler — the file to pass to `--resume` |
| `logs/<arch>.csv` | one row per epoch (loss, accuracy, lr, seconds) |

---

## Architectures

| Model | Input | Params | MACs/image | Head share | Note |
|---|---:|---:|---:|---:|---|
| LeNet-5 | 32×32 | 0.06 M | 0.7 M | 17% | **Only works at exactly 32×32** — its third conv must land on 1×1 |
| AlexNet | 224×224 | 58.29 M | 1,017 M | **94%** | 54.5 M params sit in two fully-connected layers |
| ResNet-50 | 224×224 | 23.51 M | 4,087 M | 0.02% | Bottleneck blocks `[3,4,6,3]`, v1.5 stride placement |
| GoogLeNet | 224×224 | 5.98 M | 1,582 M | 0.03% | 9 Inception modules; follows the paper (5×5 branch) |
| ConvNeXt-Tiny | 224×224 | 27.82 M | — | 0.01% | ImageNet-pretrained; only the head is new |

AlexNet, ResNet, GoogLeNet and ConvNeXt accept a range of input sizes thanks to adaptive pooling.
LeNet does not — that is an architectural constraint, not a bug.

---

## Evaluation

`--test` prints accuracy, per-class precision/recall/F1, macro-F1 and a confusion matrix.
`reports/` additionally holds ROC and precision–recall curves for all models.

**mAP note:** the numbers labelled `mAP` above are **image-level** average precision (mean AP over
the two one-vs-rest classes). They are *not* detection mAP@0.5 — that requires bounding-box
predictions and IoU matching, which these classifiers do not produce.

---

## Known limitations

1. **Only 1,431 unique source photos.** 86% of the training files are augmentations of those.
   Augmentation teaches invariance; it cannot teach new defect morphologies. This is the binding
   constraint on accuracy, not the architecture choice.
2. **A 308-image test set cannot resolve small differences.** One image = 0.32 pp.
3. **Confident failures on out-of-distribution inputs.** Two clean phones are misclassified as
   defective by *all five* models with high confidence — one photographed from the back (no screen
   visible), one in an unusual setting. Aggregate accuracy hides this; there is no "unknown" class
   and no confidence threshold.
4. **The annotations are underused.** The dataset carries 14,621 bounding boxes across three defect
   types, but the pipeline collapses everything to clean/defective. It cannot say *what* kind of
   defect or *where* it is.
5. **No detection baseline yet.** For reference, state-of-the-art detection on comparable
   smartphone-defect benchmarks (e.g. MSAdaNet, *Sensors* 2026) reaches only **mAP@0.5 ≈ 55%** —
   detection mAP is not comparable to the ~98% classification accuracy reported here.

---

## Roadmap

- [x] Five architectures implemented and trained
- [x] Unified data pipeline (YOLO + ImageFolder), AMP, resume, early stopping, CSV logging
- [x] Metrics suite: confusion matrix, P/R/F1, AUC, AP, ROC and PR curves
- [ ] `--labels type` models (clean / dot / scratch / crack)
- [ ] Verify that the 8,586 `_augN` images have correctly transformed bounding boxes
- [ ] YOLO detection baseline on this dataset
- [ ] Confidence threshold / "unknown" class for out-of-distribution inputs
- [ ] Pretrained backbones for the four from-scratch models, for a fair comparison

---

## Author

**Eshqo'ziyev Azamat** — Эшқўзиев Азамат
Artificial Intelligence Projects Office, "Agrobank" JSCB