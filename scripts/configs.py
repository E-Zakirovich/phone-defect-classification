"""
configs.py
~~~~~~~~~~

Single place for the defaults and per-architecture settings, so nothing has to
be duplicated across main.py, train.py and data.py.
"""

from dataclasses import dataclass
from pathlib import Path

# ---- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"


@dataclass(frozen=True)
class ArchSpec:
    """Everything that differs between architectures."""
    name: str
    image_size: int     # the resolution the network expects
    lr: float           # sane starting learning rate for this net
    batch_size: int     # fits comfortably on a 12 GB card


# LeNet's third conv must land on 1x1, which only happens at 32x32.
# AlexNet has no BatchNorm: 1e-3 explodes its logits (val loss ~8000), 1e-4 is stable.
# ResNet has BatchNorm and tolerates a slightly higher LR than AlexNet.
ARCHITECTURES = {
    "lenet":   ArchSpec("lenet",   32,  1e-3, 64),
    "alexnet": ArchSpec("alexnet", 224, 1e-4, 32),
    "resnet":  ArchSpec("resnet",  224, 3e-4, 32),
    "googlenet": ArchSpec("googlenet", 224, 5e-4, 32),
    "convnext":  ArchSpec("convnext", 224, 3e-4, 32),
}

# ---- training defaults ---------------------------------------------------
EPOCHS = 15
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
EARLY_STOP_PATIENCE = 5      # epochs without val improvement before stopping
VAL_FRACTION = 0.2           # only used when the dataset has no val split
NUM_WORKERS = 8
SEED = 42
