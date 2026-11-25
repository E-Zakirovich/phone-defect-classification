"""
convnext.py
~~~~~~~~~~~

In this file, I am going to use convnext_tiny with a help of python 3.14,
OOP and PyTorch.

ConvNeXt-Tiny (Liu et al., 2022) -- a modernised ConvNet that borrows design
choices from Vision Transformers: 7x7 depthwise convolutions, LayerNorm instead
of BatchNorm, GELU, and an inverted-bottleneck block layout.

Unlike every other architecture in this project, this one is **pretrained on
ImageNet** by default. That matters here: the dataset only has ~1.4k unique
source photos, so features learned from 1.28M images do most of the work and
the fine-tuning only re-points them at clean/defective.

    import torchvision.models as models
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

The only change is the classifier head: 1000 ImageNet classes -> num_classes.
"""

import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.1):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = convnext_tiny(weights=weights)
        self.pretrained = pretrained

        # torchvision's head is Sequential(LayerNorm2d, Flatten, Linear(768, 1000))
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

        # a little dropout before the new head helps with a small dataset
        if dropout and dropout > 0:
            self.model.classifier[1] = nn.Sequential(
                nn.Flatten(1), nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.model(x)          # raw logits for nn.CrossEntropyLoss
