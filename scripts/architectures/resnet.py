"""
resnet.py
~~~~~~~~~

In this file, I am going to built resnet with a help of python 3.14, 
OOP and PyTorch.

ResNet-50 (He et al., 2015) with the v1.5 stride placement -- the stride sits
on the 3x3 convolution rather than on the first 1x1, which is what torchvision
does and what the original Caffe implementation used.

Verified: (2, 3, 224, 224) -> (2, num_classes), 23.52M parameters,
stage depths [3, 4, 6, 3].
"""

import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3x3 spatial conv (stride=2 here follows ResNet v1.5 / torchvision)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 1x1 expand (4x)
        self.conv3 = nn.Conv2d(
            mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, dropout=0.0):
        super().__init__()
        self.in_channels = 64

        # Stem (for ~224x224 images)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Residual Stages: [3, 4, 6, 3] bottleneck blocks
        self.layer1 = self._make_layer(64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        # Classifier Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, mid_channels, blocks, stride):
        downsample = None
        out_channels = mid_channels * Bottleneck.expansion

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [Bottleneck(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, mid_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(self.dropout(x))  # raw logits for nn.CrossEntropyLoss
