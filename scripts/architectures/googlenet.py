"""
googlenet.py
~~~~~~~~~~~~

In this file, I am going to built googlenet with a help of python 3.14, 
OOP and PyTorch.

GoogLeNet / Inception v1 (Szegedy et al., 2014).

The idea: instead of committing to one kernel size per layer, run 1x1, 3x3 and
5x5 convolutions *in parallel* on the same input and concatenate the results.
The 1x1 convolutions before the 3x3 and 5x5 branches are "reduction" layers --
they shrink the channel count so the expensive convolutions stay affordable.

    input
      |-- 1x1 conv ---------------------------|
      |-- 1x1 conv -> 3x3 conv ---------------|--> concat -> output
      |-- 1x1 conv -> 5x5 conv ---------------|
      |-- 3x3 maxpool -> 1x1 conv ------------|

Two auxiliary classifiers branch off after 4a and 4d. They existed to push
gradient into the middle of what was then a very deep network; modern training
mostly does not need them, so they are OFF by default and forward() returns a
single logits tensor. Pass aux_logits=True to get a tuple of three.
"""

import torch
from torch import nn
from torch.nn import functional as F


class BasicConv2d(nn.Module):
    """conv -> batchnorm -> relu, the unit every branch is built from."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Inception(nn.Module):
    """One Inception block: four parallel branches, concatenated on channels."""

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3,
                 ch5x5red, ch5x5, pool_proj):
        super().__init__()

        # branch 1: a single 1x1 convolution
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # branch 2: 1x1 reduce, then 3x3
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        # branch 3: 1x1 reduce, then 5x5
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        # branch 4: max pooling, then 1x1 to compress the pooled channels
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )


class InceptionAux(nn.Module):
    """Small classifier attached to the middle of the network."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # the original used avg_pool 5x5/3 then a 5x5 conv, which only lines up
        # by luck at one exact input size; adaptive pooling makes it work for any
        x = F.adaptive_avg_pool2d(x, (5, 5))
        x = self.conv0(x)
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, aux_logits=False, dropout=0.4):
        super().__init__()
        self.aux_logits = aux_logits

        # ---- stem: 224x224 -> 28x28 -------------------------------------
        self.conv1 = BasicConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- inception 3a / 3b: 28x28 -> 14x14 --------------------------
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- inception 4a..4e: 14x14 -> 7x7 -----------------------------
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- inception 5a / 5b: 7x7 -------------------------------------
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # ---- classifier --------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        # 3x
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # 4x
        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.aux_logits and self.training else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.aux_logits and self.training else None

        x = self.inception4e(x)
        x = self.maxpool4(x)

        # 5x
        x = self.inception5a(x)
        x = self.inception5b(x)

        # classifier
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)          # raw logits for nn.CrossEntropyLoss

        if self.aux_logits and self.training:
            return x, aux1, aux2
        return x
