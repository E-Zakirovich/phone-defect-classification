"""
resnet.py
~~~~~~~~~

In this file, I am going to built resnet with a help of python 3.14, 
OOP and PyTorch. 
"""

from torch import nn

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        