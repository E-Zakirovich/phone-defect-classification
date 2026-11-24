"""
leenet.py
~~~~~~~~~

In this file, I am going to built leenet with a help of python 3.14, 
OOP and PyTorch. 
"""

from torch import nn

class LeeNet(nn.Module):
    def __init__(self):
        super(LeeNet, self).__init__()