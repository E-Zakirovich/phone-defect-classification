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

        # first convolutional layer
        self.convolutional_layer_one = nn.Conv2d(
            in_channels = 3,
            out_channels = 6,
            kernel_size = 5,
            stride = 1,
            padding = 0
        )


        # second convolutional layer
        self.convolutional_layer_two = nn.Conv2d(
            in_channels = 6,
            out_channels = 16,
            kernel_size = 5,
            stride = 1,
            padding = 0
        )

        # third convolutional layer
        self.convolutional_layer_three = nn.Conv2d(
            in_channels = 16,
            out_channels = 120,
            kernel_size = 5,
            stride = 1,
            padding = 0
        )

        # pooling layer
        self.pooling = nn.AvgPool2d(
            kernel_size = 2,
            stride = 2
        )

        # activation functions
        self.tanh = nn.Tanh() # for convolution and hidden layers
        self.softmax = nn.Softmax() # for output layer

