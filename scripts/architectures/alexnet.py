"""
alexnet.py
~~~~~~~~~

In this file, I am going to built leenet with a help of python 3.14, 
OOP and PyTorch. 
"""

from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # first convolutonal layer
        self.convolutional_layer_one = nn.Conv2d(
            in_channels = 3,
            out_channels = 96,
            kernel_size = 11,
            stride = 1
            padding = 0
        )

        # second convolutional layer
        self.convolutional_layer_two = nn.Conv2d(
            in_channels = 96,
            out_channels = 256,
            kernel_size = 5,
            stride = 1,
            padding = 2
        )

        # third convolutional layer 
        self.convolutional_layer_three = nn.Conv2d(
            in_channels = 256,
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1 
        )

        # fourth convolutional layer
        self.convolutional_layer_four = nn.Conv2d(
            in_channels = 384,
            out_channels = 394,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        # fifth convolutional layer
        self.convolutional_layer_four = nn.Conv2d(
            in_channels = 384,
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        # activation functions for hidden, convolution and output layers
        self.relu = nn.ReLU() # for convolutions and hidden layers
        self.softmax = nn.Softmax() # for output layer