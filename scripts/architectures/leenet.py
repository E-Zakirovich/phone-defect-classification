"""
leenet.py
~~~~~~~~~

In this file, I am going to built leenet with a help of python 3.14, 
OOP and PyTorch.

LeNet-5 (LeCun et al., 1998).

NOTE: this network expects 32x32 input. The third convolution reduces 5x5
down to a single 1x1 cell, which is what makes the 120 -> 84 -> n head work.
Feeding it 224x224 will NOT resize anything -- it will just produce a huge
flattened vector and fail in fully_connected_one.
"""

from torch import nn

class LeeNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # first convolutional layer: 32x32 -> 28x28
        self.convolutional_layer_one = nn.Conv2d(
            in_channels = 3,
            out_channels = 6,
            kernel_size = 5,
            stride = 1,
            padding = 0
        )


        # second convolutional layer: 14x14 -> 10x10
        self.convolutional_layer_two = nn.Conv2d(
            in_channels = 6,
            out_channels = 16,
            kernel_size = 5,
            stride = 1,
            padding = 0
        )

        # third convolutional layer: 5x5 -> 1x1
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

        # I am going to make nnet, the size of the nnet is 120 -> 84 -> n

        # fully connected layer one (connection between hidden layer and output layer)
        self.fully_connected_one = nn.Linear(
            in_features = 120, out_features = 84
        )

        # fully connected layer two (connection between hidden layer and output layer)
        self.fully_connected_two = nn.Linear(
            in_features = 84, out_features = num_classes
        )

    # forward propagation
    def forward(self, x):

        # first convolutional layer
        x = self.pooling(self.tanh(self.convolutional_layer_one(x)))

        # second convolutional layer
        x = self.pooling(self.tanh(self.convolutional_layer_two(x)))

        # third convolutional layer -- no pooling here, it is already 1x1
        x = self.tanh(self.convolutional_layer_three(x))

        # falattening last layer for neural networks
        x = x.flatten(1)

        # connection between input layer and hidden layer
        x = self.tanh(self.fully_connected_one(x))

        # connection between hidden layer and output layer
        # note: raw logits, no softmax -- CrossEntropyLoss applies log-softmax
        x = self.fully_connected_two(x)

        # return the result
        return x
