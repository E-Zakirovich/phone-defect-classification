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

        # I am going to make nnet, the size of the nnet is 120 -> 84 -> 7

        # fully connected layer one (connection between hidden layer and output layer)
        self.fully_connected_one = nn.Linear(
            input = 120, out_features=84
        )

        # fully connected layer two (connection between hidden layer and output layer)
        self.fully_connected_two = nn.Linear(
            input = 84, out_features = 3
        )

    # forward propagation
    def forward(self, x):

        # first convolutional layer
        x = self.pooling(self.tanh(self.convolutional_layer_one(x)))

        # second convolutional layer
        x = self.pooling(self.tanh(self.convolutional_layer_two(x)))

        # third convolutional layer
        x = self.pooling(self.tanh(self.convolutional_layer_three(x)))

        # falattening last layer for neural networks
        x = x.view(x.size(0), -1)

        # connection between input layer and hidden layer
        x = self.tanh(self.fully_connected_one(x))

        # connection between hidden layer and output layer
        x = self.softmax(self.fully_connected_two(x))

        # return the result
        return x