"""
alexnet.py
~~~~~~~~~

In this file, I am going to built alexnet with a help of python 3.14, 
OOP and PyTorch.

AlexNet (Krizhevsky et al., 2012).

Differences from the paper, on purpose:
  * an AdaptiveAvgPool2d((6, 6)) sits before the classifier, so any input
    from 128x128 up to 224x224 (and beyond) produces the 6x6x256 = 9216
    vector the first fully connected layer expects. Without it the net only
    works at exactly 227x227.
  * no Local Response Normalisation (deprecated in modern PyTorch; BatchNorm
    would be the usual replacement).
"""

from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # first convolutonal layer
        self.convolutional_layer_one = nn.Conv2d(
            in_channels = 3,
            out_channels = 96,
            kernel_size = 11,
            stride = 4,
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
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        # fifth convolutional layer
        self.convolutional_layer_five = nn.Conv2d(
            in_channels = 384,
            out_channels = 256,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        # pooling layer
        self.pooling = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2
        )

        # makes the flatten size independent of the input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        # activation functions for hidden and convolution layers
        self.relu = nn.ReLU() # for convolutions and hidden layers

        # dropout keeps 60M parameters from memorising a small dataset
        self.dropout = nn.Dropout(p = 0.5)

        # making whole neural networks, the size is 9216 -> 4096 -> 4096 -> n
        
        # fully connected layer one (connection between input layer and hidden layer)
        self.fully_connected_one = nn.Linear(
            in_features = 9216, out_features = 4096
        )

        # fully connected layer two (connection between hidden layer and hidden layer)
        self.fully_connected_two = nn.Linear(
            in_features = 4096, out_features = 4096
        )

        # fully connected layer three (connection between hidden layer and output layer)
        self.fully_connected_three = nn.Linear(
            in_features = 4096, out_features = num_classes
        )

    # forward propagation
    def forward(self, x):

        # first convolutional layer
        x = self.pooling(self.relu(self.convolutional_layer_one(x)))

        # second convolutional layer
        x = self.pooling(self.relu(self.convolutional_layer_two(x)))

        # third convolutional layer
        x = self.relu(self.convolutional_layer_three(x))

        # fourth convolutional layer
        x = self.relu(self.convolutional_layer_four(x))

        # fifth convolutional layer
        x = self.pooling(self.relu(self.convolutional_layer_five(x)))

        # fixed 6x6 grid whatever the input resolution was
        x = self.adaptive_pool(x)

        # flattening the output of the last convolutional layer
        x = x.flatten(1)

        # connection between input layer and hidden layer
        x = self.dropout(self.relu(self.fully_connected_one(x)))

        # connection between hidden layer and hidden layer
        x = self.dropout(self.relu(self.fully_connected_two(x)))

        # connection between hidden layer and output layer
        # note: raw logits, no softmax -- CrossEntropyLoss applies log-softmax
        x = self.fully_connected_three(x)

        # returning the result
        return x
