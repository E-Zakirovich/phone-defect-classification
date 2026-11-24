"""
alexnet.py
~~~~~~~~~

In this file, I am going to built alexnet with a help of python 3.14, 
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
            out_channels = 394,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        # fifth convolutional layer
        self.convolutional_layer_five = nn.Conv2d(
            in_channels = 384,
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        # pooling layer
        self.pooling = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2
        )

        # activation functions for hidden, convolution and output layers
        self.relu = nn.ReLU() # for convolutions and hidden layers
        self.softmax = nn.Softmax() # for output layer

        # making whole neural networks, the size is 9216 -> 4096 -> 4096 -> 7
        
        # fully connected layer one (connection between input layer and hidden layer)
        self.fully_connected_one = nn.Linear(
            input = 9216, out_features = 4096
        )

        # fully connected layer two (connection between hidden layer and hidden layer)
        self.fully_connected_two = nn.Linear(
            input = 4096, out_features = 4096
        )

        # fully connected layer three (connection between hidden layer and output layer)
        self.fully_connected_three = nn.Linear(
            input = 4096, out_features = 3
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

        # flattening the output of the last convolutional layer
        x = x.view(x.size(0), -1)

        # connection between input layer and hidden layer
        x = self.relu(self.fully_connected_one(x))

        # connection between hidden layer and hidden layer
        x = self.relu(self.fully_connected_two(x))

        # connection between hidden layer and output layer
        x = self.softmax(self.fully_connected_three(x))

        # returning the result
        return x