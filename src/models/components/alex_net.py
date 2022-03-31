from torch import nn


class AlexNet(nn.Module):
    """AlexNet
    based on: https://blog.paperspace.com/popular-deep-learning-architectures-alexnet-vgg-googlenet/
    also: https://github.com/dansuh17/alexnet-pytorch/blob/master/model.py
    """

    def __init__(
        self,
        in_channel: int = 3,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.in_size = 227  # input size is 227x227
        self.in_channel = in_channel
        self.num_classes = num_classes

        # -- convolution layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=11, stride=4),  # 227x227x3 -> 55x55x96
            nn.ReLU(),
            # normalize across local region
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55x55x96 -> 27x27x96
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 27x27x96 -> 27x27x256
            nn.ReLU(),
            # normalize across local region
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x256 -> 13x13x256
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 13x13x256 -> 13x13x384
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 13x13x384 -> 13x13x384
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 13x13x384 -> 13x13x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13x256 -> 6x6x256
        )
        # -- fully connected layers, with no dropout
        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),  # for non-linearity
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        self.__init_bias()  # initialize bias

    def forward(self, x):
        assert x.shape[2] == self.in_size, "Input must be 227x227"

        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # flatten dimension, x.size(0) is batch size, -1 is inferred
        x = self.fc_layer(x)
        return x

    def __init_bias(self):
        for layer in self.conv_layer:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.conv_layer[4].bias, 1)
        nn.init.constant_(self.conv_layer[10].bias, 1)
        nn.init.constant_(self.conv_layer[12].bias, 1)
