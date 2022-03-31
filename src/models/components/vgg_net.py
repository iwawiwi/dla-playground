import torch
from torch import nn


class VGG16Net(nn.Module):
    """VGG16 network based on: https://blog.paperspace.com/popular-deep-learning-architectures-
    alexnet-vgg-googlenet/ and
    https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py."""

    def __init__(
        self,
        in_channel: int = 3,
        im_size: int = 224,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.in_channel = in_channel
        self.num_classes = num_classes
        self.im_size = im_size

        # -- convolution layers, without batch normalization
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),  # 28x28x3 -> 28x28x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 28x28x64 -> 28x28x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28x64 -> 14x14x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 14x14x64 -> 14x14x128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 14x14x128 -> 14x14x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14x128 -> 7x7x128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 7x7x128 -> 7x7x256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 7x7x256 -> 7x7x256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1),  # 7x7x256 -> 7x7x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7x256 -> 4x4x256
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 4x4x256 -> 4x4x512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 -> 4x4x512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1),  # 4x4x512 -> 4x4x512, for linear transformation
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4x512 -> 2x2x512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 2x2x512 -> 2x2x512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 2x2x512 -> 2x2x512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1),  # 2x2x512 -> 2x2x512, for linear transformation
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2x512 -> 1x1x512
        )
        # -- check output size of convolution layers
        conv_out = self.__check_output_dim(self.conv_layer)
        # -- fully connected layers, with no dropout
        in_fc = conv_out[0] * conv_out[1] * conv_out[2]
        # -- get number maximum power of 2 from input size as fully connected output
        out_fc = 2 ** (in_fc.bit_length() - 2)

        self.fc_layer = nn.Sequential(
            nn.Linear(in_fc, out_fc),
            nn.ReLU(),
            nn.Linear(out_fc, out_fc),
            nn.ReLU(),
            nn.Linear(out_fc, num_classes),
        )

    def __check_output_dim(self, layer):
        x = torch.rand(1, self.in_channel, self.im_size, self.im_size)
        x = layer(x)
        return x.shape[1:]  # return shape except batch size

    def forward(self, x):
        assert x.shape[2] % 4 == 0, "image shape must be divisible by 4"

        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # flatten dimension, x.size(0) is batch size, -1 is inferred
        x = self.fc_layer(x)
        return x


class MiniVGGNet(nn.Module):
    """MiniVGG network based on: https://machinelearningmastery.com/how-to-develop-a-cnn-from-
    scratch-for-cifar-10-photo-classification/"""

    def __init__(
        self,
        in_channel: int = 3,
        im_size: int = 224,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.in_channel = in_channel
        self.num_classes = num_classes
        self.im_size = im_size

        # -- convolution layers, without batch normalization
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1),  # 32x32x3 -> 32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32x32x32 -> 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32x32 -> 16x16x32
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16x32 -> 16x16x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 16x16x64 -> 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x64 -> 8x8x64
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8x8x64 -> 8x8x128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 8x8x128 -> 8x8x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x128 -> 4x4x128
            nn.Dropout(0.2),
        )
        # -- check output size of convolution layers
        conv_out = self.__check_output_dim(self.conv_layer)
        # -- fully connected layers, with no dropout
        in_fc = conv_out[0] * conv_out[1] * conv_out[2]
        # -- get number maximum power of 2 from input size as fully connected output
        out_fc = 2 ** (in_fc.bit_length() - 2)

        self.fc_layer = nn.Sequential(
            nn.Linear(in_fc, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def __check_output_dim(self, layer):
        x = torch.rand(1, self.in_channel, self.im_size, self.im_size)
        x = layer(x)
        return x.shape[1:]  # return shape except batch size

    def forward(self, x):
        assert x.shape[2] % 4 == 0, "image shape must be divisible by 4"

        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # flatten dimension, x.size(0) is batch size, -1 is inferred
        x = self.fc_layer(x)
        return x
