import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.batchnorm(self.conv(x))
        x = self.relu(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool
    ) -> None:
        super().__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1_pool, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class GoogleNet(nn.Module):
    """GoogleNet network based on:

    https://www.youtube.com/watch?v=uQc4Fs7yx5I&ab_channel=AladdinPersson.
    """

    def __init__(
        self,
        in_channel: int = 3,
        im_size: int = 224,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.im_size = im_size  # image size input
        self.in_channel = in_channel
        self.num_classes = num_classes

        # -- convolution layers
        self.conv1 = ConvBlock(
            in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -- inception layer
        self.inception3a = InceptionBlock(
            in_channels=192,
            out_1x1=64,
            red_3x3=96,
            out_3x3=128,
            red_5x5=16,
            out_5x5=32,
            out_1x1_pool=32,
        )
        self.inception3b = InceptionBlock(
            in_channels=256,
            out_1x1=128,
            red_3x3=128,
            out_3x3=192,
            red_5x5=32,
            out_5x5=96,
            out_1x1_pool=64,
        )  # in_channels=out_1x1_pool+out_3x3+out_5x5+out_1x1
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(
            in_channels=480,
            out_1x1=192,
            red_3x3=96,
            out_3x3=208,
            red_5x5=16,
            out_5x5=48,
            out_1x1_pool=64,
        )
        self.inception4b = InceptionBlock(
            in_channels=512,
            out_1x1=160,
            red_3x3=112,
            out_3x3=224,
            red_5x5=24,
            out_5x5=64,
            out_1x1_pool=64,
        )
        self.inception4c = InceptionBlock(
            in_channels=512,
            out_1x1=128,
            red_3x3=128,
            out_3x3=256,
            red_5x5=24,
            out_5x5=64,
            out_1x1_pool=64,
        )
        self.inception4d = InceptionBlock(
            in_channels=512,
            out_1x1=112,
            red_3x3=144,
            out_3x3=288,
            red_5x5=32,
            out_5x5=64,
            out_1x1_pool=64,
        )
        self.inception4e = InceptionBlock(
            in_channels=528,
            out_1x1=256,
            red_3x3=160,
            out_3x3=320,
            red_5x5=32,
            out_5x5=128,
            out_1x1_pool=128,
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(
            in_channels=832,
            out_1x1=256,
            red_3x3=160,
            out_3x3=320,
            red_5x5=32,
            out_5x5=128,
            out_1x1_pool=128,
        )
        self.inception5b = InceptionBlock(
            in_channels=832,
            out_1x1=384,
            red_3x3=192,
            out_3x3=384,
            red_5x5=48,
            out_5x5=128,
            out_1x1_pool=128,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        # -- fully connected layers, with no dropout
        self.fc_layer = nn.Linear(1024, num_classes)

    def forward(self, x):
        assert x.shape[2] == self.im_size, "Input must be of size {}, but is {}".format(
            self.in_size, x.shape[2]
        )

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_layer(x)

        return x
