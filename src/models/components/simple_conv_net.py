import torch
from torch import nn
from torch.nn import functional as F


class MyConvNet(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        img_channel: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=img_channel, out_channels=16, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        _, channels, width, height = self.__check_final_conv_size(
            torch.rand(1, img_channel, img_size, img_size)
        )

        self.fc1 = nn.Linear(in_features=channels * width * height, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def __check_final_conv_size(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))

        return x.size()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
