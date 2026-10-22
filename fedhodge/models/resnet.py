from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs):
        output = torch.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return torch.relu(output + self.shortcut(inputs))


class ResNet(nn.Module):
    def __init__(self, blocks: list[int], num_classes: int):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._layer(64, blocks[0], 1)
        self.layer2 = self._layer(128, blocks[1], 2)
        self.layer3 = self._layer(256, blocks[2], 2)
        self.layer4 = self._layer(512, blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _layer(self, channels: int, count: int, stride: int):
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(count - 1))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        output = torch.relu(self.bn1(self.conv1(inputs)))
        output = self.layer4(
            self.layer3(self.layer2(self.layer1(output)))
        )
        return self.linear(torch.flatten(self.pool(output), 1))


def ResNet18(num_classes: int = 10):
    return ResNet([2, 2, 2, 2], num_classes)
