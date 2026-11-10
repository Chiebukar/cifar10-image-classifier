
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with an optional projection shortcut
    when the input and output dimensions differ.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut handles changes in dimensions.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class CustomResidualCNN(nn.Module):
    """
    Custom residual CNN for CIFAR-10 classification.

    Input:  (batch_size, 3, 32, 32)
    Output: (batch_size, 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(
            in_channels=64,
            out_channels=64,
            num_blocks=2,
            first_stride=1,
        )

        self.stage2 = self._make_stage(
            in_channels=64,
            out_channels=128,
            num_blocks=2,
            first_stride=2,
        )

        self.stage3 = self._make_stage(
            in_channels=128,
            out_channels=256,
            num_blocks=2,
            first_stride=2,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
    ) -> nn.Sequential:

        layers = [
            ResidualBlock(
                in_channels,
                out_channels,
                stride=first_stride,
            )
        ]

        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.pool(x)

        return self.classifier(x)
