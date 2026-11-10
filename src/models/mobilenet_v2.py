
import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    mobilenet_v2,
)


class MobileNetV2Classifier(nn.Module):
    """ImageNet-pretrained MobileNetV2 adapted for CIFAR-10."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = (
            MobileNet_V2_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = mobilenet_v2(weights=weights)

        input_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            input_features,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
