import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Simple classification head using Adaptive Average Pooling.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),  # Pool features to 1x1
            nn.Flatten(),
            nn.Linear(in_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature map (e.g., from encoder's pool5).

        Returns:
            torch.Tensor: Classification logits.
        """
        return self.head(x)
