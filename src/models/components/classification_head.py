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
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Optional dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Optional dropout
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature map (e.g., from encoder's pool5).

        Returns:
            torch.Tensor: Classification logits.
        """
        return self.head(x)
