import torch
import torch.nn as nn


def dice_coefficient_metric(y_pred, y_true, epsilon=1e-5) -> torch.Tensor:
    # Ensure inputs are float type
    y_pred = y_pred.float()
    y_true = y_true.float()

    dice_numerator = 2 * torch.sum(y_true * y_pred)
    dice_denominator = torch.sum(y_true) + torch.sum(y_pred)
    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    return dice_score


def soft_dice_loss(y_pred, y_true, epsilon=1e-5) -> torch.Tensor:
    return 1 - dice_coefficient_metric(y_pred, y_true, epsilon)


class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_numerator = 2 * torch.sum(y_true * y_pred)
        dice_denominator = torch.sum(y_true**2) + torch.sum(y_pred**2)
        dice_score = (dice_numerator + self.epsilon) / (dice_denominator + self.epsilon)
        return 1 - dice_score
