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


def multiclass_soft_dice_loss(y_pred, y_true, epsilon=1e-5) -> torch.Tensor:
    num_classes = y_pred.shape[1]
    dice_sum = 0
    dice_scores = []
    for class_index in range(num_classes):
        # class_true = y_true[:, class_index, ...]
        class_pred = y_pred[:, class_index, ...]
        intersection = (y_true * class_pred).sum()
        union = y_true.sum() + class_pred.sum()
        dice_score = (2.0 * intersection + epsilon) / (union + epsilon)
        dice_sum += 1 - dice_score
        dice_scores.append(dice_score)
    dice_scores = torch.stack(dice_scores).mean()
    return (dice_scores, dice_sum / num_classes)


class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_numerator = 2 * torch.sum(y_true * y_pred)
        dice_denominator = torch.sum(y_true**2) + torch.sum(y_pred**2)
        dice_score = (dice_numerator + self.epsilon) / (dice_denominator + self.epsilon)
        return 1 - dice_score


class MultiClassSoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        num_classes = y_pred.shape[1]
        dice_sum = 0

        for class_index in range(num_classes):
            # class_true = y_true[:, class_index, ...] # y_true is already B, 1, H, W
            class_pred = y_pred[:, class_index, ...]
            intersection = (y_true * class_pred).sum()
            union = y_true.sum() + class_pred.sum()
            dice_score = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
            dice_sum += 1 - dice_score

        return dice_sum / num_classes
