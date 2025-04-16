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


# Example logic from previous suggestion
def calculate_per_sample_dice(pred_prob, true_mask, smooth=1e-6) -> torch.Tensor:
    pred_mask = (pred_prob > 0.5).float()
    true_mask = true_mask.float()
    # Flatten spatial dimensions H, W but keep batch dimension B separate
    pred_flat = pred_mask.view(pred_mask.shape[0], -1)  # Shape: [B, H*W]
    true_flat = true_mask.view(true_mask.shape[0], -1)  # Shape: [B, H*W]

    # Sum along the flattened spatial dimension (dim=1)
    intersection = (pred_flat * true_flat).sum(dim=1)  # Shape: [B]
    denominator = pred_flat.sum(dim=1) + true_flat.sum(dim=1)  # Shape: [B]
    dice = (2.0 * intersection + smooth) / (denominator + smooth)  # Shape: [B]
    return dice  # Returns a tensor of Dice scores, one for each sample in the batch
