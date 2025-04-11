import enum
import logging

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import AUROC

from src.losses.dice_loss import dice_coefficient_metric

log = logging.getLogger(__name__)


class MetricKey(enum.Enum):
    """Keys for metrics dictionary."""

    # Classification
    CLS_LOSS = "cls_loss"
    CLS_ACC = "cls_acc"
    CLS_AUROC = "cls_auroc"
    VAL_CLS_LOSS = "val_cls_loss"
    VAL_CLS_ACC = "val_cls_acc"
    VAL_CLS_AUROC = "val_cls_auroc"

    # Segmentation
    SEG_LOSS = "seg_loss"
    SEG_DICE = "seg_dice"
    VAL_SEG_LOSS = "val_seg_loss"
    VAL_SEG_DICE = "val_seg_dice"

    # Combined/Overall (Optional)
    # TOTAL_LOSS = "total_loss"
    # VAL_TOTAL_LOSS = "val_total_loss"


def accuracy(preds: Tensor, labels: Tensor) -> Tensor:
    """Calculates classification accuracy."""

    pred_labels = torch.argmax(preds, dim=1)
    correct = torch.sum(pred_labels == labels).item()
    total = len(pred_labels)
    return torch.tensor(correct / total, device=preds.device)


class SegmentationBaseModel(nn.Module):
    """
    Base model for combined image segmentation and classification tasks.

    Handles the basic training and validation steps, loss calculation,
    and metric aggregation. Subclasses should implement the actual
    network architecture in the `forward` method.
    """

    def __init__(
        self,
        segmentation_criterion: nn.Module,
        classification_criterion: nn.Module,
    ) -> None:
        """
        Initializes the BaseModel.

        Args:
            segmentation_criterion: Loss function for segmentation (e.g., DiceLoss, CrossEntropyLoss).
            classification_criterion: Loss function for classification (e.g., CrossEntropyLoss).
            class_weights: Optional weights for the classification loss.
        """
        super().__init__()

        if segmentation_criterion is None or classification_criterion is None:
            raise ValueError("Both segmentation and classification criteria must be provided.")

        self.segmentation_criterion = segmentation_criterion
        self.classification_criterion = classification_criterion
        self.cls_auroc = AUROC(task="multiclass", num_classes=3)
        log.info(f"Segmentation Criterion: {type(self.segmentation_criterion).__name__}")
        log.info(f"Classification Criterion: {type(self.classification_criterion).__name__}")

    def _prepare_gt_masks(self, masks: Tensor) -> Tensor:
        """
        Ensures ground truth masks are in the correct shape [B, H, W]
        and type torch.long for loss calculation.
        """
        if masks.ndim == 4 and masks.shape[1] == 1:
            # Input shape [B, 1, H, W] -> Output shape [B, H, W]
            processed_masks = masks.squeeze(1)
        elif masks.ndim == 3:
            # Input shape [B, H, W] -> Output shape [B, H, W]
            processed_masks = masks
        else:
            raise ValueError(
                f"Unexpected ground truth masks shape: {masks.shape}. "
                f"Expected [B, 1, H, W] or [B, H, W]."
            )

        # Ensure dtype is suitable for criteria like CrossEntropyLoss
        # Adjust if your criterion requires a different type (e.g., float for BCE)
        return processed_masks.long()

    def _calculate_segmentation_metrics(
        self, seg_logits: Tensor, gt_masks: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Calculates segmentation loss and Dice score.

        Args:
            seg_logits: Raw logits from the model's segmentation head ([B, C, H, W]).
            gt_masks: Ground truth masks, prepared by _prepare_gt_masks ([B, H, W]).

        Returns:
            A tuple containing (segmentation_loss, dice_score).
        """
        seg_logits = F.sigmoid(seg_logits)
        seg_loss = self.segmentation_criterion(seg_logits, gt_masks)

        # Ensure dice_coefficient_metric handles multi-class correctly if needed
        dice_score = dice_coefficient_metric(seg_logits, gt_masks)

        return seg_loss, dice_score

    def _calculate_classification_metrics(
        self, cls_logits: Tensor, gt_labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Calculates classification loss and accuracy.

        Args:
            cls_logits: Raw logits from the model's classification head ([B, NumClasses]).
            gt_labels: Ground truth labels ([B]).

        Returns:
            A tuple containing (classification_loss, classification_accuracy).
        """
        cls_loss = self.classification_criterion(cls_logits, gt_labels)
        cls_acc = accuracy(cls_logits, gt_labels)
        cls_auroc = self.cls_auroc(cls_logits, gt_labels)
        return cls_loss, cls_acc, cls_auroc

    def _shared_step(self, batch: tuple[Tensor, dict[str, Tensor]]) -> dict[str, Tensor]:
        """Performs a forward pass and calculates losses/metrics."""
        images, targets = batch

        # --- Forward Pass ---
        # Assumes the model's forward pass returns a dictionary
        # with keys 'masks' (segmentation logits) and 'labels' (classification logits)
        outputs = self(images)
        seg_logits = outputs["masks"]
        cls_logits = outputs["labels"]

        # --- Prepare Ground Truth ---
        gt_labels = targets["labels"]  # Shape [B]
        # gt_masks = self._prepare_gt_masks(targets["masks"])  # Shape [B, H, W]
        gt_masks = targets["masks"]  # Shape [B, 1, H, W]
        # --- Calculate Metrics ---
        seg_loss, seg_dice = self._calculate_segmentation_metrics(seg_logits, gt_masks)
        cls_loss, cls_acc, cls_auroc = self._calculate_classification_metrics(
            cls_logits, gt_labels
        )

        return {
            "seg_loss": seg_loss,
            "seg_dice": seg_dice,
            "cls_loss": cls_loss,
            "cls_acc": cls_acc,
            "cls_auroc": cls_auroc,
            # Optional: Combine losses if needed for backpropagation
            # "total_loss": seg_loss + cls_loss # Example weighting
        }

    def training_step(self, batch: tuple[Tensor, dict[str, Tensor]]) -> dict[str, Tensor]:
        """Performs one training step."""
        metrics = self._shared_step(batch)

        # Return metrics needed for logging or backpropagation
        return {
            MetricKey.SEG_LOSS.value: metrics["seg_loss"],
            MetricKey.SEG_DICE.value: metrics["seg_dice"],
            MetricKey.CLS_LOSS.value: metrics["cls_loss"],
            MetricKey.CLS_ACC.value: metrics["cls_acc"],
            MetricKey.CLS_AUROC.value: metrics["cls_auroc"],
            # Return total loss if the optimizer needs it directly
            # MetricKey.TOTAL_LOSS.value: metrics["total_loss"]
        }

    def validation_step(self, batch: tuple[Tensor, dict[str, Tensor]]) -> dict[str, Tensor]:
        """Performs one validation step."""
        metrics = self._shared_step(batch)

        # Return detached metrics for aggregation
        return {
            MetricKey.VAL_SEG_LOSS.value: metrics["seg_loss"].detach(),
            MetricKey.VAL_SEG_DICE.value: metrics["seg_dice"].detach(),
            MetricKey.VAL_CLS_LOSS.value: metrics["cls_loss"].detach(),
            MetricKey.VAL_CLS_ACC.value: metrics["cls_acc"].detach(),
            MetricKey.VAL_CLS_AUROC.value: metrics["cls_auroc"].detach(),
        }

    def validation_epoch_end(self, outputs: list[dict[str, Tensor]]) -> dict[str, float]:
        """Aggregates metrics from validation steps across an epoch."""
        agg_metrics = {}
        # Aggregate each metric type
        for key in MetricKey:
            # Check if the validation version of the key exists in the outputs
            val_key_str = key.value
            if key.value.startswith("val_"):  # Already a validation key
                pass
            elif key.value.startswith("cls_") or key.value.startswith("seg_"):
                val_key_str = f"val_{key.value}"  # Construct validation key
            else:
                continue  # Skip non-metric keys or keys without val_ prefix convention

            # Check if this validation key was produced by validation_step
            if val_key_str in outputs[0]:
                # Stack tensors for the current metric and calculate the mean
                metric_values = torch.stack([x[val_key_str] for x in outputs])
                agg_metrics[val_key_str] = torch.mean(metric_values).item()

        return agg_metrics

    def epoch_end(self, epoch: int, results: dict[str, float]) -> None:
        """Logs aggregated results at the end of an epoch."""
        # Example logging - customize as needed
        log_message = f"Epoch [{epoch}] Validation Results: "
        log_items = []
        # Use MetricKey to ensure consistent naming
        if MetricKey.CLS_LOSS.value in results:
            log_items.append(f"{MetricKey.CLS_LOSS.value}={results[MetricKey.CLS_LOSS.value]:.4f}")
        if MetricKey.VAL_CLS_LOSS.value in results:
            log_items.append(
                f"{MetricKey.VAL_CLS_LOSS.value}={results[MetricKey.VAL_CLS_LOSS.value]:.4f}"
            )
        if MetricKey.CLS_ACC.value in results:
            log_items.append(f"{MetricKey.CLS_ACC.value}={results[MetricKey.CLS_ACC.value]:.4f}")
        if MetricKey.VAL_CLS_ACC.value in results:
            log_items.append(
                f"{MetricKey.VAL_CLS_ACC.value}={results[MetricKey.VAL_CLS_ACC.value]:.4f}"
            )
        if MetricKey.CLS_AUROC.value in results:
            log_items.append(
                f"{MetricKey.CLS_AUROC.value}={results[MetricKey.CLS_AUROC.value]:.4f}"
            )

        if MetricKey.VAL_CLS_AUROC.value in results:
            log_items.append(
                f"{MetricKey.VAL_CLS_AUROC.value}={results[MetricKey.VAL_CLS_AUROC.value]:.4f}"
            )

        if MetricKey.SEG_LOSS.value in results:
            log_items.append(f"{MetricKey.SEG_LOSS.value}={results[MetricKey.SEG_LOSS.value]:.4f}")
        if MetricKey.VAL_SEG_LOSS.value in results:
            log_items.append(
                f"{MetricKey.VAL_SEG_LOSS.value}={results[MetricKey.VAL_SEG_LOSS.value]:.4f}"
            )
        if MetricKey.SEG_DICE.value in results:
            log_items.append(f"{MetricKey.SEG_DICE.value}={results[MetricKey.SEG_DICE.value]:.4f}")
        if MetricKey.VAL_SEG_DICE.value in results:
            log_items.append(
                f"{MetricKey.VAL_SEG_DICE.value}={results[MetricKey.VAL_SEG_DICE.value]:.4f}"
            )
        log_message += ", ".join(log_items)

        log.info(log_message)
        print(log_message)
