import enum
import logging

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)


class MetricKey(enum.Enum):
    # classification
    CLS_LOSS = "cls_loss"
    CLS_TRAIN_ACC = "cls_train_acc"
    VAL_CLS_LOSS = "cls_val_loss"
    VAL_CLS_ACC = "cls_val_acc"

    # segmentation
    MASKS_LOSS = "masks_loss"
    MASKS_DICE_SCORE = "masks_dice_score"
    VAL_MASK_LOSS = "val_mask_loss"
    VAL_DICE_SCORE = "val_dice_score"


def accuracy(preds, labels) -> torch.Tensor:
    preds_labels = torch.argmax(preds, dim=1)
    return torch.tensor(torch.sum(preds_labels == labels).item() / len(preds))


class SegmentationBaseModel(nn.Module):
    def __init__(
        self,
        segmentation_criterion=None,
        classification_criterion=None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.segmentation_criterion = segmentation_criterion
        self.classification_criterion = classification_criterion

        log.info(f"Class weights: {self.class_weights}")

    def process_pred_masks_logits(self, logits, threshold=0.5):
        # Apply sigmoid to convert logits to probabilities
        probabilities = F.softmax(logits, dim=1)  # torch.sigmoid(logits)

        # Convert to binary prediction
        binary_prediction = (probabilities > threshold).float()
        return binary_prediction

    # def segmentation_metrics(self, out:torch.Tensor, labels:torch.Tensor) -> torch.Tensor:
    #     pred_masks =  self.process_pred_masks_logits(out)
    #     pred_masks = torch.argmax(pred_masks,dim=1)
    #     labels = self.preprocess_gt_masks(labels)
    #     score = dice_coefficient_metric(pred_masks, labels)
    #     return score

    def segmentation_loss(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ):
        pred_masks = self.process_pred_masks_logits(pred_mask)
        # dice_score, loss = multiclass_soft_dice_loss(pred_masks, labels)
        # for class_index in range(len(class_weights)):
        #     class_true = gt_mask[:, class_index, ...]
        #     class_pred = pred_masks[:, class_index, ...]

        loss = F.cross_entropy(pred_masks, gt_mask)

        return torch.Tensor(0), loss

    def classification_loss(
        self, out: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        return F.cross_entropy(out, labels, weight=class_weights)

    def process_gt_mask_for_loss(self, labels) -> torch.Tensor:
        # Ensure labels are in the correct shape [B, H, W] and type torch.long
        if labels.ndim == 4 and labels.shape[1] == 1:
            squeezed_labels = labels.squeeze(1)
        elif labels.ndim == 3:
            squeezed_labels = labels
        else:
            # Handle unexpected label dimensions if necessary
            raise ValueError(
                f"Unexpected labels shape: {labels.shape}. Expected [B, 1, H, W] or [B, H, W]"
            )

        return squeezed_labels.long()

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        images, labels = batch
        out = self(images)  # Generate predictions

        # Classification
        cls_loss = self.classification_loss(out["labels"], labels["labels"], self.class_weights)
        cls_train_acc = accuracy(out["labels"], labels["labels"])

        # Segmentation
        pred_masks = out["masks"]
        gt_masks = labels["masks"]

        # Convert labels to torch.long dtype
        gt_masks = self.process_gt_mask_for_loss(gt_masks)
        masks_dice_score, masks_loss = self.segmentation_loss(
            pred_masks, gt_masks, self.class_weights
        )
        # masks_dice_score = self.segmentation_metrics(pred_masks, gt_masks )

        # Return loss and accuracy in a dictionary
        return {
            MetricKey.MASKS_LOSS.value: masks_loss,
            MetricKey.MASKS_DICE_SCORE.value: masks_dice_score,
            MetricKey.CLS_LOSS.value: cls_loss,
            MetricKey.CLS_TRAIN_ACC.value: cls_train_acc,
        }

    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        images, labels = batch
        out = self(images)  # Generate predictions

        # Classification
        cls_loss = self.classification_loss(out["labels"], labels["labels"], self.class_weights)
        val_acc = accuracy(out["masks"], labels["masks"])  # Calculate accuracy

        # Segmentation
        pred_masks = out["masks"]
        gt_masks = labels["masks"]
        gt_masks = self.process_gt_mask_for_loss(gt_masks)
        masks_dice_score, masks_loss = self.segmentation_loss(
            pred_masks, gt_masks, self.class_weights
        )
        # masks_dice_score = self.segmentation_metrics(pred_masks, gt_masks )

        return {
            MetricKey.VAL_MASK_LOSS.value: masks_loss.detach(),
            MetricKey.VAL_DICE_SCORE.value: masks_dice_score,
            MetricKey.VAL_CLS_ACC.value: val_acc.detach(),
            MetricKey.VAL_CLS_LOSS.value: cls_loss.detach(),
        }

    def validation_epoch_end(self, outputs) -> dict[str, float]:
        # batch_cls_losses = [x["val_loss"] for x in outputs]
        batch_cls_losses = [x[f"{MetricKey.VAL_CLS_LOSS.value}"] for x in outputs]
        epoch_cls_loss = torch.stack(batch_cls_losses).mean()  # combine losses
        batch_mask_loss = [x[f"{MetricKey.VAL_MASK_LOSS.value}"] for x in outputs]
        # batch_mask_loss = [x["val_mask_loss"] for x in outputs]
        epoch_mask_loss = torch.stack(batch_mask_loss).mean()  # combine mask losses
        batch_cls_acc = [x[f"{MetricKey.VAL_CLS_ACC.value}"] for x in outputs]
        # batch_cls_acc = [}"] for x in outputs]
        # batch_cls_acc = [x["val_acc"] for x in outputs]
        epoch_cls_acc = torch.stack(batch_cls_acc).mean()  # combine accuracies
        # batch_val_dice_score = [x["val_dic_score"] for x in outputs]
        batch_val_dice_score = [x[f"{MetricKey.VAL_DICE_SCORE.value}"] for x in outputs]
        # batch_val_dice_score = [}"] for x in outputs]
        epoch_val_dice_score = torch.stack(batch_val_dice_score).mean()  # combine iou scores

        return {
            MetricKey.VAL_MASK_LOSS.value: epoch_mask_loss.item(),
            MetricKey.VAL_DICE_SCORE.value: epoch_val_dice_score.item(),
            MetricKey.VAL_CLS_ACC.value: epoch_cls_acc.item(),
            MetricKey.VAL_CLS_LOSS.value: epoch_cls_loss.item(),
        }

        # return {
        #     "val_loss": epoch_loss.item(),
        #     "val_mask_loss": epoch_mask_loss.item(),
        #     "val_acc": epoch_acc.item(),
        #     "val_dice_score": epoch_iou.item(),
        # }

    def epoch_end(self, epoch, result) -> None:
        # print(result.keys())
        message = (
            f"Epoch [{epoch}], "
            f"masks_loss: {result[f'{MetricKey.MASKS_LOSS.value}']:.4f},"
            f"masks_dice_score: {result[f'{MetricKey.MASKS_DICE_SCORE.value}']:.4f},"
        )  # message = f"Epoch [{epoch}], " \
        #                 f"masks_loss: {result['masks_loss']:.4f},"\
        #                 f"val_loss: {result['val_loss']:.4f}," \
        #                 f"val_mask_loss: {result['val_mask_loss']:.4f}," \
        #                 f"masks_dice_score: {result['masks_dice_score']:.4f}," \
        #                 f"val_acc: {result['val_acc']:.4f}"
        log.info(message)
        print(message)
