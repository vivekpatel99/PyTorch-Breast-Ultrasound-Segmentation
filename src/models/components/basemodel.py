import logging

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)


def accuracy(preds, labels) -> torch.Tensor:
    preds_labels = torch.argmax(preds, dim=1)
    return torch.tensor(torch.sum(preds_labels == labels).item() / len(preds))


class SegmentationBaseModel(nn.Module):

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(
            out["labels"], labels["labels"]
        )  # , weight=class_weights)  # Calculate loss

        # Calculate training accuracy
        train_acc = accuracy(out["labels"], labels["labels"])

        # Return loss and accuracy in a dictionary
        return {"loss": loss, "train_acc": train_acc}

    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(
            out["labels"],
            labels["labels"],
        )  # weight=class_weights)  # Calculate loss

        val_acc = accuracy(out["labels"], labels["labels"])  # Calculate accuracy

        return {"val_loss": loss.detach(), "val_acc": val_acc.detach()}

    def validation_epoch_end(self, outputs) -> dict[str, float]:
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()  # combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result) -> None:
        message = f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, train_acc: {result['train_acc']:.4f}, val_acc: {result['val_acc']:.4f}"
        log.info(message)
        print(message)
