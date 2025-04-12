import logging
from typing import Any

import torch

from src.models.basemodel import MetricKey

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model, val_dl) -> dict[str, float]:
    model.eval()  # set model to evaluate mode
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer) -> list[float] | None:
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer: torch.optim.Optimizer,
    reduce_lr_on_plateau: Any,
    epochs: int = 2,
    device_type: str = "cuda",
    dtype=torch.float16,
) -> list[dict[str, float]]:
    log.info(f"Training on {device_type}")
    torch.cuda.empty_cache()
    history = []
    result = {}

    scaler = torch.GradScaler()

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        masks_dice_sc = []
        train_accuracies = []
        cls_loss = []
        masks_losses = []
        lrs = []
        total_losses = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            # Runs the forward pass with autocasting.
            with torch.autocast(device_type=device_type, dtype=dtype):
                step_output = model.training_step(batch)
                seg_loss = step_output[f"{MetricKey.SEG_LOSS.value}"]
                seg_dice = step_output[f"{MetricKey.SEG_DICE.value}"]

                cls_acc = step_output[f"{MetricKey.CLS_ACC.value}"]
                cls_loss = step_output[f"{MetricKey.CLS_LOSS.value}"]

            # Detach loss and accuracy before appending to avoid holding onto computation graph
            train_losses.append(cls_loss.detach())
            masks_losses.append(seg_loss.detach())  # Assuming train_acc is a tensor
            masks_dice_sc.append(seg_dice)  # Assuming train_acc is a tensor
            train_accuracies.append(cls_acc.detach())  # Assuming train_acc is a tensor

            total_loss = step_output[f"{MetricKey.TOTAL_LOSS.value}"]
            total_losses += total_loss
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(total_loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Record & update learning rate
            lrs.append(get_lr(optimizer))

            # Updates the scale for next iteration.
            scaler.update()

        # Validation Phase
        result = evaluate(model, validation_dataloader)
        result[f"{MetricKey.TOTAL_LOSS.value}"] = total_losses
        result[f"{MetricKey.LR.value}"] = lrs[0]
        result[f"{MetricKey.CLS_LOSS.value}"] = torch.stack(train_losses).mean().item()
        result[f"{MetricKey.SEG_LOSS.value}"] = torch.stack(masks_losses).mean().item()
        result[f"{MetricKey.SEG_DICE.value}"] = torch.stack(masks_dice_sc).mean().item()
        result[f"{MetricKey.CLS_ACC.value}"] = torch.stack(train_accuracies).mean().item()

        reduce_lr_on_plateau.step(result[f"{MetricKey.VAL_SEG_LOSS.value}"])

        model.epoch_end(epoch, result)
        history.append(result)
    log.info("Finished Training")
    return history
