import logging
from typing import Any

import mlflow
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
    reduce_lr_on_plateau: Any | None = None,
    epochs: int = 2,
    device_type: str = "cuda",
    dtype=torch.float16,
) -> list[dict[str, float]]:
    log.info(f"Training on {device_type}")

    torch.cuda.empty_cache()
    history = []

    scaler = torch.GradScaler()

    for epoch in range(epochs):
        # Training Phase
        model.train()
        # Initialize lists for all training metrics for the current epoch
        train_seg_losses = []
        train_seg_dices = []
        train_cls_losses = []
        train_cls_accs = []
        train_cls_aurocs = []
        train_total_losses = []  # Store individual batch total losses for averaging

        current_lr = get_lr(optimizer)  # Get LR at the start of the epoch

        for batch in train_dataloader:
            optimizer.zero_grad(
                set_to_none=True
            )  # Use set_to_none=True for potential performance improvement
            # Runs the forward pass with autocasting.
            with torch.autocast(device_type=device_type, dtype=dtype):
                step_output = model.training_step(batch)
                total_loss = step_output[f"{MetricKey.TOTAL_LOSS.value}"]

            # Detach loss and accuracy before appending to avoid holding onto computation graph
            train_seg_losses.append(step_output[MetricKey.SEG_LOSS.value].detach())
            train_seg_dices.append(
                step_output[MetricKey.SEG_DICE.value].detach()
            )  # Dice is often calculated without grads already
            train_cls_losses.append(step_output[MetricKey.CLS_LOSS.value].detach())
            train_cls_accs.append(step_output[MetricKey.CLS_ACC.value].detach())
            train_cls_aurocs.append(step_output[MetricKey.CLS_AUROC.value].detach())
            train_total_losses.append(total_loss.detach())  # Detach total loss per batch

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(total_loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

        # Validation Phase
        val_results = evaluate(model, validation_dataloader)

        # Calculate average training metrics for the epoch
        epoch_results = {}
        epoch_results[f"{MetricKey.SEG_LOSS.value}"] = torch.stack(train_seg_losses).mean().item()
        epoch_results[f"{MetricKey.SEG_DICE.value}"] = torch.stack(train_seg_dices).mean().item()
        epoch_results[f"{MetricKey.CLS_LOSS.value}"] = torch.stack(train_cls_losses).mean().item()
        epoch_results[f"{MetricKey.CLS_ACC.value}"] = torch.stack(train_cls_accs).mean().item()
        epoch_results[f"{MetricKey.CLS_AUROC.value}"] = torch.stack(train_cls_aurocs).mean().item()
        epoch_results[f"{MetricKey.TOTAL_LOSS.value}"] = (
            torch.stack(train_total_losses).mean().item()
        )  # Average total loss
        epoch_results[MetricKey.LR.value] = current_lr  # Log LR for this epoch

        # Add validation results (which should already have 'val_' prefix from validation_epoch_end)
        epoch_results.update(val_results)

        if reduce_lr_on_plateau:
            reduce_lr_on_plateau.step(epoch_results[f"{MetricKey.VAL_SEG_LOSS.value}"])

        # Log all collected metrics for this epoch to MLflow
        # MLflow expects a flat dictionary of metric names to scalar values
        mlflow.log_metrics(epoch_results, step=epoch)

        model.epoch_end(epoch, epoch_results)
        history.append(epoch_results)

    log.info("Finished Training")
    return history
