import logging
import os

import hydra
import mlflow
import numpy as np
import pyrootutils
import torch
import torchmetrics
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report
from torch.nn import functional as F
from tqdm.auto import tqdm

from utils.visualizations import plot_confusion_matrix

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

if os.getenv("DATA_ROOT") is None:
    os.environ["DATA_ROOT"] = ""

from src.losses.dice_loss import dice_coefficient_metric
from src.utils.gpu_utils import DeviceDataLoader, get_default_device, to_device

log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DeviceDataLoader,
    device: torch.device,
    class_names: list[str],
    seg_threshold: float = 0.5,
    cls_weights=None,
    num_samples_to_plot: int = 9,  # <-- Add parameter for number of samples
) -> tuple[dict[str, float], str, np.ndarray, np.ndarray | None]:
    """
    Evaluates the model on the test dataset and returns segmentation and
    classification metrics.

    Args:
        model: The trained PyTorch model to evaluate.
        test_loader: DataLoader containing the test dataset, wrapped with DeviceDataLoader.
        device: The torch device (e.g., 'cuda' or 'cpu') to perform evaluation on.
        num_classes: The number of classes for the classification task.
        seg_threshold: The threshold to convert segmentation probabilities to binary masks.

    Returns:
        A dictionary containing the calculated metrics:
        - 'test_dice': Dice score for segmentation.
        - 'test_iou': Intersection over Union (IoU) for segmentation.
        - 'test_accuracy': Accuracy for classification.
        - 'test_precision': Precision for classification.
        - 'test_recall': Recall for classification.
        - 'test_f1': F1-score for classification.
        - 'test_auc': Area Under the ROC Curve (AUC) for classification.
    """
    log.info("Starting model evaluation on the test set...")
    num_classes = len(class_names)
    # --- Initialize Metrics ---
    # Segmentation Metrics
    iou_metric = torchmetrics.JaccardIndex(task="binary", threshold=seg_threshold).to(
        device
    )  # Assuming binary segmentation

    # Classification Metrics (adjust task based on your problem: binary, multiclass, multilabel)
    # Assuming binary classification for this example. Change 'task' and 'num_classes' if needed.
    task = "multiclass"
    cls_criterion = torch.nn.CrossEntropyLoss(weight=cls_weights)
    accuracy_metric = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(device)
    auc_metric = torchmetrics.AUROC(task=task, num_classes=num_classes).to(device)

    # --- Evaluation Loop ---
    dice_metric = []
    all_preds = []
    all_labels = []
    with torch.no_grad():  # Disable gradient calculations
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            images, targets = batch
            masks_true, labels_true = targets["masks"], targets["labels"]

            # Forward pass
            outputs = model(images)
            masks_pred_logits = outputs["masks"]
            labels_pred_logits = outputs["labels"]

            # --- Update Segmentation Metrics ---
            # Apply sigmoid for binary
            masks_pred_prob = F.sigmoid(masks_pred_logits)

            dice_metric_value = dice_coefficient_metric(masks_pred_prob, masks_true)
            iou_metric.update(masks_pred_prob, masks_true.long())

            # --- Update Classification Metrics ---
            accuracy_metric.update(labels_pred_logits, labels_true)
            auc_metric.update(labels_pred_logits, labels_true)

            # --- Collect data for sklearn classification_report ---
            # Convert logits to class predictions
            preds = torch.argmax(labels_pred_logits, dim=1)

            # Move predictions and true labels for this batch to CPU and store them
            all_preds.append(preds.cpu())
            all_labels.append(labels_true.cpu())
            dice_metric.append(dice_metric_value)

    # --- Compute Final Metrics ---
    test_dice = sum(dice_metric) / len(dice_metric)
    test_iou = iou_metric.compute().item()
    test_accuracy = accuracy_metric.compute().item()
    test_auc = auc_metric.compute().item()

    # --- Prepare data for sklearn ---
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    # --- Generate sklearn Classification Report ---
    log.info("Generating classification report...")
    # Ensure target_names match the number of classes if provided

    # Generate the report
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=False
    )

    log.info(f"\n  {report_str}")

    metrics = {
        "test_dice": test_dice.item(),
        "test_iou": test_iou,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
    }

    log.info("Evaluation finished.")
    log.info(f"Test Metrics: {metrics}")

    # Reset metrics for potential future use
    iou_metric.reset()
    accuracy_metric.reset()
    auc_metric.reset()

    return metrics, report_str, y_true, y_pred


if __name__ == "__main__":
    # Register a resolver for torch dtypes
    OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))
    with initialize(config_path="../configs", job_name="training_setup", version_base=None):
        cfg: DictConfig = compose(config_name="train.yaml")
    device = get_default_device()

    model_uri = "runs:/d5bffd078a4448bc895872a7af270dd0/model"

    model = mlflow.pytorch.load_model(model_uri)
    model = to_device(model, device)
    data_module = hydra.utils.instantiate(cfg.datamodule)
    test_dl = data_module.test_dataloader()  # Get the raw test dataloader
    test_loader = DeviceDataLoader(test_dl, device)  # Wrap it
    num_classes = len(data_module.classes)

    (test_metrics, cls_report, y_true, y_pred) = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        seg_threshold=0.5,
        class_names=data_module.classes,
        cls_weights=data_module.class_weights,
    )
    # roc_curve=plot_roc_curve(y_true, y_pred)
    cm_fig_norm = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=data_module.classes,
    )
    # print(test_metrics)
    # print(cls_report)
