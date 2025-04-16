import os
from typing import Any

import pyrootutils

from evaluate import evaluate_model
from src.utils.visualizations import (
    create_prediction_gif,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_histogram,
)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

if os.getenv("DATA_ROOT") is None:
    os.environ["DATA_ROOT"] = ""

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
import mlflow  # noqa: E402
import torch  # noqa: E402
from mlflow.models.signature import ModelSignature  # noqa: E402
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torchinfo import summary  # noqa: E402

from src.utils.gpu_utils import DeviceDataLoader  # noqa: E402
from src.utils.gpu_utils import get_default_device, to_device
from src.utils.train_utils import fit

log = logging.getLogger(__name__)

# Register a resolver for torch dtypes
OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))


def model_signature(model, train_dl) -> ModelSignature:
    # This ensures layers like Dropout and BatchNorm behave correctly for inference and saves computation.
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(train_dl))
        out = model(images)
        signature = infer_signature(
            model_input={"image_input": images.cpu().numpy()},
            model_output={
                "output": {
                    "masks": out["masks"].cpu().numpy(),
                    "labels": out["labels"].cpu().numpy(),
                }
            },
        )
    return signature


def train(cfg: DictConfig) -> tuple[dict[str, float], str, Any]:
    log.info(f"Instantiating mlflow experiment <{cfg.task_name}>")
    mlflow.set_experiment(f"{cfg.task_name}")

    # -- Initialization ---
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    data_module = hydra.utils.instantiate(cfg.datamodule)

    class_weights = data_module.class_weights

    # train_dl, val_dl = data_module.train_dataloader(), data_module.val_dataloader()
    train_dl, val_dl = data_module.get_sampled_dataloader()

    segmentation_criterion = hydra.utils.instantiate(cfg.losses.segmentation_criterion)
    classification_criterion = hydra.utils.instantiate(
        cfg.losses.classification_criterion, weight=class_weights
    )

    # torch.cuda.empty_cache()
    device = get_default_device()

    model = hydra.utils.instantiate(
        cfg.models.model,
        segmentation_criterion=segmentation_criterion,
        classification_criterion=classification_criterion,
    )
    model = torch.compile(model)

    # -- GPU setup ---
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)

    log.info(f"Instantiating optimizer <{cfg.models.optimizer._target_}>")
    optimizer = hydra.utils.instantiate(cfg.models.optimizer, params=model.parameters())

    log.info("Starting training!")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info(f"run_id: {run_id}")

        EPOCHS = cfg.trainer.max_epochs

        mlflow.log_params({"epochs": EPOCHS})
        mlflow.log_params({"batch_size": cfg.datamodule.batch_size})
        mlflow.log_params({"optimizer": cfg.models.optimizer.values()})

        # Log model summary.
        results_dir = Path(cfg.paths.results_dir)
        results_dir.mkdir(exist_ok=True)
        summery_path = results_dir / "model_summary.txt"
        with open(str(summery_path), "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact(str(summery_path))

        history = fit(
            model=model,
            train_dataloader=train_dl,
            validation_dataloader=val_dl,
            epochs=EPOCHS,
            optimizer=optimizer,
            device_type=device.type,
            dtype=torch.float16,
            reduce_lr_on_plateau=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=5
            ),
        )

        mlflow.pytorch.log_model(model, "model", signature=model_signature(model, train_dl))
        mlflow.log_metrics(history[0])

    return history[0], run_id, data_module


@hydra.main(config_path=str(root / "configs"), config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:

    history, mlflow_run_id, data_module = train(cfg)
    # data_module = hydra.utils.instantiate(cfg.datamodule)
    test_ds = data_module.test_dataloader()
    model = mlflow.pytorch.load_model(f"runs:/{mlflow_run_id}/model")
    device = get_default_device()
    model = to_device(model, device)
    # model evaluation using val_dl
    test_metrics, cls_report, y_true, y_pred, plot_samples, per_sample_dice_scores = (
        evaluate_model(
            model=model,
            test_loader=DeviceDataLoader(test_ds, device),
            device=device,
            seg_threshold=0.5,
            class_names=data_module.classes,
        )
    )
    mlflow.log_metrics(test_metrics)
    mlflow.log_text(cls_report, "classification_report.txt")

    log.info("Generating prediction visualization plot...")

    # model visualization and figure logging
    roc_curve = plot_roc_curve(y_true, y_pred)
    mlflow.log_figure(roc_curve, "roc_curve.png")

    cm_fig_norm = plot_confusion_matrix(y_true, y_pred, class_names=data_module.classes)
    cm_fig_norm_path = f"{cfg.paths.results_dir}/confusion_matrix.png"
    cm_fig_norm.savefig(cm_fig_norm_path)
    mlflow.log_figure(cm_fig_norm, "confusion_matrix.png")

    gif_output_path = f"{cfg.paths.results_dir}/predictions_animation.gif"
    create_prediction_gif(
        images=plot_samples["images"],
        true_masks=plot_samples["true_masks"],
        pred_masks=plot_samples["pred_masks"],
        true_labels=plot_samples["true_labels"],
        pred_labels=plot_samples["pred_labels"],
        class_names=data_module.classes,
        gif_path=gif_output_path,
        duration=5,
    )
    mlflow.log_artifact(gif_output_path)
    hist_fig = plot_score_histogram(per_sample_dice_scores, score_type="Dice")
    hist_output_path = f"{cfg.paths.results_dir}/dice_score_histogram.png"
    hist_fig.savefig(hist_output_path)
    mlflow.log_figure(hist_fig, "dice_score_histogram.png")


if __name__ == "__main__":
    main()
