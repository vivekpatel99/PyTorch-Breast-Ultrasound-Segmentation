import logging
import os

import hydra
import mlflow
import pyrootutils
import torch
from mlflow.models.signature import ModelSignature, infer_signature
from omegaconf import DictConfig
from torchinfo import summary

from src.utils.gpu_utils import DeviceDataLoader, get_default_device, to_device
from src.utils.train_utils import fit

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

if os.getenv("DATA_ROOT") is None:
    os.environ["DATA_ROOT"] = ""

log = logging.getLogger(__name__)


def model_signature(model, train_dl) -> ModelSignature:
    # This ensures layers like Dropout and BatchNorm behave correctly for inference and saves computation.
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(train_dl))
        out = model(images)
        signature = infer_signature(
            model_input={"image_input": images.numpy()},
            model_output={
                "output": {"masks": out["masks"].numpy(), "labels": out["labels"].numpy()}
            },
        )
    return signature


def train(cfg: DictConfig) -> dict[str, float]:
    log.info(f"Instantiating mlflow experiment <{cfg.task_name}>")
    mlflow.set_experiment(f"{cfg.task_name}")

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    data_module = hydra.utils.instantiate(cfg.datamodule)

    class_weights = data_module.class_weights

    train_dl, val_dl = data_module.get_sampled_dataloader()

    segmentation_criterion = hydra.utils.instantiate(cfg.losses.segmentation_criterion)
    classification_criterion = hydra.utils.instantiate(
        cfg.losses.classification_criterion, weight=class_weights
    )

    torch.cuda.empty_cache()
    device = get_default_device()

    # gpu_weights = to_device(class_weights, device)

    model = hydra.utils.instantiate(
        cfg.models.model,
        segmentation_criterion=segmentation_criterion,
        classification_criterion=classification_criterion,
    )
    model = torch.compile(model)

    # gpu setup
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)
    optimizer = hydra.utils.instantiate(cfg.models.optimizer, params=model.parameters(), lr=1e-4)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info(f"run_id: {run_id}")

        EPOCHS = cfg.trainer.max_epochs

        mlflow.log_params({"epochs": EPOCHS})
        mlflow.log_params({"batch_size": cfg.datamodule.batch_size})
        mlflow.log_params({"optimizer": cfg.models.optimizer.values()})
        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

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
        # saving the trained model
        mlflow.pytorch.log_model(model, "model", signature=model_signature(model, train_dl))
        # model evaluation using val_dl
        # model visualization and figure logging
    return history[0]


@hydra.main(config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    history = train(cfg)


if __name__ == "__main__":
    main()
