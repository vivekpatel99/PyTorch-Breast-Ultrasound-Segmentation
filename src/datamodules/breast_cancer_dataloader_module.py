import logging
import sys

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

log = logging.getLogger(__name__)


class TransformWrapper(Dataset):
    def __init__(self, dataset, xform: T.Compose) -> None:
        self.dataset = dataset
        self.xforms = xform

        self.train_todtype = T.ToDtype(torch.float32, scale=True)
        self.mask_todtype = T.ToDtype(torch.uint8, scale=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Get the raw image, mask, and label from the underlying dataset
        img, target = self.dataset[index]

        # Apply transformations to the image and mask

        transformed_img, transformed_target = self.xforms(img, target)

        # Convert the transformed image to float32
        transformed_img = self.train_todtype(transformed_img)

        # # Convert the mask to uint8
        transformed_target["mask"] = self.mask_todtype(transformed_target["masks"])

        # Return the transformed image, mask, and label
        return transformed_img, transformed_target


class BreastCancerDataLoaderModule(Dataset):
    def __init__(
        self,
        data: DictConfig,  # This will receive the dataset config
        transforms_cfg: DictConfig,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):

        # initialize the BreastCancerDataset using the provided config - returns dict with initialize object with dict
        self.dataset_config = hydra.utils.instantiate(data)
        self.dataset = self.dataset_config["dataset"]

        # initialize the transforms from the config
        self.train_xform = T.Compose(transforms_cfg["train_transforms"])

        self.valid_xform = T.Compose(transforms_cfg["val_transforms"])
        # self.test__xform = T.Compose(transforms_cfg["test_transforms"].append(
        #      T.ToDtype(torch.float32, scale=True),))

        self.train_dataset, self.val_dataset = self.setup()
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.persistent_workers: bool = persistent_workers
        # self.test_dataset: Dataset | None  = None

    def setup(self) -> tuple[Dataset, Dataset]:
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`, `self.test_dataset`."""
        log.info("Splitting dataset")
        train_dataset, val_dataset = train_test_split(
            self.dataset,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=self.dataset.labels,
        )
        return train_dataset, val_dataset

    def train_dataloader(self) -> DataLoader:
        log.info("Creating train dataloader")
        data = TransformWrapper(self.train_dataset, self.train_xform)
        return DataLoader(
            data,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        log.info("Creating val dataloader")
        data = TransformWrapper(self.val_dataset, self.valid_xform)
        return DataLoader(
            dataset=data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError


@hydra.main(version_base="1.2", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    dataloader = hydra.utils.instantiate(cfg.datamodule)
    _ = dataloader.train_dataloader()
    _ = dataloader.val_dataloader()


if __name__ == "__main__":
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )
    sys.path.append(str(root))

    main()
