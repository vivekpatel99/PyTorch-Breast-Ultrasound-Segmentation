import logging
import sys

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as v2

log = logging.getLogger(__name__)


class TransformWrapper(Dataset):
    def __init__(self, dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
            y["masks"] = self.transform(y["masks"])

        return x, y


class BreastCancerDataLoaderModule(Dataset):
    def __init__(
        self,
        data: DictConfig,  # This will receive the dataset config
        transforms: DictConfig,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):

        # initialize the BreastCancerDataset using the provided config - returns dict with initialize object
        self.dataset_config = hydra.utils.instantiate(data)
        self.dataset = self.dataset_config["dataset"]

        # initialize the transforms from the config
        self.train_xform = v2.Compose(transforms["train_transforms"])
        self.valid_xform = v2.Compose(transforms["val_transforms"])
        self.test__xform = v2.Compose(transforms["test_transforms"])

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

        return DataLoader(
            TransformWrapper(self.train_dataset, self.train_xform),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        log.info("Creating val dataloader")
        return DataLoader(
            dataset=TransformWrapper(self.val_dataset, self.valid_xform),
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
