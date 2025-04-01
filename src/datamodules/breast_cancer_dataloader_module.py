import logging
import os
import sys
from pathlib import Path

import hydra
import omegaconf
import pyrootutils
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as v2

log = logging.getLogger(__name__)


class BreastCancerDataLoaderModule(Dataset):
    def __init__(
        self,
        data: DictConfig,  # This will receive the dataset config
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):

        # initialize the transforms from the config
        # self.transform = v2.Compose(
        #     [hydra.utils.instantiate(transform) for transform in cfg.data.train_transforms]
        # )
        # initialize the BreastCancerDataset using the provided config - returns dict with initialize object
        self.dataset = hydra.utils.instantiate(data)["dataset"]

        # self.dataset = BreastCancerDataset(data_dir=data_dir, transform=self.transform)
        self.train_dataset, self.val_dataset = self.setup()
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.persistent_workers: bool = persistent_workers
        # self.test_dataset: Dataset | None  = None

    def setup(self) -> tuple[Dataset, Dataset]:
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`, `self.test_dataset`."""
        log.info("Splitting dataset")
        self.train_dataset, self.val_dataset = train_test_split(
            self.dataset,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=self.dataset.labels,
        )
        return self.train_dataset, self.val_dataset

    def train_dataloader(self) -> DataLoader:
        log.info("Creating train dataloader")
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        log.info("Creating val dataloader")
        return DataLoader(
            dataset=self.val_dataset,
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
    # cfg: DictConfig | omegaconf.ListConfig = omegaconf.OmegaConf.load(
    #     root / "configs" / "datamodule" / "breast_cancer_datamodule.yaml"
    # )
    # Small hack to allow running this script without staring main config
    # del cfg["defaults"]
    # cfg["batch_size"] = 8
