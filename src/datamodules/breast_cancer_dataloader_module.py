import logging
import sys

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

log = logging.getLogger(__name__)


class TransformWrapper(Dataset):
    def __init__(
        self,
        dataset,
        shared_xform: v2.Compose | None = None,
        image_xform: v2.Compose | None = None,
        mask_xform: v2.Compose | None = None,
    ) -> None:
        self.dataset = dataset
        self.shared_xform = shared_xform
        self.image_xform = image_xform
        self.mask_xform = mask_xform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Get the raw image, mask, and label from the underlying dataset
        img, target = self.dataset[index]

        # Apply transformations to the image and mask
        if self.shared_xform is not None:
            img, target = self.shared_xform(img, target)

        if self.image_xform is not None:
            img = self.image_xform(img)

        if self.mask_xform is not None:
            target["masks"] = self.mask_xform(target["masks"])

        # Return the transformed image, mask, and label
        return img, target


class BreastCancerDataLoaderModule(Dataset):
    def __init__(
        self,
        data: DictConfig,  # This will receive the dataset config
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.image_xform = None
        self.masks_xform = None
        self.valid_shared_xform = None
        self.valid_image_xform = None
        self.valid_masks_xform = None
        self.shared_xform = None

        # initialize the BreastCancerDataset using the provided config - returns dict with initialize object with dict
        self.dataset_config = hydra.utils.instantiate(data)
        self.dataset = self.dataset_config["dataset"]
        self.class_weights = self.dataset.class_weights

        # initialize the transforms from the config

        if self.dataset_config["train_shared_transforms"] is not None:
            self.shared_xform = v2.Compose(self.dataset_config["train_shared_transforms"])

        if self.dataset_config["train_image_trasforms"] is not None:
            self.image_xform = v2.Compose(self.dataset_config["train_image_trasforms"])

        if self.dataset_config["train_masks_transforms"] is not None:
            self.masks_xform = v2.Compose(self.dataset_config["train_masks_transforms"])

        if self.dataset_config["val_shared_transforms"] is not None:
            self.valid_shared_xform = v2.Compose(self.dataset_config["val_shared_transforms"])

        if self.dataset_config["val_image_transforms"] is not None:
            self.valid_image_xform = v2.Compose(self.dataset_config["val_image_transforms"])

        if self.dataset_config["val_masks_transforms"] is not None:
            self.valid_masks_xform = v2.Compose(self.dataset_config["val_masks_transforms"])

        self.train_dataset, self.val_dataset = self.setup()
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.persistent_workers: bool = persistent_workers

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
        data = TransformWrapper(
            dataset=self.train_dataset,
            shared_xform=self.shared_xform,
            image_xform=self.image_xform,
            mask_xform=self.masks_xform,
        )

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
        data = TransformWrapper(
            dataset=self.val_dataset,
            shared_xform=self.valid_shared_xform,
            image_xform=self.valid_image_xform,
            mask_xform=self.valid_masks_xform,
        )
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
    train_dl = dataloader.train_dataloader()
    _ = dataloader.val_dataloader()
    for images, targets in train_dl:
        print(images.shape, targets["masks"].shape, targets["labels"].shape)

        print(f"images:{images.dtype}, {images[0].min()}, {images[0].max()}")
        print(
            f'masks {targets["masks"].dtype}, {targets["masks"][0].min()}, {targets["masks"][0].max()}'
        )
        print(
            f'labels {targets["labels"].dtype}, {targets["labels"][0].min()}, {targets["labels"][0].max()}'
        )
        break


if __name__ == "__main__":
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )
    sys.path.append(str(root))
    # Register a resolver for torch dtypes
    OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))

    main()
