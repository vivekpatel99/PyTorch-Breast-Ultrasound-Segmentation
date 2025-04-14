import logging
import sys

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
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

        # Convert masks to binary
        target["masks"][target["masks"] == 255.0] = 1.0

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
        val_split: float = 0.2,  # Added val_split parameter
        random_state: int = 42,  # Added random_state for reproducibility
    ):
        super().__init__()
        self.image_xform = None
        self.masks_xform = None
        self.valid_shared_xform = None
        self.valid_image_xform = None
        self.valid_masks_xform = None
        self.shared_xform = None

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.persistent_workers: bool = (
            persistent_workers and self.num_workers > 0
        )  # Only if workers > 0
        self.val_split: float = val_split
        self.random_state: int = random_state

        # initialize the BreastCancerDataset using the provided config - returns dict with initialize object with dict
        self.dataset_config = hydra.utils.instantiate(data)
        self.dataset = self.dataset_config["dataset"]
        self.class_weights = self.dataset.class_weights
        self.classes = self.dataset.classes
        self.sample_len = 0
        # initialize the transforms from the config
        self._init_transforms()  # Encapsulated transform initialization
        self.train_dataset, self.val_dataset = self.split_and_preprocess_datasets()

    def _init_transforms(self) -> None:
        """Initializes transforms from the configuration."""
        if self.dataset_config.get("train_shared_transforms"):  # Use .get for safety
            self.shared_xform = v2.Compose(self.dataset_config["train_shared_transforms"])

        if self.dataset_config.get("train_image_trasforms"):
            self.image_xform = v2.Compose(self.dataset_config["train_image_trasforms"])

        if self.dataset_config.get("train_masks_transforms"):
            self.masks_xform = v2.Compose(self.dataset_config["train_masks_transforms"])

        if self.dataset_config.get("val_shared_transforms"):
            self.valid_shared_xform = v2.Compose(self.dataset_config["val_shared_transforms"])

        if self.dataset_config.get("val_image_transforms"):
            self.valid_image_xform = v2.Compose(self.dataset_config["val_image_transforms"])

        if self.dataset_config.get("val_masks_transforms"):
            self.valid_masks_xform = v2.Compose(self.dataset_config["val_masks_transforms"])

    def split_and_preprocess_datasets(self) -> tuple[Dataset, Dataset]:
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`, `self.test_dataset`."""
        log.info("Splitting dataset")
        # Create indices for splitting
        indices = list(range(len(self.dataset)))

        train_indices, val_indices = train_test_split(
            indices,
            test_size=self.val_split,
            random_state=self.random_state,
            shuffle=True,
            stratify=self.dataset.labels,
        )
        # Use Subset to avoid copying data
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)

        train_dataset = TransformWrapper(
            dataset=train_subset,
            shared_xform=self.shared_xform,
            image_xform=self.image_xform,
            mask_xform=self.masks_xform,
        )
        val_dataset = TransformWrapper(
            dataset=val_subset,
            shared_xform=self.valid_shared_xform,
            image_xform=self.valid_image_xform,
            mask_xform=self.valid_masks_xform,
        )
        return train_dataset, val_dataset

    def train_dataloader(self, train_ds: Dataset | None = None) -> DataLoader:
        log.info("Creating train dataloader")
        if train_ds is None:
            train_ds = self.train_dataset
        return DataLoader(
            train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # Drop last if batch size doesn't divide evenly
        )

    def val_dataloader(self, val_ds: Dataset | None = None) -> DataLoader:
        log.info("Creating val dataloader")
        if val_ds is None:
            val_ds = self.val_dataset
        return DataLoader(
            dataset=val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )

    def get_sampled_dataloader(
        self, sample_fraction: float = 0.2
    ) -> tuple[DataLoader, DataLoader]:
        """
        Creates stratified train and validation dataloaders from a fraction of the data.

        Args:
            sample_fraction: The fraction of the total dataset to use for the sample.
            sample_random_state: Random state for sampling reproducibility. Defaults to None.

        Returns:
            A tuple containing the sampled train DataLoader and sampled validation DataLoader.
        """

        if not 0.0 <= sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be between 0 and 1")
        log.info(f"Creating sampled dataloaders using {sample_fraction:.1%} of the data.")

        sampled_indices, _ = train_test_split(
            range(len(self.dataset)),
            train_size=sample_fraction,
            random_state=self.random_state,
            shuffle=True,
            stratify=self.dataset.labels,
        )
        self.sample_len = len(sampled_indices)
        log.info(f"Sampled dataset size: {self.sample_len}")
        print(f"Sampled dataset size: {self.sample_len}")

        sampled_labels = [
            self.dataset.labels[i] for i in sampled_indices
        ]  # Get labels for the sample

        # Split the *sampled* indices into train and validation sets
        # Use the same validation split ratio as the full dataset
        # Ensure there's at least one sample in validation if split is small
        current_val_split = (
            min(self.val_split, 1 - sample_fraction) if len(sampled_indices) > 1 else 0
        )
        if len(sampled_indices) <= 1 or current_val_split == 0:
            # Handle cases with very few samples - put all in train
            log.warning(
                f"Sample size ({len(sampled_indices)}) too small for validation split. Using all for training."
            )
            train_sample_indices = sampled_indices
            val_sample_indices = []

        else:
            train_sample_indices, val_sample_indices = train_test_split(
                sampled_indices,
                test_size=current_val_split,
                random_state=self.random_state,
                shuffle=True,
                stratify=sampled_labels,
            )

        log.info(
            f"Sampled dataset split: {len(train_sample_indices)} train, {len(val_sample_indices)} val samples."
        )
        # Create Subsets and Wrap with Transforms
        train_sample_subset = Subset(self.dataset, train_sample_indices)
        val_sample_subset = (
            Subset(self.dataset, val_sample_indices) if val_sample_indices else None
        )  # Handle empty val set

        train_sample_transformed = TransformWrapper(
            dataset=train_sample_subset,
            shared_xform=self.shared_xform,
            image_xform=self.image_xform,
            mask_xform=self.masks_xform,
        )
        if val_sample_subset:
            val_sample_transformed = TransformWrapper(
                dataset=val_sample_subset,
                shared_xform=self.valid_shared_xform,  # Use validation transforms
                image_xform=self.valid_image_xform,
                mask_xform=self.valid_masks_xform,
            )
        else:
            val_sample_transformed = None  # No validation data

        # 4. Create DataLoaders
        train_sample_dl = self.train_dataloader(train_ds=train_sample_transformed)

        if val_sample_transformed:
            val_sample_dl = self.val_dataloader(val_ds=val_sample_transformed)
        else:
            # Return an empty dataloader or handle as needed if no validation samples
            log.warning("Returning an empty validation dataloader due to small sample size.")
            val_sample_dl = DataLoader([])  # Empty dataloader

        return train_sample_dl, val_sample_dl

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
