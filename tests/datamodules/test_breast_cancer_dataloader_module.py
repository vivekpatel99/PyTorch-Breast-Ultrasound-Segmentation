# tests/datamodules/test_simple_breast_cancer_dataloader_module.py

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Assuming your module is located at src.datamodules.breast_cancer_dataloader_module
from src.datamodules.breast_cancer_dataloader_module import (
    BreastCancerDataLoaderModule,
    TransformWrapper,
)


# --- Mock Dataset (Simplified from existing tests) ---
class SimpleMockBreastCancerDataset(Dataset):
    """A very simple mock dataset for basic testing."""

    def __init__(self, num_samples=50, img_size=(1, 32, 32)):
        self.num_samples = num_samples
        self.img_size = img_size
        # Generate reproducible labels for stratification testing
        self.labels = [i % 2 for i in range(num_samples)]  # Simple binary labels
        self.class_weights = torch.tensor([0.5, 0.5])  # Dummy weights
        self.classes = ["class_0", "class_1"]  # Dummy classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index >= self.num_samples:
            raise IndexError
        # Return dummy data matching the expected structure
        image = torch.randn(self.img_size)
        # Simple binary mask
        mask = (
            torch.randint(0, 2, (1, self.img_size[1], self.img_size[2]), dtype=torch.float32)
            * 255.0
        )
        label = torch.tensor(self.labels[index], dtype=torch.long)
        target = {"masks": mask, "labels": label}
        # Simulate the structure returned by hydra instantiation if needed
        # In this simple case, we assume the datamodule accesses the dataset directly
        # return {"dataset": self, "other_stuff": None} # If hydra returns a dict
        return image, target  # If hydra returns the dataset instance directly


# --- Fixtures ---
@pytest.fixture
def mock_dataset_instance():
    """Provides a fresh instance of the simple mock dataset."""
    return SimpleMockBreastCancerDataset(num_samples=50)


@pytest.fixture
def simple_base_config() -> DictConfig:
    """Provides a minimal OmegaConf DictConfig for the DataModule."""
    conf = OmegaConf.create(
        {
            # Config structure mimics how hydra would pass it
            "data": {
                # Point to the mock dataset within this test file
                "_target_": "tests.datamodules.test_simple_breast_cancer_dataloader_module.SimpleMockBreastCancerDataset",
                "num_samples": 50,
                # No transforms needed for these simple tests
                "train_shared_transforms": None,
                "train_image_transforms": None,  # Corrected typo if present in original
                "train_masks_transforms": None,
                "val_shared_transforms": None,
                "val_image_transforms": None,
                "val_masks_transforms": None,
            },
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "val_split": 0.2,  # 10 val
            "test_split": 0.2,  # 10 test -> 30 train/val -> 6 val / 24 train
            "random_state": 42,
        }
    )
    return conf


# --- Simple Test Class ---


# Patch hydra's instantiate where it's called inside the DataModule's __init__
# We mock it to return our predefined mock dataset instance
@patch("src.datamodules.breast_cancer_dataloader_module.hydra.utils.instantiate")
class TestSimpleBreastCancerDataLoaderModule:

    def test_initialization(self, mock_instantiate, simple_base_config, mock_dataset_instance):
        """Test if the DataModule initializes without errors."""

        # Configure the mock to return the dataset when called with the data config
        # And None for transforms
        def instantiate_side_effect(cfg, *args, **kwargs):
            if cfg == simple_base_config.data:
                # Simulate hydra returning a dict containing the dataset
                return {"dataset": mock_dataset_instance}
            elif cfg is None:
                return None
            elif isinstance(cfg, list):  # Handle transform lists
                # In this simple test, transforms are None, but handle the case
                return None  # Or mock v2.Compose if transforms were defined
            else:
                return MagicMock()  # Fallback

        mock_instantiate.side_effect = instantiate_side_effect

        # Instantiate the DataModule
        dm = BreastCancerDataLoaderModule(**simple_base_config)

        # Basic assertions
        assert dm is not None
        assert dm.batch_size == simple_base_config.batch_size
        assert dm.val_split == simple_base_config.val_split
        assert dm.test_split == simple_base_config.test_split
        assert dm.dataset == mock_dataset_instance  # Check dataset assignment

        # Check if split datasets were created (they are TransformWrappers around Subsets)
        assert hasattr(dm, "train_dataset")
        assert hasattr(dm, "val_dataset")
        assert hasattr(dm, "test_datasets")  # Note: attribute name is test_datasets
        assert isinstance(dm.train_dataset, TransformWrapper)
        assert isinstance(dm.val_dataset, TransformWrapper)
        assert isinstance(dm.test_datasets, TransformWrapper)  # Corrected attribute name check

    def test_dataloader_creation(
        self, mock_instantiate, simple_base_config, mock_dataset_instance
    ):
        """Test if dataloaders can be created."""

        # Configure mock instantiate as in the previous test
        def instantiate_side_effect(cfg, *args, **kwargs):
            if cfg == simple_base_config.data:
                return {"dataset": mock_dataset_instance}
            return None  # Simplify for transforms in this test

        mock_instantiate.side_effect = instantiate_side_effect

        dm = BreastCancerDataLoaderModule(**simple_base_config)

        # Create dataloaders
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()

        # Assert they are DataLoader instances
        assert isinstance(train_dl, DataLoader)
        assert isinstance(val_dl, DataLoader)
        assert isinstance(test_dl, DataLoader)

        # Check basic properties (optional, but good)
        assert train_dl.batch_size == simple_base_config.batch_size
        assert val_dl.batch_size == simple_base_config.batch_size
        assert test_dl.batch_size == simple_base_config.batch_size

    def test_dataset_splitting_sizes(
        self, mock_instantiate, simple_base_config, mock_dataset_instance
    ):
        """Test if the dataset splitting results in expected subset sizes."""

        # Configure mock instantiate
        def instantiate_side_effect(cfg, *args, **kwargs):
            if cfg == simple_base_config.data:
                return {"dataset": mock_dataset_instance}
            return None

        mock_instantiate.side_effect = instantiate_side_effect

        dm = BreastCancerDataLoaderModule(**simple_base_config)

        total_samples = len(mock_dataset_instance)  # 50
        test_split_ratio = simple_base_config.test_split  # 0.2
        val_split_ratio = simple_base_config.val_split  # 0.2

        # Calculate expected sizes (approximate due to stratification and integer rounding)
        expected_test_size = int(total_samples * test_split_ratio)  # 50 * 0.2 = 10
        remaining_size = total_samples - expected_test_size  # 40
        # Val split is calculated relative to the *remaining* data after test split
        relative_val_split = val_split_ratio / (
            1.0 - test_split_ratio
        )  # 0.2 / (1 - 0.2) = 0.2 / 0.8 = 0.25
        expected_val_size = int(remaining_size * relative_val_split)  # 40 * 0.25 = 10
        expected_train_size = remaining_size - expected_val_size  # 40 - 10 = 30

        # Assert the lengths of the datasets wrapped by TransformWrapper
        # The length of TransformWrapper is the length of the underlying dataset (Subset)
        assert len(dm.test_datasets) == expected_test_size  # Check test_datasets attribute
        assert len(dm.val_dataset) == expected_val_size
        assert len(dm.train_dataset) == expected_train_size

        # Verify total adds up
        assert len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_datasets) == total_samples

    def test_dataloader_batch_content(
        self, mock_instantiate, simple_base_config, mock_dataset_instance
    ):
        """Test fetching a batch and checking basic content structure and types."""

        # Configure mock instantiate
        def instantiate_side_effect(cfg, *args, **kwargs):
            if cfg == simple_base_config.data:
                return {"dataset": mock_dataset_instance}
            return None

        mock_instantiate.side_effect = instantiate_side_effect

        dm = BreastCancerDataLoaderModule(**simple_base_config)
        train_dl = dm.train_dataloader()

        # Get one batch
        img_batch, target_batch = next(iter(train_dl))

        # --- Basic Checks ---
        assert isinstance(img_batch, torch.Tensor)
        assert isinstance(target_batch, dict)
        assert "masks" in target_batch
        assert "labels" in target_batch
        assert isinstance(target_batch["masks"], torch.Tensor)
        assert isinstance(target_batch["labels"], torch.Tensor)

        # --- Shape Checks (considering batch size and drop_last=True for train) ---
        expected_batch_size = simple_base_config.batch_size
        assert (
            img_batch.shape[0] <= expected_batch_size
        )  # Can be smaller if dataset size isn't multiple
        assert target_batch["masks"].shape[0] <= expected_batch_size
        assert target_batch["labels"].shape[0] <= expected_batch_size
        # Check feature shapes match mock dataset (excluding batch dim)
        assert img_batch.shape[1:] == mock_dataset_instance.img_size
        assert target_batch["masks"].shape[1:] == (
            1,
            mock_dataset_instance.img_size[1],
            mock_dataset_instance.img_size[2],
        )

        # --- Mask Binarization Check (simple version) ---
        # Check mask values are 0.0 or 1.0 after TransformWrapper
        unique_mask_values = torch.unique(target_batch["masks"])
        assert torch.all((unique_mask_values == 0.0) | (unique_mask_values == 1.0))
