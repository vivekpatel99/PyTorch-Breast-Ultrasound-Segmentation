# tests/datamodules/components/test_breast_cancer_dataset.py
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import tv_tensors

from src.datamodules.components.breast_cancer_dataset import BreastCancerDataset

# Make sure the src directory is in the Python path for imports
# This might be handled by your pytest configuration (e.g., pytest.ini or conftest.py)
# Or you might need to adjust sys.path if running directly
# Example:
# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def dummy_data_dir(tmp_path_factory) -> Path:
    """Creates a temporary directory structure with dummy image/mask files."""
    tmp_dir = tmp_path_factory.mktemp("dataset")
    data_dir = tmp_dir / "Dataset_BUSI_with_GT"  # Matches the expected structure
    data_dir.mkdir()

    classes = ["benign", "malignant", "normal"]
    num_samples_per_class = {"benign": 2, "malignant": 3, "normal": 1}  # Uneven for weight testing
    img_size = (32, 32)  # Small dummy images

    log.info(f"Creating dummy data in {data_dir}")

    for class_name in classes:
        class_dir = data_dir / class_name
        class_dir.mkdir()
        count = num_samples_per_class[class_name]
        for i in range(1, count + 1):
            # Create dummy RGB image
            img_array = np.random.randint(0, 256, (*img_size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, "RGB")
            img_path = class_dir / f"{class_name} ({i}).png"
            img.save(img_path)

            # Create dummy grayscale mask
            mask_array = (
                np.random.randint(0, 2, (*img_size, 1), dtype=np.uint8) * 255
            )  # Binary mask
            mask = Image.fromarray(mask_array.squeeze(), "L")  # Grayscale
            mask_path = class_dir / f"{class_name} ({i})_mask.png"
            mask.save(mask_path)

    log.info(f"Dummy data created with classes: {classes}")
    return data_dir


# --- Test Class ---


class TestBreastCancerDataset:

    @pytest.fixture(scope="class")
    def dataset(self, dummy_data_dir) -> BreastCancerDataset:
        """Initializes the dataset using the dummy data directory."""
        # Mock opendatasets.download to prevent actual download attempts
        with patch("opendatasets.download") as mock_download:
            # We pass a dummy URL, it won't be used if data exists
            dataset = BreastCancerDataset(data_dir=dummy_data_dir, dataset_url="dummy/url")
            mock_download.assert_not_called()  # Ensure download wasn't called as data exists
            return dataset

    def test_initialization(self, dataset, dummy_data_dir):
        """Tests dataset initialization attributes."""
        assert dataset.data_dir == dummy_data_dir
        assert dataset.root_data_dir == dummy_data_dir.parent
        assert sorted(dataset.class_names) == sorted(["benign", "malignant", "normal"])
        assert dataset.num_classes == 3
        assert dataset.label_mapping == {"benign": 0, "malignant": 1, "normal": 2}
        print(len(dataset.images))
        assert len(dataset.images) == 12  # Total samples
        assert len(dataset.masks) == len(dataset.images)
        assert len(dataset.labels) == len(dataset.images)
        assert dataset._class_weights is not None
        assert isinstance(dataset.class_weights, torch.Tensor)

    def test_len(self, dataset):
        """Tests the __len__ method."""
        assert len(dataset) == 12  # 2 benign + 3 malignant + 1 normal with their masks

    def test_getitem(self, dataset):
        """Tests the __getitem__ method."""
        img, target = dataset[0]  # Get the first item

        # Check types
        assert isinstance(img, tv_tensors.Image)
        assert isinstance(target, dict)
        assert isinstance(target["masks"], tv_tensors.Mask)
        assert isinstance(target["labels"], int)  # Should be int index, not tensor

        # Check shapes (C, H, W) for image, (1, H, W) for mask
        assert img.shape == (3, 32, 32)
        assert target["masks"].shape == (1, 32, 32)

        # Check label value range
        assert 0 <= target["labels"] < dataset.num_classes

        # Check tensor dtypes
        assert img.dtype == torch.uint8  # io.decode_image reads as uint8
        assert target["masks"].dtype == torch.uint8

    def test_classes_property(self, dataset):
        """Tests the classes property."""
        assert isinstance(dataset.classes, list)
        assert sorted(dataset.classes) == sorted(["benign", "malignant", "normal"])

    def test_class_to_idx_property(self, dataset):
        """Tests the class_to_idx property."""
        assert isinstance(dataset.class_to_idx, dict)
        assert dataset.class_to_idx == dataset.label_mapping

    def test_class_weights_property(self, dataset):
        """Tests the class_weights property and calculation."""
        weights = dataset.class_weights
        assert isinstance(weights, torch.Tensor)
        assert weights.dtype == torch.float32
        assert weights.shape == (dataset.num_classes,)
        # Check if weights are calculated (should not be uniform due to uneven samples)
        # Exact values depend on scikit-learn's calculation, but they shouldn't all be 1.0
        # Example check: weights shouldn't be all equal if classes are imbalanced
        unique_weights = torch.unique(weights)
        assert len(unique_weights) > 1  # Expecting different weights for imbalanced classes

        # Rough check based on counts (2, 3, 1) -> total 6
        # Expected weights proportional to N / (n_classes * n_samples_class)
        # benign: 6 / (3 * 2) = 1.0
        # malignant: 6 / (3 * 3) = 0.666...
        # normal: 6 / (3 * 1) = 2.0
        # Note: sklearn might normalize differently, but relative order should hold
        # We need to map the calculated weights back to the correct class index
        benign_idx = dataset.label_mapping["benign"]
        malignant_idx = dataset.label_mapping["malignant"]
        normal_idx = dataset.label_mapping["normal"]

        # Check relative magnitudes (normal should have highest weight, malignant lowest)
        assert weights[normal_idx] > weights[benign_idx]
        assert weights[benign_idx] > weights[malignant_idx]

    def test_get_data(self, dataset):
        """Tests the get_data method implicitly via init and checks return types."""
        # get_data is called during init, we check the results stored in the instance
        assert isinstance(dataset.images, list)
        assert isinstance(dataset.masks, list)
        assert isinstance(dataset.labels, list)
        assert all(isinstance(p, Path) for p in dataset.images)
        assert all(isinstance(p, Path) for p in dataset.masks)
        assert all(isinstance(s, str) for s in dataset.labels)
        assert len(dataset.images) == len(dataset)
        assert len(dataset.masks) == len(dataset)
        assert len(dataset.labels) == len(dataset)

    def test_download_dataset_value_error(self, tmp_path):
        """Tests that ValueError is raised if URL is None and download is needed."""
        empty_root = tmp_path / "empty_root"
        empty_root.mkdir()
        data_dir = empty_root / "Dataset_BUSI_with_GT"
        # data_dir does NOT exist here

        with pytest.raises(ValueError, match="Dataset URL not provided"):
            # Pass None for dataset_url, init should try to download
            _ = BreastCancerDataset(data_dir=data_dir, dataset_url=None)

    @patch("opendatasets.download")  # Mock the download function
    def test_download_dataset_called(self, mock_download, tmp_path):
        """Tests that od.download is called when the data directory is empty."""
        empty_root = tmp_path / "empty_root_for_download"
        empty_root.mkdir()
        data_dir = empty_root / "Dataset_BUSI_with_GT"
        # data_dir does NOT exist initially
        dataset_url = "test/kaggle-dataset"

        # We expect __init__ to fail after download because no data is *actually*
        # downloaded by the mock. We just check the call.
        try:
            _ = BreastCancerDataset(data_dir=data_dir, dataset_url=dataset_url)
        except FileNotFoundError:  # It will fail when trying to read the (non-existent) dirs
            pass  # Expected failure after mock download

        # Assert that download was called once with the correct arguments
        mock_download.assert_called_once_with(
            dataset_id_or_url=dataset_url,
            data_dir=str(empty_root),  # download expects the parent dir
        )
