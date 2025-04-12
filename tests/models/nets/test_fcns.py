# tests/models/components/nets/test_fcns.py

import pytest
import torch

# Assuming your project structure allows this import
# Adjust the path if necessary based on your test setup
from src.models.components.nets.fcns import FCN8Decoder, FCN16Decoder

# --- Test Configuration ---
BATCH_SIZE = 2
INPUT_H, INPUT_W = 224, 224  # Standard VGG input size
SEG_NUM_CLASSES = 1  # Example number of segmentation classes

# Expected feature map dimensions after VGG pooling layers for a 224x224 input
# and the custom final_conv layer in VGGNetEncoder
FEATURE_SHAPES = {
    "pool3": (BATCH_SIZE, 256, INPUT_H // 8, INPUT_W // 8),  # 28x28
    "pool4": (BATCH_SIZE, 512, INPUT_H // 16, INPUT_W // 16),  # 14x14
    "pool5": (BATCH_SIZE, 4096, INPUT_H // 32, INPUT_W // 32),  # 7x7 (after final_conv)
}

# Expected output shape after upsampling back to original size
EXPECTED_OUTPUT_SHAPE = (BATCH_SIZE, SEG_NUM_CLASSES, INPUT_H, INPUT_W)


# --- Helper Function ---
def create_dummy_features(shapes: dict[str, tuple]) -> dict[str, torch.Tensor]:
    """Creates a dictionary of dummy feature tensors."""
    return {name: torch.randn(shape) for name, shape in shapes.items()}


# --- Test Fixtures ---
@pytest.fixture(scope="module")
def dummy_features_fcn16() -> dict[str, torch.Tensor]:
    """Provides dummy features needed for FCN16Decoder."""
    return create_dummy_features(
        {"pool4": FEATURE_SHAPES["pool4"], "pool5": FEATURE_SHAPES["pool5"]}
    )


@pytest.fixture(scope="module")
def dummy_features_fcn8() -> dict[str, torch.Tensor]:
    """Provides dummy features needed for FCN8Decoder."""
    return create_dummy_features(
        {
            "pool3": FEATURE_SHAPES["pool3"],
            "pool4": FEATURE_SHAPES["pool4"],
            "pool5": FEATURE_SHAPES["pool5"],
        }
    )


# --- Test Cases ---


def test_fcn16_decoder_init():
    """Tests FCN16Decoder initialization."""
    decoder = FCN16Decoder(seg_num_classes=SEG_NUM_CLASSES)
    assert isinstance(decoder, FCN16Decoder)
    assert decoder.seg_num_classes == SEG_NUM_CLASSES
    # Check if key layers exist (optional sanity check)
    assert hasattr(decoder, "upsample_5")
    assert hasattr(decoder, "upsample_4")
    assert hasattr(decoder, "mask_segmentation")


def test_fcn16_decoder_forward(dummy_features_fcn16):
    """Tests the forward pass of FCN16Decoder."""
    decoder = FCN16Decoder(seg_num_classes=SEG_NUM_CLASSES)
    output = decoder(dummy_features_fcn16)

    assert isinstance(output, torch.Tensor)
    assert output.shape == EXPECTED_OUTPUT_SHAPE
    assert output.dtype == torch.float32


def test_fcn8_decoder_init():
    """Tests FCN8Decoder initialization."""
    decoder = FCN8Decoder(seg_num_classes=SEG_NUM_CLASSES)
    assert isinstance(decoder, FCN8Decoder)
    assert decoder.seg_num_classes == SEG_NUM_CLASSES
    # Check if key layers exist (optional sanity check)
    assert hasattr(decoder, "upsample_5")
    assert hasattr(decoder, "upsample_4")
    assert hasattr(decoder, "upsample3")  # Note: FCN8 has upsample3, FCN16 doesn't use pool3 skip
    assert hasattr(decoder, "mask_segmentation")


def test_fcn8_decoder_forward(dummy_features_fcn8):
    """Tests the forward pass of FCN8Decoder."""
    decoder = FCN8Decoder(seg_num_classes=SEG_NUM_CLASSES)
    # Correct the typo in the forward method argument name if it exists in your actual code
    # Assuming the argument name is 'features' based on FCN16Decoder
    output = decoder(dummy_features_fcn8)  # Use 'features' if you fixed the typo 'feaatures'

    assert isinstance(output, torch.Tensor)
    assert output.shape == EXPECTED_OUTPUT_SHAPE
    assert output.dtype == torch.float32


@pytest.mark.parametrize("num_classes", [1, 5, 10])
def test_fcn_decoders_different_classes(num_classes, dummy_features_fcn8, dummy_features_fcn16):
    """Tests decoders with varying number of output classes."""
    expected_shape = (BATCH_SIZE, num_classes, INPUT_H, INPUT_W)

    # Test FCN16
    decoder16 = FCN16Decoder(seg_num_classes=num_classes)
    output16 = decoder16(dummy_features_fcn16)
    assert output16.shape == expected_shape

    # Test FCN8
    decoder8 = FCN8Decoder(seg_num_classes=num_classes)
    output8 = decoder8(dummy_features_fcn8)
    assert output8.shape == expected_shape


# --- Potential Improvements Noticed ---
# 1. Typo in FCN8Decoder forward method argument: 'feaatures' should be 'features'.
# 2. FCN16Decoder forward method signature indicates it returns a dict, but the implementation returns a Tensor.
#    The signature should be `-> torch.Tensor`.
# 3. Consider adding type hints to the forward method arguments (`features: dict[str, torch.Tensor]`).
