# tests/models/test_vggnet_fcn8_segmentation_model.py

import pytest
import torch
import torch.nn as nn

from src.models.basemodel import MetricKey  # For integration test

# Assuming your project structure allows this import
# Adjust the path if necessary based on your test setup
from src.models.vggnet_fcn_segmentation_model import VGGNetFCNSegmentationModel

# --- Test Configuration ---
BATCH_SIZE = 2
INPUT_C = 3  # Input channels (e.g., RGB)
INPUT_H, INPUT_W = 224, 224  # Standard VGG input size
SEG_NUM_CLASSES_DEFAULT = 1  # Example binary segmentation
CLS_NUM_CLASSES_DEFAULT = 3  # Example classification classes
VGG_TYPE_DEFAULT = "vgg11"  # Use a smaller VGG for faster testing
FCN_TYPE_DEFAULT = "fcn8"


# --- Helper Dummy Loss Functions ---
# --- Helper Dummy Loss Functions ---
class DummyLoss(nn.Module):
    """A dummy loss function that returns a constant scalar tensor
    but maintains graph connection to its input."""

    def __init__(self):
        super().__init__()

    def forward(
        self, preds, target=None, **kwargs
    ):  # Accept predictions and optional target/kwargs
        # Perform a dummy operation involving preds to connect to the graph.
        # Multiplying the mean by 0.0 ensures the output value is constant (0.5),
        # but establishes the gradient track back to 'preds'.
        # This assumes 'preds' is a tensor that requires gradients coming from the model.
        loss_val = 0.5
        # Ensure preds is float for mean()
        # Add a check in case preds somehow doesn't require grad (it should in training_step)
        if preds.requires_grad:
            # Calculate mean, multiply by 0, add constant. grad_fn will now exist (e.g., AddBackward0)
            loss = torch.mean(preds.float()) * 0.0 + loss_val
        else:
            # Fallback if input doesn't require grad (shouldn't happen in training_step test)
            loss = torch.tensor(loss_val, device=preds.device)

        return loss


# --- Test Fixtures ---
@pytest.fixture(scope="module")
def dummy_input() -> torch.Tensor:
    """Provides a dummy input tensor."""
    return torch.randn(BATCH_SIZE, INPUT_C, INPUT_H, INPUT_W)


@pytest.fixture(scope="module")
def dummy_segmentation_criterion() -> nn.Module:
    """Provides a dummy segmentation loss."""
    return DummyLoss()  # Replace with actual loss if needed for specific tests


@pytest.fixture(scope="module")
def dummy_classification_criterion() -> nn.Module:
    """Provides a dummy classification loss."""
    return DummyLoss()  # Replace with actual loss if needed for specific tests


@pytest.fixture
def default_model(
    dummy_segmentation_criterion, dummy_classification_criterion
) -> VGGNetFCNSegmentationModel:
    """Provides a default VGGNetFCNSegmentationModel instance."""
    return VGGNetFCNSegmentationModel(
        segmentation_criterion=dummy_segmentation_criterion,
        classification_criterion=dummy_classification_criterion,
        seg_num_classes=SEG_NUM_CLASSES_DEFAULT,
        cls_num_classes=CLS_NUM_CLASSES_DEFAULT,
        vggnet_type=VGG_TYPE_DEFAULT,
        fcn_type=FCN_TYPE_DEFAULT,
        seg_weight=0.9,
        cls_weight=0.1,
    )


# --- Test Cases ---


def test_model_initialization(default_model):
    """Tests if the model initializes correctly."""
    assert isinstance(default_model, VGGNetFCNSegmentationModel)
    assert hasattr(default_model, "encoder")
    assert hasattr(default_model, "decoder")
    assert hasattr(default_model, "classification_head")
    assert default_model.decoder.seg_num_classes == SEG_NUM_CLASSES_DEFAULT
    assert default_model.classification_head.head[-1].out_features == CLS_NUM_CLASSES_DEFAULT


@pytest.mark.parametrize(
    "vgg_type, fcn_type, seg_classes, cls_classes, seg_weight, cls_weight",  # Corrected single string
    [
        ("vgg11", "fcn8", 1, 3, 0.9, 0.1),
        ("vgg16_bn", "fcn16", 5, 2, 0.9, 0.1),
        ("vgg19", "fcn8", 1, 10, 0.9, 0.1),
    ],
)
def test_model_initialization_variants(
    dummy_segmentation_criterion,
    dummy_classification_criterion,
    vgg_type,
    fcn_type,
    seg_classes,
    cls_classes,
    seg_weight,
    cls_weight,
):
    """Tests initialization with different configurations."""
    model = VGGNetFCNSegmentationModel(
        segmentation_criterion=dummy_segmentation_criterion,
        classification_criterion=dummy_classification_criterion,
        seg_num_classes=seg_classes,
        cls_num_classes=cls_classes,
        vggnet_type=vgg_type,
        fcn_type=fcn_type,
        seg_weight=seg_weight,
        cls_weight=cls_weight,
    )
    assert isinstance(model, VGGNetFCNSegmentationModel)
    assert model.decoder.seg_num_classes == seg_classes
    # Accessing the last layer of the sequential head remains correct
    assert model.classification_head.head[-1].out_features == cls_classes
    # Check FCN type loaded correctly (simple check based on class name)
    if fcn_type == "fcn8":
        assert "FCN8Decoder" in str(type(model.decoder))
    elif fcn_type == "fcn16":
        assert "FCN16Decoder" in str(type(model.decoder))


def test_invalid_fcn_type(dummy_segmentation_criterion, dummy_classification_criterion):
    """Tests that an invalid fcn_type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown FCN type"):
        VGGNetFCNSegmentationModel(
            segmentation_criterion=dummy_segmentation_criterion,
            classification_criterion=dummy_classification_criterion,
            fcn_type="fcn32",  # Invalid type
            seg_weight=0.9,
            cls_weight=0.1,
        )


def test_forward_pass(default_model, dummy_input):
    """Tests the forward pass execution and output structure."""
    output = default_model(dummy_input)
    assert isinstance(output, dict)
    assert "masks" in output
    assert "labels" in output
    assert isinstance(output["masks"], torch.Tensor)
    assert isinstance(output["labels"], torch.Tensor)


@pytest.mark.parametrize(
    "seg_classes, cls_classes",
    [
        (1, 3),
        (5, 2),
        (1, 1),  # Binary classification
    ],
)
def test_output_shapes(
    dummy_segmentation_criterion,
    dummy_classification_criterion,
    dummy_input,
    seg_classes,
    cls_classes,
):
    """Tests the shapes of the output tensors."""
    model = VGGNetFCNSegmentationModel(
        segmentation_criterion=dummy_segmentation_criterion,
        classification_criterion=dummy_classification_criterion,
        seg_num_classes=seg_classes,
        cls_num_classes=cls_classes,
        vggnet_type=VGG_TYPE_DEFAULT,  # Keep VGG fixed for shape consistency
        fcn_type=FCN_TYPE_DEFAULT,
        seg_weight=0.9,
        cls_weight=0.1,
    )
    output = model(dummy_input)

    expected_mask_shape = (BATCH_SIZE, seg_classes, INPUT_H, INPUT_W)
    expected_label_shape = (BATCH_SIZE, cls_classes)

    assert output["masks"].shape == expected_mask_shape
    assert output["labels"].shape == expected_label_shape
    assert output["masks"].dtype == torch.float32
    assert output["labels"].dtype == torch.float32


def test_training_step_integration(default_model, dummy_input):
    """Tests if the training_step runs with the model's output."""
    # Create a dummy target dictionary
    dummy_targets = {
        "masks": torch.randint(
            0, 2, (BATCH_SIZE, 1, INPUT_H, INPUT_W), dtype=torch.float32
        ),  # Example binary mask
        "labels": torch.randint(0, CLS_NUM_CLASSES_DEFAULT, (BATCH_SIZE,), dtype=torch.long),
    }
    dummy_batch = (dummy_input, dummy_targets)

    # Ensure requires_grad is True for inputs if testing backward pass
    # dummy_input.requires_grad_(True) # Uncomment if you need to test gradient flow

    metrics = default_model.training_step(dummy_batch)

    assert isinstance(metrics, dict)
    assert MetricKey.SEG_LOSS.value in metrics
    assert MetricKey.SEG_DICE.value in metrics
    assert MetricKey.CLS_LOSS.value in metrics
    assert MetricKey.CLS_ACC.value in metrics
    assert MetricKey.CLS_AUROC.value in metrics  # Added AUROC check

    # Check if loss has gradient function (indicates it's ready for backprop)
    assert metrics[MetricKey.SEG_LOSS.value].grad_fn is not None
    assert metrics[MetricKey.CLS_LOSS.value].grad_fn is not None


def test_validation_step_integration(default_model, dummy_input):
    """Tests if the validation_step runs and detaches tensors."""
    # Create a dummy target dictionary
    dummy_targets = {
        "masks": torch.randint(0, 2, (BATCH_SIZE, 1, INPUT_H, INPUT_W), dtype=torch.float32),
        "labels": torch.randint(0, CLS_NUM_CLASSES_DEFAULT, (BATCH_SIZE,), dtype=torch.long),
    }
    dummy_batch = (dummy_input, dummy_targets)

    metrics = default_model.validation_step(dummy_batch)

    assert isinstance(metrics, dict)
    assert MetricKey.VAL_SEG_LOSS.value in metrics
    assert MetricKey.VAL_SEG_DICE.value in metrics
    assert MetricKey.VAL_CLS_LOSS.value in metrics
    assert MetricKey.VAL_CLS_ACC.value in metrics
    assert MetricKey.VAL_CLS_AUROC.value in metrics  # Added AUROC check

    # Check if tensors are detached (no grad_fn)
    assert metrics[MetricKey.VAL_SEG_LOSS.value].grad_fn is None
    assert metrics[MetricKey.VAL_CLS_LOSS.value].grad_fn is None
    assert metrics[MetricKey.VAL_SEG_DICE.value].grad_fn is None
    assert metrics[MetricKey.VAL_CLS_ACC.value].grad_fn is None
    assert metrics[MetricKey.VAL_CLS_AUROC.value].grad_fn is None  # AUROC state is detached
