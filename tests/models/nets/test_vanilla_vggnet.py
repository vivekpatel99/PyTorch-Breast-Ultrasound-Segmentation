import pytest
import torch

from src.models.components.nets.vanilla_vggnet import VanillaVGGNet


class TestVanillaVGGNet:
    def test_vggnet_creation(self) -> None:
        """Test if the VGGNet model can be created with different configurations."""
        # Test with default parameters
        model = VanillaVGGNet()
        assert model is not None

        # Test with different input shape
        model = VanillaVGGNet(input_shape=(3, 128, 128))
        assert model is not None

        # Test with different number of classes
        model = VanillaVGGNet(num_classes=5)
        assert model is not None

        # Test with different vgg_type
        model = VanillaVGGNet(vgg_type="vgg11")
        assert model is not None
        model = VanillaVGGNet(vgg_type="vgg13")
        assert model is not None
        model = VanillaVGGNet(vgg_type="vgg16")
        assert model is not None

    def test_vggnet_forward_pass(self) -> None:
        """Test if the forward pass of the VGGNet model works correctly."""
        # Test with default parameters
        model = VanillaVGGNet()
        input_tensor = torch.randn(1, 1, 256, 256)  # Batch size of 1
        output = model(input_tensor)
        assert output.shape == (1, 3)  # Check output shape

        # Test with different input shape
        model = VanillaVGGNet(input_shape=(3, 128, 128))
        input_tensor = torch.randn(2, 3, 128, 128)  # Batch size of 2
        output = model(input_tensor)
        assert output.shape == (2, 3)  # Check output shape

        # Test with different number of classes
        model = VanillaVGGNet(num_classes=5)
        input_tensor = torch.randn(4, 1, 256, 256)  # Batch size of 4
        output = model(input_tensor)
        assert output.shape == (4, 5)  # Check output shape

        # Test with different vgg_type
        model = VanillaVGGNet(vgg_type="vgg11")
        input_tensor = torch.randn(1, 1, 256, 256)  # Batch size of 1
        output = model(input_tensor)
        assert output.shape == (1, 3)  # Check output shape

        model = VanillaVGGNet(vgg_type="vgg13")
        input_tensor = torch.randn(1, 1, 256, 256)  # Batch size of 1
        output = model(input_tensor)
        assert output.shape == (1, 3)  # Check output shape

        model = VanillaVGGNet(vgg_type="vgg16")
        input_tensor = torch.randn(1, 1, 256, 256)  # Batch size of 1
        output = model(input_tensor)
        assert output.shape == (1, 3)  # Check output shape

    def test_invalid_vgg_type(self) -> None:
        """Test if an invalid vgg_type raises a KeyError."""
        with pytest.raises(KeyError):
            VanillaVGGNet(vgg_type="vgg20")

    def test_conv_layers_output_shape(self) -> None:
        """Test if the output shape of the convolutional layers is correct."""
        model = VanillaVGGNet()
        input_tensor = torch.randn(1, 1, 256, 256)
        output = model._conv_layers(input_tensor)
        assert output.shape == (1, 512, 4, 4)

        model = VanillaVGGNet(vgg_type="vgg11")
        input_tensor = torch.randn(1, 1, 256, 256)
        output = model._conv_layers(input_tensor)
        assert output.shape == (1, 512, 4, 4)

        model = VanillaVGGNet(vgg_type="vgg13")
        input_tensor = torch.randn(1, 1, 256, 256)
        output = model._conv_layers(input_tensor)
        assert output.shape == (1, 512, 4, 4)

        model = VanillaVGGNet(vgg_type="vgg16")
        input_tensor = torch.randn(1, 1, 256, 256)
        output = model._conv_layers(input_tensor)
        assert output.shape == (1, 512, 4, 4)

    def test_classifier_output_shape(self) -> None:
        """Test if the output shape of the classifier is correct."""
        model = VanillaVGGNet()
        dummy_input = torch.randn(1, 512, 4, 4)
        flattened_input = dummy_input.view(dummy_input.size(0), -1)
        output = model._classifier(flattened_input)
        assert output.shape == (1, 3)

        model = VanillaVGGNet(num_classes=5)
        dummy_input = torch.randn(1, 512, 4, 4)
        flattened_input = dummy_input.view(dummy_input.size(0), -1)
        output = model._classifier(flattened_input)
        assert output.shape == (1, 5)
