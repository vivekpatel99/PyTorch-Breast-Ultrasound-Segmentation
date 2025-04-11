import pytest
import torch

from src.models.components.nets.fcns import FCN8Decoder, FCN16Decoder
from src.models.components.nets.vgg_net_encoder import VGGNetEncoder


class TestFCNDecoders:
    @pytest.fixture
    def encoder(self) -> VGGNetEncoder:
        """Fixture to create a VGGNetEncoder instance."""
        # Using vgg16 as it's a common choice and matches the decoder expectations
        return VGGNetEncoder(pretrained_weights="DEFAULT", model="vgg16")

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_creation(self, decoder_class, encoder) -> None:
        """Test if the FCN decoder models can be created."""
        num_classes = 3  # Default number of classes used in decoder if not specified
        decoder = decoder_class(encoder=encoder, num_classes=num_classes)
        assert decoder is not None
        assert decoder.num_classes == num_classes

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass(self, decoder_class, encoder) -> None:
        """Test if the forward pass of the FCN decoder models works correctly."""
        num_classes = 3  # Default number of classes
        decoder = decoder_class(encoder=encoder, num_classes=num_classes)
        input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1
        output = decoder(input_tensor)

        assert isinstance(output, dict)
        assert "masks" in output
        assert "labels" in output
        assert output["masks"].shape == (1, num_classes, 224, 224)  # Check masks output shape
        assert output["labels"].shape == (1, num_classes)  # Check labels output shape

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass_batch(self, decoder_class, encoder) -> None:
        """Test if the forward pass of the FCN decoder models works correctly with batch size > 1."""
        num_classes = 3  # Default number of classes
        batch_size = 4
        decoder = decoder_class(encoder=encoder, num_classes=num_classes)
        input_tensor = torch.randn(batch_size, 3, 224, 224)  # Batch size of 4
        output = decoder(input_tensor)

        assert isinstance(output, dict)
        assert "masks" in output
        assert "labels" in output
        assert output["masks"].shape == (
            batch_size,
            num_classes,
            224,
            224,
        )  # Check masks output shape
        assert output["labels"].shape == (batch_size, num_classes)  # Check labels output shape

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass_num_classes(self, decoder_class, encoder) -> None:
        """Test if the forward pass of the FCN decoder models works correctly with different number of classes."""
        num_classes = 5
        batch_size = 1
        decoder = decoder_class(encoder=encoder, num_classes=num_classes)
        input_tensor = torch.randn(batch_size, 3, 224, 224)  # Batch size of 1
        output = decoder(input_tensor)

        assert isinstance(output, dict)
        assert "masks" in output
        assert "labels" in output
        assert output["masks"].shape == (
            batch_size,
            num_classes,
            224,
            224,
        )  # Check masks output shape
        assert output["labels"].shape == (batch_size, num_classes)  # Check labels output shape

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_requires_encoder(self, decoder_class) -> None:
        """Test that the decoder requires an encoder instance."""
        with pytest.raises(TypeError):  # Expect a TypeError if encoder is not provided
            decoder_class(num_classes=3)

    # Optional: Add a test for encoder type if needed, though type hinting helps
    # def test_decoder_encoder_type(self):
    #     """Test decoder creation with a non-nn.Module encoder."""
    #     class NotAnEncoder:
    #         pass
    #     encoder_instance = NotAnEncoder()
    #     with pytest.raises(AttributeError): # Or TypeError depending on implementation details
    #         FCN8Decoder(encoder=encoder_instance, num_classes=3)
