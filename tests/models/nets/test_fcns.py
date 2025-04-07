import pytest
import torch

from src.models.components.nets.fcns import FCN8Decoder, FCN16Decoder
from src.models.components.nets.vgg_net_encoder import VGGNetEncoder


class TestFCNDecoders:
    @pytest.fixture
    def encoder(self) -> VGGNetEncoder:
        """Fixture to create a VGGNetEncoder instance."""
        return VGGNetEncoder(pretrained_weights="DEFAULT", model="vgg16")

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_creation(self, decoder_class, encoder) -> None:
        """Test if the FCN decoder models can be created."""
        decoder = decoder_class(encoder=encoder)
        assert decoder is not None

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass(self, decoder_class, encoder) -> None:
        """Test if the forward pass of the FCN decoder models works correctly."""
        decoder = decoder_class(encoder=encoder)
        input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1
        output = decoder(input_tensor)
        assert output.shape == (1, 3, 224, 224)  # Check output shape

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass_batch(self, decoder_class, encoder) -> None:
        """Test if the forward pass of the FCN decoder models works correctly with batch size > 1."""
        decoder = decoder_class(encoder=encoder)
        input_tensor = torch.randn(4, 3, 224, 224)  # Batch size of 4
        output = decoder(input_tensor)
        assert output.shape == (4, 3, 224, 224)  # Check output shape

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass_num_classes(self, decoder_class, encoder) -> None:
        """Test if the forward pass of the FCN decoder models works correctly with different number of classes."""
        num_classes = 5
        decoder = decoder_class(encoder=encoder, num_classes=num_classes)
        input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1
        output = decoder(input_tensor)
        assert output.shape == (1, num_classes, 224, 224)  # Check output shape
