# /workspaces/PyTorch-Breast-Ultrasound-Segmentation/tests/models/nets/test_fcns.py
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
        cls_num_classes = 3
        seg_num_classes = 2  # Use a distinct value for segmentation classes
        decoder = decoder_class(
            encoder=encoder, cls_num_classes=cls_num_classes, seg_num_classes=seg_num_classes
        )
        assert decoder is not None
        assert decoder.cls_num_classes == cls_num_classes
        assert decoder.seg_num_classes == seg_num_classes  # Verify seg_num_classes is set

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass(self, decoder_class, encoder) -> None:
        """Test if the forward pass of the FCN decoder models works correctly."""
        cls_num_classes = 3
        seg_num_classes = 2  # Use a distinct value for segmentation classes
        decoder = decoder_class(
            encoder=encoder, cls_num_classes=cls_num_classes, seg_num_classes=seg_num_classes
        )
        input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1
        output = decoder(input_tensor)

        assert isinstance(output, dict)
        assert "masks" in output
        assert "labels" in output
        # Check masks output shape using seg_num_classes
        assert output["masks"].shape == (1, seg_num_classes, 224, 224)
        # Check labels output shape using cls_num_classes
        assert output["labels"].shape == (1, cls_num_classes)

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass_batch(self, decoder_class, encoder) -> None:
        """Test if the forward pass of the FCN decoder models works correctly with batch size > 1."""
        cls_num_classes = 3
        seg_num_classes = 2  # Use a distinct value for segmentation classes
        batch_size = 4
        decoder = decoder_class(
            encoder=encoder, cls_num_classes=cls_num_classes, seg_num_classes=seg_num_classes
        )
        input_tensor = torch.randn(batch_size, 3, 224, 224)  # Batch size of 4
        output = decoder(input_tensor)

        assert isinstance(output, dict)
        assert "masks" in output
        assert "labels" in output
        # Check masks output shape using seg_num_classes and batch_size
        assert output["masks"].shape == (
            batch_size,
            seg_num_classes,
            224,
            224,
        )
        # Check labels output shape using cls_num_classes and batch_size
        assert output["labels"].shape == (batch_size, cls_num_classes)

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass_cls_num_classes(self, decoder_class, encoder) -> None:
        """Test forward pass with different number of classification classes."""
        cls_num_classes = 5  # Vary classification classes
        seg_num_classes = 2  # Keep segmentation classes fixed for this test
        batch_size = 1
        decoder = decoder_class(
            encoder=encoder, cls_num_classes=cls_num_classes, seg_num_classes=seg_num_classes
        )
        input_tensor = torch.randn(batch_size, 3, 224, 224)  # Batch size of 1
        output = decoder(input_tensor)

        assert isinstance(output, dict)
        assert "masks" in output
        assert "labels" in output
        # Check masks output shape using seg_num_classes
        assert output["masks"].shape == (
            batch_size,
            seg_num_classes,
            224,
            224,
        )
        # Check labels output shape using the varied cls_num_classes
        assert output["labels"].shape == (batch_size, cls_num_classes)

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_forward_pass_seg_num_classes(self, decoder_class, encoder) -> None:
        """Test forward pass with different number of segmentation classes."""
        cls_num_classes = 3  # Keep classification classes fixed
        seg_num_classes = 4  # Vary segmentation classes
        batch_size = 1
        decoder = decoder_class(
            encoder=encoder, cls_num_classes=cls_num_classes, seg_num_classes=seg_num_classes
        )
        input_tensor = torch.randn(batch_size, 3, 224, 224)  # Batch size of 1
        output = decoder(input_tensor)

        assert isinstance(output, dict)
        assert "masks" in output
        assert "labels" in output
        # Check masks output shape using the varied seg_num_classes
        assert output["masks"].shape == (
            batch_size,
            seg_num_classes,
            224,
            224,
        )
        # Check labels output shape using cls_num_classes
        assert output["labels"].shape == (batch_size, cls_num_classes)

    @pytest.mark.parametrize("decoder_class", [FCN8Decoder, FCN16Decoder])
    def test_decoder_requires_encoder(self, decoder_class) -> None:
        """Test that the decoder requires an encoder instance."""
        with pytest.raises(TypeError):  # Expect a TypeError if encoder is not provided
            # Need to provide both class args now, even if checking for encoder presence
            decoder_class(cls_num_classes=3, seg_num_classes=1)

    # Optional: Add a test for encoder type if needed, though type hinting helps
    # def test_decoder_encoder_type(self):
    #     """Test decoder creation with a non-nn.Module encoder."""
    #     class NotAnEncoder:
    #         pass
    #     encoder_instance = NotAnEncoder()
    #     with pytest.raises(AttributeError): # Or TypeError depending on implementation details
    #         FCN8Decoder(encoder=encoder_instance, cls_num_classes=3, seg_num_classes=1)
