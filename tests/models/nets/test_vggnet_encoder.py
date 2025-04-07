import pytest
import torch

from src.models.components.nets.vgg_net_encoder import VGGNetEncoder


@pytest.mark.parametrize(
    "model_name, expected_shapes",
    [
        (
            "vgg11",
            {
                "pool1": torch.Size([1, 64, 112, 112]),
                "pool2": torch.Size([1, 128, 56, 56]),
                "pool3": torch.Size([1, 256, 28, 28]),
                "pool4": torch.Size([1, 512, 14, 14]),
                "pool5": torch.Size([1, 4096, 7, 7]),
            },
        ),
        (
            "vgg11_bn",
            {
                "pool1": torch.Size([1, 64, 112, 112]),
                "pool2": torch.Size([1, 128, 56, 56]),
                "pool3": torch.Size([1, 256, 28, 28]),
                "pool4": torch.Size([1, 512, 14, 14]),
                "pool5": torch.Size([1, 4096, 7, 7]),
            },
        ),
        (
            "vgg13",
            {
                "pool1": torch.Size([1, 64, 112, 112]),
                "pool2": torch.Size([1, 128, 56, 56]),
                "pool3": torch.Size([1, 256, 28, 28]),
                "pool4": torch.Size([1, 512, 14, 14]),
                "pool5": torch.Size([1, 4096, 7, 7]),
            },
        ),
        (
            "vgg13_bn",
            {
                "pool1": torch.Size([1, 64, 112, 112]),
                "pool2": torch.Size([1, 128, 56, 56]),
                "pool3": torch.Size([1, 256, 28, 28]),
                "pool4": torch.Size([1, 512, 14, 14]),
                "pool5": torch.Size([1, 4096, 7, 7]),
            },
        ),
        (
            "vgg16",
            {
                "pool1": torch.Size([1, 64, 112, 112]),
                "pool2": torch.Size([1, 128, 56, 56]),
                "pool3": torch.Size([1, 256, 28, 28]),
                "pool4": torch.Size([1, 512, 14, 14]),
                "pool5": torch.Size([1, 4096, 7, 7]),
            },
        ),
        (
            "vgg16_bn",
            {
                "pool1": torch.Size([1, 64, 112, 112]),
                "pool2": torch.Size([1, 128, 56, 56]),
                "pool3": torch.Size([1, 256, 28, 28]),
                "pool4": torch.Size([1, 512, 14, 14]),
                "pool5": torch.Size([1, 4096, 7, 7]),
            },
        ),
        (
            "vgg19",
            {
                "pool1": torch.Size([1, 64, 112, 112]),
                "pool2": torch.Size([1, 128, 56, 56]),
                "pool3": torch.Size([1, 256, 28, 28]),
                "pool4": torch.Size([1, 512, 14, 14]),
                "pool5": torch.Size([1, 4096, 7, 7]),
            },
        ),
        (
            "vgg19_bn",
            {
                "pool1": torch.Size([1, 64, 112, 112]),
                "pool2": torch.Size([1, 128, 56, 56]),
                "pool3": torch.Size([1, 256, 28, 28]),
                "pool4": torch.Size([1, 512, 14, 14]),
                "pool5": torch.Size([1, 4096, 7, 7]),
            },
        ),
    ],
)
def test_vgg_net_encoder_output_shapes(model_name, expected_shapes):
    """
    Test the output shapes of VGGNetEncoder for different VGG models.
    """
    batch_size, n_class, h, w = 1, 3, 224, 224
    vgg_encoder = VGGNetEncoder(pretrained_weights="DEFAULT", model=model_name)
    input_tensor = torch.randn(batch_size, 3, h, w)
    outputs = vgg_encoder(input_tensor)

    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == set(expected_shapes.keys())

    for key, expected_shape in expected_shapes.items():
        assert outputs[key].shape == expected_shape, f"Shape mismatch for {key} in {model_name}"


def test_vgg_net_encoder_invalid_model():
    """
    Test that VGGNetEncoder raises a ValueError for an unsupported model.
    """
    with pytest.raises(ValueError):
        VGGNetEncoder(model="invalid_model")
