import logging
import sys

import pyrootutils
import torch
import torch.nn as nn
from torchvision import models

log = logging.getLogger(__name__)


class VGGNetEncoder(nn.Module):
    def __init__(
        self,
        pretrained_weights: str | None = "DEFAULT",
        model: str = "vgg11",
        requires_grad: bool = True,
        remove_fc: bool = True,
        show_params: bool = False,
    ):
        super().__init__()
        self.model_name = model
        # Create a dictionary mapping model names to model constructors
        vgg_models = {
            "vgg11": models.vgg11,
            "vgg11_bn": models.vgg11_bn,
            "vgg13": models.vgg13,
            "vgg13_bn": models.vgg13_bn,
            "vgg16": models.vgg16,
            "vgg16_bn": models.vgg16_bn,
            "vgg19": models.vgg19,
            "vgg19_bn": models.vgg19_bn,
        }
        # Check if the model is supported
        if model not in vgg_models:
            log.error(f"Unsupported VGG model: {model}")
            raise ValueError(f"Unsupported VGG model: {model}")
        log.info(f"Loading VGG model: {model}")
        # Load the requested model

        if pretrained_weights:
            self.vgg = vgg_models[model](weights=pretrained_weights)
        else:
            self.vgg = vgg_models[model]()

        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.vgg.classifier

        # Dynamically find the indices of MaxPool2d layers
        self.pool_indices = []
        for i, layer in enumerate(self.vgg.features):
            if isinstance(layer, nn.MaxPool2d):
                self.pool_indices.append(i)

        if show_params:
            log.info(f"Pool indices for {model}: {self.pool_indices}")
            for i, layer in enumerate(self.vgg.features):
                log.info(f"Layer {i}: {layer}")

        # number of filters for the output convolutional layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding="same"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x) -> dict[str, torch.Tensor]:
        features = {}

        # Process each layer and store the output after each MaxPool2d
        for i, layer in enumerate(self.vgg.features):
            x = layer(x)
            if i in self.pool_indices:
                pool_idx = self.pool_indices.index(i) + 1
                features[f"pool{pool_idx}"] = x
        # input images are 224x224 pixels so they will be downsampled to 7x7 after the pooling layers above.
        # we can extract more features by chaining two more convolution layers.
        features["pool5"] = self.final_conv(x)

        return features


if __name__ == "__main__":
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )
    sys.path.append(str(root))
    batch_size, n_class, h, w = 2, 3, 224, 224
    vgg_encoder = VGGNetEncoder(pretrained_weights="DEFAULT", model="vgg11")
    input_tensor = torch.randn(batch_size, n_class, h, w)  # Example input
    outputs = vgg_encoder(input_tensor)

    # outputs will be a list of 5 tensors, each representing the output of a max-pooling layer
    # for i, output in enumerate(outputs.values()):
    #     print(f"Output from max-pooling layer {i+1}: {output.shape}")

    # pool5 shape torch.Size([1, 4096, 7, 7])
    # Output from max-pooling layer 1: torch.Size([1, 64, 112, 112])
    # Output from max-pooling layer 2: torch.Size([1, 128, 56, 56])
    # Output from max-pooling layer 3: torch.Size([1, 256, 28, 28])
    # Output from max-pooling layer 4: torch.Size([1, 512, 14, 14])
    # Output from max-pooling layer 5: torch.Size([1, 4096, 7, 7])
