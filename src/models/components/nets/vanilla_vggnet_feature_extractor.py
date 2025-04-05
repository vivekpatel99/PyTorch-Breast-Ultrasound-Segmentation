import torch

from src.models.components.nets.vggnet_utils import VGGConfig, VGGUtils


class VanillaVGGNetFeatureExtractor(torch.nn.Module, VGGConfig):
    def __init__(self, input_shape: tuple = (1, 256, 256), vgg_type: str = "vgg19"):
        """
        Initializes the VanillaVGGNet model as a feature extractor.

        Args:
            input_shape (tuple): The shape of the input tensor, default is (1, 256, 256).
            vgg_type (str): The type of VGGNet architecture to use, options are "vgg11", "vgg13", "vgg16", "vgg19", default is "vgg19".

        This constructor sets up the convolutional layers based on the specified VGG architecture.
        """

        super().__init__()
        self.in_channels = input_shape[0]
        self._conv_layers = VGGUtils._create_conv_layers(
            self.vgg_types[vgg_type], self.in_channels
        )

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C', H', W') where C', H', W' are the output channels, height, and width after the convolutional layers.
        """
        x = self._conv_layers(x)
        return x
