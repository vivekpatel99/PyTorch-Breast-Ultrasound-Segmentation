import torch
import torch.nn as nn

from src.models.components.nets.vggnet_utils import VGGConfig, VGGUtils


class VanillaVGGNet(nn.Module, VGGConfig):
    def __init__(
        self, input_shape: tuple = (1, 256, 256), num_classes: int = 3, vgg_type: str = "vgg19"
    ):
        """
        Initializes the VanillaVGGNet model.

        Args:
            input_shape (tuple): The shape of the input tensor, default is (3, 256, 256).
            num_classes (int): The number of output classes, default is 3.
            vgg_type (str): The type of VGGNet architecture to use, options are "vgg11", "vgg13", "vgg16", "vgg19", default is "vgg19".

        This constructor sets up the convolutional layers based on the specified VGG architecture and initializes
        the classifier with fully connected layers tailored to the number of output classes.
        """

        super().__init__()

        self.in_channels = input_shape[0]
        self._conv_layers = VGGUtils._create_conv_layers(
            self.vgg_types[vgg_type], self.in_channels
        )
        # Determine the output shape of the convolutional layers automatically
        with torch.no_grad():
            dummy_input = torch.randn(1, self.in_channels, input_shape[1], input_shape[2])
            output = self._conv_layers(dummy_input)

            # output.size - [32,512,4,4]
            # output.view(output.size(0), -1).shape - [32, 512*4*4] (-1 mean use all elements/flatten)
            self.flattened_size = output.view(output.size(0), -1).shape[1]

        self._classifier = nn.Sequential(
            nn.Linear(in_features=self.flattened_size, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes)
        """
        x = self._conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self._classifier(x)
        return x
