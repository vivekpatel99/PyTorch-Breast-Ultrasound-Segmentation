import torch
import torch.nn as nn


class VanillaVGGNet(nn.Module):
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
        self.vgg_types = {
            "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "vgg16": [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                "M",
            ],
            "vgg19": [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                512,
                "M",
            ],
        }
        self.in_channels = input_shape[0]
        self._conv_layers = self._create_conv_layers(self.vgg_types[vgg_type])
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

    def _create_conv_layers(self, architecture) -> nn.Sequential:
        """
        Creates a nn.Sequential container of convolutional layers based on the specified architecture.

        Args:
            architecture (list): A list of integers and strings where each integer represents the number of filters in a convolutional layer and each string represents a max pool layer. For example, [64, "M", 128, "M"].

        Returns:
            nn.Sequential: A nn.Sequential container of convolutional layers based on the specified architecture.
        """
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

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
