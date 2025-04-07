from torch import nn


class VGGUtils:
    def __init__(self) -> None:
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

    def _create_conv_layers(self, architecture, in_channels) -> nn.Sequential:
        """
        Creates a nn.Sequential container of convolutional layers based on the specified architecture.

        Args:
            architecture (list): A list of integers and strings where each integer represents the number of filters in a convolutional layer and each string represents a max pool layer. For example, [64, "M", 128, "M"].

        Returns:
            nn.Sequential: A nn.Sequential container of convolutional layers based on the specified architecture.
        """
        layers = []
        in_channels = in_channels
        for x in self.vgg_types[architecture]:
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
