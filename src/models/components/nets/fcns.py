import sys

import pyrootutils
import torch
import torch.nn as nn

from src.models.components.nets.vgg_net_encoder import VGGNetEncoder


class FCN16Decoder(nn.Module):
    def __init__(self, seg_num_classes: int = 1):
        super().__init__()
        self.seg_num_classes = seg_num_classes
        self.upsample_5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=4096,
                out_channels=512,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

        self.bn5 = nn.BatchNorm2d(512)
        self.upsample_4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        self.fcn16_mask_segmentation = nn.Conv2d(32, self.seg_num_classes, kernel_size=1)

    def forward(self, features) -> dict[str, torch.Tensor]:
        pool5 = features["pool5"]
        pool4 = features["pool4"]

        # First upsampling + skip connection with pool4
        upsampled_5 = self.upsample_5(pool5)

        x4 = self.bn5(upsampled_5 + pool4)

        # Second upsampling + skip connection with pool3
        #  # upsample the resulting tensor of the operation you just did
        upsample_4 = self.upsample_4(x4)

        # Progressive upsampling to original image size
        score = self.upsample3(upsample_4)
        score = self.upsample2(score)
        score = self.upsample1(score)

        return self.fcn16_mask_segmentation(score)


class FCN8Decoder(nn.Module):
    def __init__(self, seg_num_classes: int = 1):
        super().__init__()
        self.seg_num_classes = seg_num_classes
        self.upsample_5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=4096,
                out_channels=512,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.bn5 = nn.BatchNorm2d(512)
        self.upsample_4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.bn4 = nn.BatchNorm2d(256)

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                stride=2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        self.fcn8_mask_segmentation = nn.Conv2d(32, self.seg_num_classes, kernel_size=1)

    def forward(self, feaatures) -> dict[str, torch.Tensor]:
        pool5 = feaatures["pool5"]
        pool4 = feaatures["pool4"]
        pool3 = feaatures["pool3"]

        # First upsampling + skip connection with pool4
        upsampled_5 = self.upsample_5(pool5)

        x4 = self.bn5(upsampled_5 + pool4)

        # Second upsampling + skip connection with pool3
        #  # upsample the resulting tensor of the operation you just did
        upsample_4 = self.upsample_4(x4)

        # load the pool 3 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
        x3 = self.bn4(upsample_4 + pool3)

        # Progressive upsampling to original image size
        score = self.upsample3(x3)
        score = self.upsample2(score)
        score = self.upsample1(score)

        return self.fcn8_mask_segmentation(score)


if __name__ == "__main__":
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )
    sys.path.append(str(root))
    batch_size, n_class, h, w = 2, 3, 224, 224  # Standard VGG input size

    # Create models
    encoder = VGGNetEncoder(pretrained_weights="DEFAULT")
    fcn8 = FCN8Decoder(seg_num_classes=1)

    # Test input
    test_input = torch.randn(batch_size, 3, 224, 224)
    input_tensor = torch.autograd.Variable(test_input)
    output = fcn8(input_tensor)
