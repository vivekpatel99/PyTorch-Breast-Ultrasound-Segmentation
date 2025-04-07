import torch
from torch import nn

from src.models.components.nets.fcns import FCN16Decoder
from src.models.components.nets.vgg_net_encoder import VGGNetEncoder


class VGGNetFCNSegmentationModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.encoder = VGGNetEncoder(pretrained_weights="DEFAULT", model="vgg16")
        self.decoder = FCN16Decoder(encoder=self.encoder, num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        x = self.encoder(x)
        return x
