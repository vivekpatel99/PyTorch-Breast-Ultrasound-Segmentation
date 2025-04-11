import torch

from src.models.basemodel import SegmentationBaseModel
from src.models.components.nets.fcns import FCN8Decoder
from src.models.components.nets.vgg_net_encoder import VGGNetEncoder


class VGGNetFCN8SegmentationModel(SegmentationBaseModel):
    def __init__(
        self,
        num_classes: int,
        segmentation_criterion: torch.nn.Module,
        classification_criterion: torch.nn.Module,
        vggnet_type: str = "vgg16",
    ) -> None:
        super().__init__(
            segmentation_criterion=segmentation_criterion,
            classification_criterion=classification_criterion,
        )

        self.encoder = VGGNetEncoder(pretrained_weights="DEFAULT", model=vggnet_type)
        self.decoder = FCN8Decoder(encoder=self.encoder, num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        x = self.decoder(x)
        return x
