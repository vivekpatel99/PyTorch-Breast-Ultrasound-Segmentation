import torch

from src.models.basemodel import SegmentationBaseModel
from src.models.components.nets.fcns import FCN16Decoder
from src.models.components.nets.vgg_net_encoder import VGGNetEncoder


class VGGNetFCN16SegmentationModel(SegmentationBaseModel):
    def __init__(
        self,
        num_classes: int,
        vggnet_type: str = "vgg16",
        class_weights: torch.Tensor | None = None,
        segmentation_criterion=None,
        classification_criterion=None,
    ) -> None:
        super().__init__(
            class_weights=class_weights,
            segmentation_criterion=segmentation_criterion,
            classification_criterion=classification_criterion,
        )

        self.encoder = VGGNetEncoder(pretrained_weights="DEFAULT", model=vggnet_type)
        self.decoder = FCN16Decoder(encoder=self.encoder, num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        x = self.decoder(x)
        return x
