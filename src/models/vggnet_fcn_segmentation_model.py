import torch

from src.models.basemodel import SegmentationBaseModel
from src.models.components.classification_head import ClassificationHead
from src.models.components.nets.fcns import FCN8Decoder, FCN16Decoder
from src.models.components.nets.vgg_net_encoder import VGGNetEncoder


class VGGNetFCNSegmentationModel(SegmentationBaseModel):
    def __init__(
        self,
        segmentation_criterion: torch.nn.Module,
        classification_criterion: torch.nn.Module,
        seg_num_classes: int = 1,
        cls_num_classes: int = 3,
        vggnet_type: str = "vgg16",
        fcn_type: str = "fcn8",
    ) -> None:
        super().__init__(
            segmentation_criterion=segmentation_criterion,
            classification_criterion=classification_criterion,
        )
        # pool3_channels = 256 if 'bn' not in vggnet_type else 256 # Example for vgg16/19
        # pool4_channels = 512 if 'bn' not in vggnet_type else 512 # Example for vgg16/19
        pool5_channels = 4096  # From the final_conv in VGGNetEncoder

        self.encoder = VGGNetEncoder(pretrained_weights="DEFAULT", model=vggnet_type)
        if fcn_type.lower() == "fcn8":
            self.decoder = FCN8Decoder(seg_num_classes=seg_num_classes)
        elif fcn_type.lower() == "fcn16":
            self.decoder = FCN16Decoder(seg_num_classes=seg_num_classes)
        else:
            raise ValueError(f"Unknown FCN type: {fcn_type}")

        self.classification_head = ClassificationHead(
            in_channels=pool5_channels, num_classes=cls_num_classes  # Takes pool5 features
        )

    def forward(self, x) -> dict[str, torch.Tensor]:
        # 1. Get features from the encoder
        features = self.encoder(x)
        seg_logits = self.decoder(features)  # Output shape: [B, seg_num_classes, H, W]
        cls_logits = self.classification_head(
            features["pool5"]
        )  # Output shape: [B, cls_num_classes]
        return {"masks": seg_logits, "labels": cls_logits}
