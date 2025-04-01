import logging
from pathlib import Path

import hydra
import opendatasets as od
import pyrootutils
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision import io

log = logging.getLogger(__name__)


class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, dataset_url: str) -> None:
        self.label_mapping = {}

        # self.transform = transform

        # paths setup
        self.data_dir = Path(data_dir)
        self.root_data_dir = self.data_dir.parent
        self.dataset_url = dataset_url
        self.class_names = [_dir.stem for _dir in self.data_dir.iterdir()]
        self.num_classes = len(self.class_names)
        self.label_mapping = {name: i for i, name in enumerate(self.class_names)}

        if len(list(self.root_data_dir.iterdir())) == 0:
            self.download_dataset()
        self.images, self.masks, self.labels = self.get_data()

    def download_dataset(self) -> None:
        log.info(f"Downloading dataset from kaggle at {self.root_data_dir}")
        od.download(dataset_id_or_url=self.dataset_url, data_dir=str(self.root_data_dir))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img = io.read_image(str(self.images[index])) / 255.0
        mask = io.read_image(str(self.masks[index])) / 255.0

        # Convert label to numerical representation
        label = self.labels[index]
        label_index = self.label_mapping[label]

        # One-hot encode the label
        label_one_hot = F.one_hot(torch.tensor(label_index), num_classes=self.num_classes).float()

        target = {}
        target["masks"] = mask
        target["labels"] = label_one_hot
        return img, target

    def get_data(self) -> tuple[list[Path], list[Path], list[str]]:
        log.info(f"Getting data from {self.data_dir}")
        org_images = []
        masks = []
        labels = []

        for _dir in self.data_dir.iterdir():
            dir_name = _dir.stem
            img_nums = (img.stem.split("(")[-1].split(")")[0] for img in _dir.glob("*.png"))
            for num in img_nums:
                # normal (1).png
                img_path = f"{_dir/dir_name} ({num})"
                org_img_path = Path(f"{img_path}.png")
                # normal (2)_mask
                mask_path = Path(f"{img_path}_mask.png")
                org_images.append(org_img_path)
                labels.append(dir_name)
                masks.append(mask_path)

        return org_images, masks, labels


@hydra.main(version_base="1.2", config_path="../../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    dataset = hydra.utils.instantiate(cfg.data.dataset)
    for img, target in dataset:
        print(img.shape, target["masks"].shape, target["labels"].shape)
        break


if __name__ == "__main__":
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    # dataset = BreastCancerDataset(
    #     data_dir=root/'data/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT',
    #     dataset_url = 'https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset'
    #     )
    # for img, target in dataset:
    #     print(img.shape, target["masks"].shape, target["labels"].shape)
    #     break
    # import omegaconf
    # import pyrootutils
    # from omegaconf import DictConfig

    main()
    # cfg: DictConfig | omegaconf.ListConfig = omegaconf.OmegaConf.load(
    #     root / "configs" / "data"
    # )

    # dataset = hydra.utils.instantiate(cfg.dataset)
