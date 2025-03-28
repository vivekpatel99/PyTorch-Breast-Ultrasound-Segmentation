import logging
import os
from pathlib import Path, PurePath

import opendatasets as od
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision import io

log = logging.getLogger(__name__)


class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        self.label_mapping = {}
        self.num_classes = 0
        # paths setup
        self.root_dir = Path(cfg.paths.root_dir)
        self.root_data_dir = Path(cfg.paths.root_data_dir)
        self.dataset_url = cfg.data.url
        self.data_dir = self.root_data_dir / Path(cfg.data.dataset_dir)

        if len(list(self.root_data_dir.iterdir())) == 0:
            self.download_dataset()
        self.images, self.masks, self.labels = self.get_data()

    def download_dataset(self) -> None:
        log.info(f"Downloading dataset from kaggle at {self.root_data_dir}")
        od.download(dataset_id_or_url=self.dataset_url, data_dir=str(self.root_data_dir))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img = self.images[index]
        mask = self.masks[index]
        label = self.labels[index]

        img = io.read_image(str(img))
        mask = io.read_image(str(mask))

        # Convert label to numerical representation
        label_index = self.label_mapping[label]

        # One-hot encode the label
        label_one_hot = F.one_hot(torch.tensor(label_index), num_classes=self.num_classes).float()

        # Ensure mask is a single channel (grayscale)
        # if mask.shape[0] > 1:
        #     mask = mask[0:1, :, :]

        target = {}
        target["mask"] = mask
        target["label"] = label_one_hot
        return img, target

    def get_data(self) -> tuple[list[Path], list[Path], list[str]]:
        log.info(f"Getting data from {self.data_dir}")
        org_images = []
        masks = []
        labels = []
        self.num_classes = 0
        for _dir in self.data_dir.iterdir():
            self.num_classes += 1
            dir_name = _dir.stem
            self.label_mapping[dir_name] = self.num_classes
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


if __name__ == "__main__":

    import pyrootutils
    from hydra import compose, initialize
    from omegaconf import DictConfig

    root: Path = pyrootutils.setup_root(
        search_from=Path().cwd(),
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )
    print(root)
    if os.getenv("DATA_ROOT") is None:
        os.environ["DATA_ROOT"] = f"{root}/data"

    with initialize(config_path="../../../configs", job_name="dataset-test", version_base=None):
        cfg: DictConfig = compose(config_name="train.yaml")
    print(cfg)
    dataset = BreastCancerDataset(cfg=cfg)
    for img, target in dataset:
        print(img.shape)
        print(target["mask"].shape)
        print(target["label"], target["label"].shape)
        break
