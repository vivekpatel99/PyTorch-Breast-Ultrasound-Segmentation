import logging

import torch

log = logging.getLogger(__name__)


def get_default_device() -> torch.device:
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device) -> torch.nn.Module | torch.Tensor | list | dict | tuple | None:
    """
    Move tensor(s) or dict/list/tuple of tensors to target device.

    Args:
        data: Data to move. Can be a tensor, list/tuple, or dictionary.
        device: Target device (e.g., torch.device('cuda')).

    Returns:
        Data structure with tensors moved to the specified device.
    """
    if isinstance(data, (list, tuple)):
        # If it's a list or tuple, apply to_device recursively to each element
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        # If it's a dictionary, apply to_device recursively to each value
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        # If it's a tensor, move it to the device
        # non_blocking=True can potentially speed up transfers for CUDA devices
        return data.to(device, non_blocking=True)
    # check if data is torch model
    elif isinstance(data, torch.nn.Module):
        return data.to(device)
    else:
        log.error(f"Unsupported data type: {type(data)}")
        raise Exception(f"Unsupported data type: {type(data)}")


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device) -> None:
        self.dl = dl
        self.device = device

    def __iter__(self):  # -> Generator[torch.Tensor | list | dict | tuple]:
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self) -> int:
        """Number of batches"""
        return len(self.dl)
