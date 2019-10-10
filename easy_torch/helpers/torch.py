
from typing import List

import torch
import numpy as np

def get_device(self, device: Union[str, torch.device]) -> torch.device:
    """Returns the asked torch.device

    Args:
        device (str): The name of the required device "gpu", "cpu", "auto"

    Return: 
        device "torch.device"
    """
    if isinstance(device, torch.device):
        return torch.device

    if device == "auto":
        return torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")

    else:
        return torch.device(device)

def _set_seed(seed: int) -> None:
        """Sets the seed for torch, cuda, and numpy.
        Args:
            seed (int): The seed
        Returns:
            None
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

def send_to(tensors: List[torch.tensor], device: torch.device) -> List[torch.tensor]:
    """Sends tensors to the device
    
    Args:
        tensors (List)
        device (torch.device)
    
    return:
        tensors to device (list)
    """
    return [tesnor.to(device) for tensor in tensors]
