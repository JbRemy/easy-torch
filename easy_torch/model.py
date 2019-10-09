from typing import List, Optional, Union

import os
from copy import copy
import warning

import torch 
import torch.nn as nn

from tqdm import tqdm 
import numpy as np

from .network import Network
from .callbacks import schedulers, _callback
from .helpers.torch import get_device, set_seed

class Model(object):
    def __init__(self, layers: List[str], device: str="auto", 
                 seed: Optional[int]=None) -> None:
        """Defines architectural parameters of the network.

        The network is not build at this point because we need to see the
        data shape for that.

        Args:
            layers (list): A list of layers following the correct syntax.
            device (str): On which device sould the model be send. "cpu",
                "gpu", "auto".
            seed (int): The seed for initialisation of the parameters.
        """
        self._layers = layers
        self._device = get_device(device)
        self._initialisation_seed = seed if seed else np.random.randint()
        if seed and self._device == torch.device("cuda"):
            if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
                warning.warn(
                    """
                    You specified a seed but don't meet the conditions for
                    reproducibility when using the CDNN backend of pytorch. You
                    may want to add : "torch.backends.cudnn.deterministic =
                    True;  torch.backends.cudnn.benchmark = False" to the
                    begining of your script.
                    """,
                    Warning
                )

    def compile(self, optimizer: str, criteron: str, 
                optimizer_kwargs: Optional[dict]=None, 
                criteron_kwargs: Optional[dict]=None) -> None:
        """Defines the training aspects of the model.

        Once a model is compiled, we know what architecture to train on which
        criteron with which algorithm.
        The optimizer is not instanciated at this point becaus we need the
        network parameters to be initialized.

        Args:
            optimizer (str): The name of a torch.optim.optimizer.
                https://pytorch.org/docs/stable/optim.html
            criteron (str): The name of a troch.nn Loss function.
                https://pytorch.org/docs/stable/nn.html
            optimizer_kwargs (dict): arguments for said optimizer.
            criteron_kwargs (dict): arguments for said criteron.
        """
        self._criteron = getattr(nn, criteron)(**criteron_kwargs)
        self._optimizer_name = optimizer
        self._optimizer_kwargs = optimizer_kwargs

    def train(self, train_loader: torch.utils.data.DataLoader, 
              epochs: int,
              log_folder: Optional[str]=None, 
              log_freq: Optional[int]=100, 
              test_loader: Optional[torch.utils.data.DataLoader]=None, 
              test_freq: Optional[int]=1000,
              callbacks: Optional[list[Union[str, _callback._CallBack]]]=None) -> None:
        """Trains the network

        """
        set_seed(self._seed)
        self._network = Network(self._layers, self._device, train_loader_SHAPE)
        self._optimizer = getattr(torch.optim, self._optimizer_name)(
            self.network.parameters(),
            **self._optimizer_kwargs
        )

