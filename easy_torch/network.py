
from typing import List, Optional, Union, Tuple
from copy import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layer_builder import LayerBuilder

# TODO evaluate the interest of haveing a function initializing weights.
# Otherwise find a way to choose weights initialization

class Network(nn.Sequential):
    """A class to implement a neural network 

    Attributes:
        nn.Sequential attributes
        device (torch.device)

    Methods:
        nn.Sequential methods
        forward(x: torch.Tensor)

    >>> architechture = ["Conv-64-ReLU", "Conv-128-ReLU", "Linear-10"]
    >>> Network(architechture)
        Network(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU()
          (2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU()
          (4): Flatten()
          (5): Linear(in_features=131072, out_features=10, bias=True)
        )

    >>> architechture = ["Conv-64-ReLU", ["Conv-128-ReLU", "Conv-256-ReLU"], "Linear-10"]
    >>> Network(architechture)
        Network(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU()
          (2): Network(
            (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU()
            (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU()
            (_skip_conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (3): Flatten()
          (4): Linear(in_features=262144, out_features=10, bias=True)
        )
    """
    def __init__(self, layers: List[Union[str, List]], device: str="auto", 
                 input_shape: List[Union[None, int]]=[None,3,32,32], 
                 skip: bool=False) -> None:
        """Builds a network with respect to the layers provided.

        The network is build by creating a list of all requiered layers and
        intializing a nn.Sequential module on this list.
        See the documentation for the synthax on layers.
        If skip is True, a skip connection is build arround the network.
        If the output of the network doesn't has the same shape as the input, 
        a convolution layer is added to the skip connection to reshape the
        data.

        Args:
            layers (list): A list of layers to build in order. Each layer
                should have the format : Type-Size-Activation.
                ex: Conv-256-ReLu, Linear-1024-ReLu-BatchNorm, Maxpool-2x2. 
                For details about the format for each layers see LayerBuilder.
                Optionally, you can add skip connections by embedding a list
                into the layers list. 
                ex: [layer_1, [skiped_layer_1, skiped_layer_2], final_layer]
            device (str): if "auto" chooses GPU if available.  Other options "cpu" or "gpu"
                (default: "auto")
            input_shape (list): The input shape of the data in NCHW format.
                (default: [None,3,32,32])
            skip (bool): True if there is a skip arround the network.

        Returns:
            None
        """
        self._GetDevice(device)
        layers_modules = self._BuildLayers(layers, copy(input_shape))
        super(Network, self).__init__(*layers_modules)
        self._skip = skip
        if skip:
            input_filters = input_shape[1]
            output_filters = self._shape[1]
            if input_filters != output_filters:
                self._skip_conv = nn.Conv2d(input_filters, output_filters,
                                           kernel_size=1, stride=1)

            else:
                self._skip_conv = None

        self = super(Network, self).to(self.device)

    def forward(self, x: torch.Tensor):
        """Processes provided data through the network

        Args:
            x (torch.Tensor): Input data.

        Return:
            (torch.Tensor): The output of the network.
        """
        out = self.__class__.forward(self, x)
        if self._skip:
            if self._skip_conv:
                skip = self._skip_conv(x)

            else:
                skip = x

            out += skip

        return out

    def _BuildLayers(self, layers: List[Union["str", List]], 
                    input_shape: List[Union[None, int]]) -> List[nn.Module]:
        """Builds the layers of the network

        Args: 
            layers (list): see self.__init__
            input_shape (list): The input shape of the data in NCHW format.

        Returns:
            List of layers (list): A lit of nn.Module
        """
        Layers = []
        self._shape = input_shape
        for layer_name in layers:
            if isinstance(layer_name, list):
                temp = self.__class__(layer_name, device=self.device, 
                                       input_shape=self._shape,
                                       skip=True)
                Layers.append(temp)
                self._shape = temp._shape

            else:
                builder = LayerBuilder(layer_name)
                self._shape, layer = builder.Build(self._shape)
                Layers += layer

        return Layers

    def _GetDevice(self, device: Union[str, torch.device]) -> None:
        """Define the device of the network

        Args:
            device (str)

        Return: 
            None
        """
        if device == "auto" and torch.cuda.is_available():
            self.device = torch.device("gpu")

        elif device == "auto" or device == "cpu":
            self.device = torch.device("cpu")

        elif device == "gpu":
            assert torch.cuda.is_available(), "Cude not available"
            self.device = torch.device("gpu")

        elif isinstance(device, torch.device):
            self.device = device

if __name__ == "__main__":
    architechture = ["Conv-64-ReLU", "Conv-128-ReLU", "Linear-10"]
    network = Network(architechture)
    print(network)
    architechture = ["Conv-64-ReLU", ["Conv-128-ReLU", "Conv-256-ReLU"], "Linear-10"]
    network = Network(architechture)
    print(network)

