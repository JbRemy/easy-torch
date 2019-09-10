
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .LayerBuilder import LayerBuilder

class Network(nn.Module):
    def __init__(self, layers, prunable=False, device="auto", 
                 input_shape=[None,3,32,32]):
        """
        layers: (list) Type-Size-Activation
            ex: Conv-256-ReLu, Linear-1024-ReLu-BatchNorm, Maxpool-2x2, 
            For details about the format for each layers see LayerBuilder.
        prunable: (Bool) if True, a method SetMask is added to all layers
        device: (str) if "auto" chooses GPU if available. 
            Other options "cpu" or "gpu"

        >>> architechture = ["Conv-64-ReLU-BatchNorm", "MaxPool-2x2",
        >>>                  "Conv-128-ReLU-BatchNorm", "Linear-10"]
        >>> Network(architechture)
        Network(
            (network): Sequential(
                (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
                (1): ReLU()
                (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
                (5): ReLU()
                (6): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (7): Flatten() 
                (8): Linear(in_features=21632, out_features=10, bias=True)
            )
        )
        """
        super(Network, self).__init__()
        self.network = self.BuildLayers(layers, input_shape)
        self.prunable = prunable
        if self.prunable:
            self.MakePrunable()

        self.GetDevice(device)
        self = super(Network, self).to(self.device)

    def forward(self, x):
        return self.network(x)

    def BuildLayers(self, layers_names, input_shape):
        """
        layers: see self.__init__
        input_shape: (list) ex: [None, 64, 64] for images of size 64x64
        """
        Layers = []
        for layer_name in layers_names:
            builder = LayerBuilder(layer_name)
            input_shape, layer = builder.Build(input_shape)
            Layers += layer

        return nn.Sequential(*Layers)

    def GetDevice(self, device):
        """
        device: see __init__
        """
        if device == "auto" and torch.cuda.is_available():
            self.device = torch.device("gpu")

        elif device == "auto" or device == "cpu":
            self.device = torch.device("cpu")

        elif device == "gpu":
            assert torch.cuda.is_available(), "Cude not available"
            self.device = torch.device("gpu")

    @staticmethod
    def MakePrunable():
        for cls in LAYERSCLASSES:
            @add_method(cls)
            def SetMask(mask):
                pass

def add_method(cls):
    """
    This decorator allows to dynamically add a method to a class
    source: https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
    """
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self 
        # but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator

if __name__ == "__main__":
    architechture = ["Conv-64-ReLU-BatchNorm", "MaxPool-2x2",
                     "Conv-128-ReLU-BatchNorm", "Linear-10"]
    network = Network(architechture)
    print(network)

