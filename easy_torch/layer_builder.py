
from typing import List, Optional, Union, Tuple
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# TODO At some point we need to split this class into several classes, one for
# every layer type. Too many repetitions in the builders

class LayerBuilder(object):
    """ A class used to build a layer

    Attributes:
        layers (list): A list of nn.Module
        parsed_name (dict): A dictionnary containning the informations on
            the layer. For more details see LayerBuild._ParseName()

    Methods:
        Build(input_shape: list) -> None
            Creates a list of torch nn.Module with respect to the parameters

    >>> builder = LayerBuilder("Conv-256x3x1x1-ReLU-BatchNorm")
    >>> print(builder.Build([None, 128, 16, 16]))
        (
            [None, 256, 16, 16], 
            [
                Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
                BatchNorm2d(256, eps=1e-05, momentum=0.1, 
                            affine=True, track_running_stats=True), ReLU()
            ]
        )
    """
    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): A definition of the layer in the following form: 
                "Type-Size-Activation-Options". For details about the format 
                for each layers see the corresponding layer builder.
                Availabe layer types:
                    - Conv (only 2d)
                    - Dropout
                    - Linear
                    - MaxPool
                    - AvgPool
                Options: 
                    - BatchNorm 

        Returns:
            None
        """
        self._ParseName(name)

        return None

    def Build(self, input_shape: List[int]) -> Tuple[List[int], List[nn.Module]]:
        """Creates a list of nn.Module

        Args:
            input_shape: (list) the shape of the input data of the layer.

        Returns: 
            (list) the new shape of the data.
            (list) List of modules.
        """
        self.layer = []
        if self.parsed_name["type"] == "Dropout":
            self.layer.append(nn.Dropout(self.parsed_name["size"]))

        if self.parsed_name["type"] == "Conv":
            input_shape, conv = self._BuildConv(input_shape)
            self.layer.append(conv)

        elif self.parsed_name["type"] == "Linear":
            if not len(input_shape) == 2:
                input_shape, flatten = self._BuildFlatten(input_shape)
                self.layer.append(flatten)

            input_shape, linear = self._BuildLinear(input_shape)
            self.layer.append(linear)

        elif self.parsed_name["type"] == "MaxPool":
            input_shape, pool = self._BuildMaxPool(input_shape)
            self.layer.append(pool)

        elif self.parsed_name["type"] == "AvgPool":
            input_shape, pool = self._BuildAvgPool(input_shape)
            self.layer.append(pool)

        else:
            return input_shape, [getattr(torch.nn, self.parsed_name["type"])()]

        if "options" in self.parsed_name:
            for opt in self.parsed_name["options"]:
                if opt == "BatchNorm":
                    self.layer.append(self._BuildBatchNorm(input_shape))

        if "activation" in self.parsed_name:
            self.layer.append(getattr(nn, self.parsed_name["activation"])())

        return input_shape, self.layer

    def _ParseName(self, name: str) -> None:
        """Parses the name of the layer and produces a dictionnary.

        Args:
            name (str): see LayerBuilder.__init__

        Returns:
            None
        """
        self.parsed_name = {}
        for i, value in enumerate(name.split("-")):
            if i == 0:
                self.parsed_name["type"] = value

            elif i == 1:
                self.parsed_name["kernel_size"] = [int(_) for _ in value.split("x")]

            elif i == 2:
                self.parsed_name["activation"] = value

            elif i >= 3:
                if "options" in self.parsed_name:
                    self.parsed_name["options"].append(value)

                else:
                    self.parsed_name["options"] = [value]

    def _BuildConv(self, input_shape: int) -> Tuple[Tuple[List[int], nn.Module]]:
        """Builds a convolutional layer

        Format of the describer (Activation and options appart):
            Conv - OutChanels x kernel_size x stride x padding
            Conv - OutChanels x kernel_size x stride 
            Conv - OutChanels x Kernel_size
            Conv - OutChanels 

            !! Up to this point it is only possible to define square kernels. !!
            !! Only 2d Convolutions implemented !!

        Args:
            input_shape (list): see LayerBuilder.Build

        Return:
            (list) the new shape of the data.
            (nn.Module) The corresponding module.
        """
        assert len(input_shape) == 4, "Only 2d convolutions implemented for now"
        if len(input_shape) == 4:
            kwargs = {"in_channels":input_shape[1], "out_channels":None, 
                  "kernel_size":3, "stride":1, "padding":1,
                  "dilation":1, "groups":1, "bias":True}
            if "kernel_size" in self.parsed_name:
                keys = ["out_channels", "kernel_size", "stride", "padding"]
                for i,v in enumerate(self.parsed_name["kernel_size"]):
                    kwargs[keys[i]] = v

            input_shape[1] = kwargs["out_channels"]
            input_shape[2] = self._ConvShapeModifier(input_shape[2], kwargs)
            input_shape[3] = self._ConvShapeModifier(input_shape[3], kwargs)

            return input_shape, nn.Conv2d(**kwargs)

    def _BuildMaxPool(self, input_shape: int) -> Tuple[Tuple[List[int], nn.Module]]:
        """Builds a Pooling layer

        Format of the describer (Activation and options appart):
            Conv - OutChanels x kernel_size x stride 
            Conv - OutChanels x Kernel_size
            Conv - OutChanels 

            !! Up to this point it is only possible to define square kernels. !!
            !! Only 2d Pooling implemented !!

        Args:
            input_shape (list): see LayerBuilder.Build

        Return:
            (list) the new shape of the data.
            (nn.Module) The corresponding module.
        """
        assert len(input_shape) == 4, "Only 2d max-pooling implemented for now"
        kwargs = {"kernel_size":2, "stride":1, "padding":0, "dilation":1,
                  "return_indices":False, "ceil_mode":False}
        if len(input_shape) == 4:
            if "kernel_size" in self.parsed_name:
                keys = ["kernel_size", "stride", "padding"]
                for i,v in enumerate(self.parsed_name["kernel_size"]):
                    kwargs[keys[i]] = v

            input_shape[2] = self._ConvShapeModifier(input_shape[2], kwargs)
            input_shape[3] = self._ConvShapeModifier(input_shape[3], kwargs)

            return input_shape, nn.MaxPool2d(**kwargs)

    def _BuildAvgPool(self, input_shape: int) -> Tuple[Tuple[List[int], nn.Module]]:
        """Builds an average pooling layer

        Format of the describer (Activation and options appart):
            MaxPool - kernel_size x stride x padding
            MaxPool - kernel_size x stride 
            MaxPool - Kernel_size

            !! Up to this point it is only possible to define square kernels. !!
            !! Only 2d Pooling implemented !!

        Args:
            input_shape (list): see LayerBuilder.Build

        Return:
            (list) the new shape of the data.
            (nn.Module) The corresponding module.
        """
        assert len(input_shape) == 4, "Only 2d avg-pooling implemented for now"
        kwargs = {"kernel_size":2, "stride":1, "padding":0,
                  "count_include_pad":True}
        if len(input_shape) == 4:
            if "kernel_size" in self.parsed_name:
                keys = ["kernel_size", "stride", "padding"]
                for i,v in enumerate(self.parsed_name["kernel_size"]):
                    kwargs[keys[i]] = v

            input_shape[2] = self._AvgShapeModifier(input_shape[2], kwargs)
            input_shape[3] = self._AvgShapeModifier(input_shape[3], kwargs)

            return input_shape, nn.AvgPool2d(**kwargs)

    def _BuildLinear(self, input_shape: int) -> Tuple[Tuple[List[int], nn.Module]]:
        """Builds a linear module.

        Format of the describer (Activation and options appart):
            Linear - OutUnits

        Args:
            input_shape (list): see LayerBuilder.Build

        Return:
            (list) the new shape of the data.
            (nn.Module) The corresponding module.
        """
        assert len(input_shape) == 2,\
                "You must flatten before passing data throug a Fully Connected layer"
        assert len(self.parsed_name["kernel_size"])==1,\
                "Too many specifications passed to Linear"
        kwargs = {"in_features":input_shape[1], 
                  "out_features":self.parsed_name["kernel_size"][0], "bias":True}
        input_shape[1] = self.parsed_name["kernel_size"][0]

        return input_shape, nn.Linear(**kwargs)

    def _BuildBatchNorm(self, input_shape: int) -> Tuple[Tuple[List[int], nn.Module]]:
        """Builds a Batch Normalizer

        Args:
            input_shape (list): see LayerBuilder.Build

        Return:
            (list) the new shape of the data.
            (nn.Module) The corresponding module.
        """
        if len(input_shape) == 4:
            return nn.BatchNorm2d(input_shape[1])

        elif len(input_shape) == 2:
            return nn.BatchNorm1d(input_shape[1])


    def _BuildFlatten(self, input_shape: int) -> Tuple[Tuple[List[int], nn.Module]]:
        """Builds a flatten module

        Args:
            input_shape (list): see LayerBuilder.Build

        Return:
            (list) the new shape of the data.
            (nn.Module) The corresponding module.
        """
        input_shape = [input_shape[0], np.prod(input_shape[1:])]

        return input_shape, Flatten()

    @staticmethod
    def _ConvShapeModifier(size, kwargs):
        """
        Computes the new shape of the image based on the details from
        https://pytorch.org/docs/stable/nn.html
        """
        size_ = copy(size)
        size_ += 2*kwargs["padding"]
        size_ -= kwargs["dilation"] * (kwargs["kernel_size"]-1) 
        size_ -= 1
        size_ /= kwargs["stride"]
        size_ += 1

        return int(np.floor(size_))

    @staticmethod
    def _AvgShapeModifier(size, kwargs):
        """
        Computes the new shape of the image based on the details from
        https://pytorch.org/docs/stable/nn.html
        """
        size_ = copy(size)
        size_ += 2*kwargs["padding"]
        size_ -= kwargs["kernel_size"]
        size_ /= kwargs["stride"]
        size_ += 1

        return int(np.floor(size_))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

if __name__ == "__main__":
    builder = LayerBuilder("Conv-256x3x1x1-ReLU-BatchNorm")
    print(builder.Build([None, 128, 16, 16]))
