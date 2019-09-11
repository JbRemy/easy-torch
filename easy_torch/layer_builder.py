
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class LayerBuilder(object):
    def __init__(self, name):
        """
        name: (str) A definition of the layer in the following form:
                    Type-Size-Activation-Options
                    ex: Conv-256-ReLU
                    For details about the format for each layers
                    see the corresponding layer builder.
                    Options can include: 
                        - BatchNorm 
        """
        self.ParseName(name)

    def Build(self, input_shape):
        """
        Builds the layer as a list of nn.modules
        input_shape: (list) the shape of the input data of the layer
        returns: (list) the new shape of the data,
                 (list) List of modules.
         >>> builder = LayerBuilder("Conv-256x3x1x1-ReLU-BatchNorm")
         >>> builder.Build([None, 128, 16, 16])
            (
                [None, 256, 16.0, 16.0], 
                [
                    Conv2d(128, 256, kernel_size=(3, 3), 
                           stride=(1, 1), padding=(1, 1)), 
                    ReLU(), 
                    BatchNorm2d(256, eps=1e-05, momentum=0.1, 
                                affine=True, track_running_stats=True)
                ]
            )
        """
        self.layer = []
        if self.parsed_name["type"] == "Conv":
            input_shape, conv = self.BuildConv(input_shape)
            self.layer.append(conv)

        elif self.parsed_name["type"] == "Linear":
            if not len(input_shape) == 2:
                input_shape, flatten = self.BuildFlatten(input_shape)
                self.layer.append(flatten)
            input_shape, linear = self.BuildLinear(input_shape)
            self.layer.append(linear)

        elif self.parsed_name ["type"] == "MaxPool":
            input_shape, pool = self.BuildMaxPool(input_shape)
            self.layer.append(pool)

        if "activation" in self.parsed_name:
            self.layer.append(getattr(nn, self.parsed_name["activation"])())

        if "options" in self.parsed_name:
            assert self.parsed_name["options"] == "BatchNorm",\
                    "Only BatchNorm is available as option"
            self.layer.append(self.BuildBatchNorm(input_shape))

        return input_shape, self.layer

    def BuildConv(self, input_shape):
        """
        Format of the describer (Activation and options appart):
            Conv - OutChanels x kernel_size x stride x padding
            Conv - OutChanels x kernel_size x stride 
            Conv - OutChanels x Kernel_size
            Conv - OutChanels 
            
            Up to this point it is only possible to define square kernels.
            ex: Conv-256
                Conv-256x3x1x1
        """
        assert len(input_shape) == 4, "Only 2d convolutions implemented for now"
        kwargs = {"in_channels":input_shape[1], "out_channels":None, 
                  "kernel_size":3, "stride":1, "padding":0,
                  "dilation":1, "groups":1, "bias":True, "padding_mode":'zeros'}
        if len(input_shape) == 4:
            keys = ["out_channels", "kernel_size", "stride", "padding"]
            for i,v in enumerate(self.parsed_name["kernel_size"]):
                kwargs[keys[i]] = v

            input_shape[1] = kwargs["out_channels"]
            input_shape[2] = self.ConvShapeModifier(input_shape[2], kwargs)
            input_shape[3] = self.ConvShapeModifier(input_shape[3], kwargs)

            return input_shape, nn.Conv2d(**kwargs)

    def BuildMaxPool(self, input_shape):
        """
        Format of the describer (Activation and options appart):
            MaxPool - kernel_size x stride x padding
            MaxPool - kernel_size x stride 
            MaxPool - Kernel_size
            
            Up to this point it is only possible to define square kernels.
            ex: Conv-256
                Conv-256x3x1x1
        """
        assert len(input_shape) == 4, "Only 2d max-pooling implemented for now"
        kwargs = {"kernel_size":2, "kernel_size":None, "stride":1, "padding":0,
                  "dilation":1, "return_indices":False, "ceil_mode":False}
        if len(input_shape) == 4:
            keys = ["kernel_size", "stride", "padding"]
            for i,v in enumerate(self.parsed_name["kernel_size"]):
                kwargs[keys[i]] = v

            input_shape[2] = self.ConvShapeModifier(input_shape[2], kwargs)
            input_shape[3] = self.ConvShapeModifier(input_shape[3], kwargs)

            return input_shape, nn.MaxPool2d(**kwargs)
        
    def BuildLinear(self, input_shape):
        """
        Format of the describer (Activation and options appart):
            Linear - OutUnits

            ex: Linear-1024
        """
        assert len(input_shape) == 2,\
                "You must flatten before passing data throug a Fully Connected layer"
        assert len(self.parsed_name["kernel_size"])==1,\
                "Too many specifications passed to Linear"
        kwargs = {"in_features":input_shape[1], 
                  "out_features":self.parsed_name["kernel_size"][0], "bias":True}
        input_shape[1] = self.parsed_name["kernel_size"][0]

        return input_shape, nn.Linear(**kwargs)

    def BuildBatchNorm(self, input_shape):
        """
        Returns the correct type of 
        """
        if len(input_shape) == 4:
            return nn.BatchNorm2d(input_shape[1])

        elif len(input_shape) == 3:
            return nn.BatchNorm1d(input_shape[1])

    def ParseName(self, name):
        """
        name: see self.__init__
        """
        self.parsed_name = {}
        for i, value in enumerate(name.split("-")):
            if i == 0:
                self.parsed_name["type"] = value

            elif i == 1:
                self.parsed_name["kernel_size"] = [int(_) for _ in value.split("x")]

            elif i == 2:
                self.parsed_name["activation"] = value

            elif i == 3:
                self.parsed_name["options"] = value

    def BuildFlatten(self, input_shape):
        input_shape = [input_shape[0], np.prod(input_shape[1:])]
        return input_shape, Flatten()

    @staticmethod
    def ConvShapeModifier(size, kwargs):
        """
        Computes the new shape of the image based on the details from
        https://pytorch.org/docs/stable/nn.html
        """
        size += 2*kwargs["padding"]
        size -= kwargs["dilation"] * (kwargs["kernel_size"]-1) 
        size -= 1
        size /= kwargs["stride"]
        size += 1

        return int(np.floor(size))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

if __name__ == "__main__":
    builder = LayerBuilder("Conv-256x3x1x1-ReLU-BatchNorm")
    print(builder.Build([None, 128, 16, 16]))
