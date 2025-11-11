import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional

import torch
from torch import nn, Tensor

__all__ = ["RegNetX"]

# Con2d + BatchNorm + Activation
class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: 3,
        stride: 1,
        norm_layer: nn.BatchNorm2d,
        activation_layer: nn.ReLU,
        bias: False
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias
            )
        )
        
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        
        super().__init__(*layers)

class SqueezeExcitation(nn.Module): # The SE block lets the network learn which channels to emphasize
    def __init__()


class SimpleStemIN(Conv2dNormActivation):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: callable[..., nn.Module],
        activation_layer: callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in,
            width_out,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

class BottleneckTransform(nn.Module):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: callable[..., nn.Module],
        activation_layer: callable[..., nn.Module],
        group_width: int,  # It controls how many channels go into each group in the 3×3 convolution.
        bottleneck_multiplier: float, # It controls the width of the bottleneck layer.
        se_ratio: float # It controls the reduction ratio in the Squeeze-and-Excitation module.
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict() # layers of the bottleneck block
        width_bottleneck = int(width_out * bottleneck_multiplier) # width of the bottleneck layer
        groups = width_bottleneck // group_width # number of groups in the 3x3 conv
        
        # 1x1 conv
        layers["conv1"] = Conv2dNormActivation(
            width_in,
            width_bottleneck,
            kernel_size = 1,
            norm_layer = norm_layer,
            activation_layer = activation_layer
        )