import torch
import torch.nn as nn
from typing import Optional, Any

class RegularBatchNorm2d(nn.Module):

    def __init__(self, num_features: int,
                eps: float = 1e-5,
                momentum: Optional[float] = 0.1,
                affine: bool = True,
                track_running_stats = True,
                activation: str = "leaky_relu",
                activation_param: float = 0.01):

        super(RegularBatchNorm2d, self).__init__()

        self.bn = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                 track_running_stats=track_running_stats)
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=activation_param)
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.act(self.bn(x))

class RegularBatchNorm3d(nn.Module):

    def __init__(self, num_features: int,
                eps: float = 1e-5,
                momentum: Optional[float] = 0.1,
                affine: bool = True,
                track_running_stats = True,
                activation: str = "leaky_relu",
                activation_param: float = 0.01):

        super(RegularBatchNorm3d, self).__init__()

        self.bn = nn.BatchNorm3d(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                 track_running_stats=track_running_stats)
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=activation_param)
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.act(self.bn(x))
