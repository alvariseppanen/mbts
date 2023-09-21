"""
Implements image encoders
"""

from collections import OrderedDict

from torch import profiler

from models.common.model.layers import *

import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models
import torch.utils.model_zoo as model_zoo

import time

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class SkyEyeEncoder(nn.Module):
    """
    2D to 3D encoder
    """

    def __init__(
        self,
        resnet_layers=18,
        cp_location=None,
        freeze=False,
        num_ch_dec=None,
        d_out=128,
        scales=range(4)
    ):
        super().__init__()

        self.encoder = ResnetEncoder(resnet_layers, True, 1)
        self.num_ch_enc = self.encoder.num_ch_enc

        self.upsample_mode = 'nearest'
        self.d_out = d_out
        self.scales = scales

        # Backbone
        self.body = body
        self.body_depth = body_depth

        # Transformer
        self.voxel_grid = voxel_grid
        self.voxel_grid_depth = voxel_grid_depth

        # Heads
        self.vol_den_head = vol_den_head
        self.vol_sem_head = vol_sem_head

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        #st = time.time()
        with profiler.record_function("encoder_forward"):
            x = torch.cat([x * .5 + .5], dim=1)
            image_features = self.encoder(x)
            outputs = self.decoder(image_features)
            #sem_outputs = self.sem_decoder(image_features)
            sem_outputs = self.sem_decoder2(image_features, outputs)
            x = [outputs[("disp", i)] for i in self.scales]
            x.append(sem_outputs["1_1"])
        #print("Encoder", time.time() - st) # ~ 10 ms
        
        return x

    @classmethod
    def from_conf(cls, conf):
        return cls(
            cp_location=conf.get("cp_location", None),
            freeze=conf.get("freeze", False),
            num_ch_dec=conf.get("num_ch_dec", None),
            d_out=conf.get("d_out", 128),
            resnet_layers=conf.get("resnet_layers", 18)
        )
