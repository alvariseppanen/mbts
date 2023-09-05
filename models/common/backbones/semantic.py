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


# Code taken from https://github.com/nianticlabs/monodepth2
#
# Godard, Clément, et al.
# "Digging into self-supervised monocular depth estimation."
# Proceedings of the IEEE/CVF international conference on computer vision.
# 2019.

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


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

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
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

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        #print(up_x.shape, concat_with.shape)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)
    
class UpSampleBN3(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN3, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with, concat_with2):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        #print(up_x.shape, concat_with.shape)
        f = torch.cat([up_x, concat_with, concat_with2], dim=1)
        return self._net(f)
    
class UpSampleBNwoS(nn.Module):
    def __init__(self, input_features, output_features):
        super(UpSampleBNwoS, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(x.shape[2]*2, x.shape[3]*2),
            mode="bilinear",
            align_corners=True,
        )
        
        return self._net(x)

class DecoderBN(nn.Module):
    def __init__(
        self, num_features, bottleneck_features, out_feature, use_decoder=True
    ):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.use_decoder = use_decoder

        self.conv2 = nn.Conv2d(
            bottleneck_features, features, kernel_size=1, stride=1, padding=1
        )

        self.out_feature_1_1 = out_feature
        self.out_feature_1_2 = out_feature
        self.out_feature_1_4 = out_feature
        self.out_feature_1_8 = out_feature
        self.out_feature_1_16 = out_feature
        self.feature_1_16 = features // 2
        self.feature_1_8 = features // 4
        self.feature_1_4 = features // 8
        self.feature_1_2 = features // 32
        self.feature_1_1 = features // 32

        if self.use_decoder:
            self.resize_output_1_1 = nn.Conv2d(
                self.feature_1_1, self.out_feature_1_1, kernel_size=1
            )
            self.resize_output_1_2 = nn.Conv2d(
                self.feature_1_2, self.out_feature_1_2, kernel_size=1
            )
            self.resize_output_1_4 = nn.Conv2d(
                self.feature_1_4, self.out_feature_1_4, kernel_size=1
            )
            self.resize_output_1_8 = nn.Conv2d(
                self.feature_1_8, self.out_feature_1_8, kernel_size=1
            )
            self.resize_output_1_16 = nn.Conv2d(
                self.feature_1_16, self.out_feature_1_16, kernel_size=1
            )

            self.up0 = UpSampleBNwoS(
                input_features=features, output_features=self.feature_1_8
            )

            self.up16 = UpSampleBN(
                skip_input=features + self.feature_1_16, output_features=self.feature_1_16
            )
            self.up8 = UpSampleBN(
                skip_input=self.feature_1_16 + self.feature_1_8, output_features=self.feature_1_8
            )
            self.up4 = UpSampleBN(
                skip_input=self.feature_1_8 + self.feature_1_4, output_features=self.feature_1_4
            )
            self.up2 = UpSampleBN(
                skip_input=self.feature_1_4 + self.feature_1_2, output_features=self.feature_1_2*2
            )
            self.up1 = UpSampleBNwoS(
                input_features=self.feature_1_2*2, output_features=self.feature_1_1
            )
        else:
            self.resize_output_1_1 = nn.Conv2d(3, out_feature, kernel_size=1)
            self.resize_output_1_2 = nn.Conv2d(32, out_feature * 2, kernel_size=1)
            self.resize_output_1_4 = nn.Conv2d(48, out_feature * 4, kernel_size=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[3],
            features[2],
            features[1],
            features[0],
        )
        #print(x_block0.shape, x_block1.shape, x_block2.shape, x_block3.shape, x_block4.shape)
        bs = x_block0.shape[0]

        x_d0 = self.conv2(x_block0)

        x_1_16 = self.up16(x_d0, x_block1)
        x_1_8 = self.up8(x_1_16, x_block2)
        x_1_4 = self.up4(x_1_8, x_block3)
        x_1_2 = self.up2(x_1_4, x_block4)
        x_1_1 = self.up1(x_1_2)
        
        return {
            "1_1": x_1_1,
            "1_2": x_1_2,
            "1_4": x_1_4,
            "1_8": x_1_8,
        }

class DecoderBN2(nn.Module):
    def __init__(
        self, num_features, bottleneck_features, out_feature, use_decoder=True
    ):
        super(DecoderBN2, self).__init__()
        features = int(num_features)
        self.use_decoder = use_decoder

        self.conv2 = nn.Conv2d(
            bottleneck_features, features, kernel_size=1, stride=1, padding=1
        )

        self.out_feature_1_1 = out_feature
        self.out_feature_1_2 = out_feature
        self.out_feature_1_4 = out_feature
        self.out_feature_1_8 = out_feature
        self.out_feature_1_16 = out_feature
        self.feature_1_16 = features // 2
        self.feature_1_8 = features // 4
        self.feature_1_4 = features // 8
        self.feature_1_2 = features // 32
        self.feature_1_1 = features // 32
        self.depth_feature = 64

        if self.use_decoder:
            self.resize_output_1_1 = nn.Conv2d(
                self.feature_1_1, self.out_feature_1_1, kernel_size=1
            )
            self.resize_output_1_2 = nn.Conv2d(
                self.feature_1_2, self.out_feature_1_2, kernel_size=1
            )
            self.resize_output_1_4 = nn.Conv2d(
                self.feature_1_4, self.out_feature_1_4, kernel_size=1
            )
            self.resize_output_1_8 = nn.Conv2d(
                self.feature_1_8, self.out_feature_1_8, kernel_size=1
            )
            self.resize_output_1_16 = nn.Conv2d(
                self.feature_1_16, self.out_feature_1_16, kernel_size=1
            )

            self.up0 = UpSampleBNwoS(
                input_features=features, output_features=self.feature_1_8
            )

            self.up16 = UpSampleBN(
                skip_input=features + self.feature_1_16, output_features=self.feature_1_16
            )
            self.up8 = UpSampleBN3(
                skip_input=self.feature_1_16 + self.feature_1_8 + self.depth_feature, output_features=self.feature_1_8
            )
            self.up4 = UpSampleBN3(
                skip_input=self.feature_1_8 + self.feature_1_4 + self.depth_feature, output_features=self.feature_1_4
            )
            self.up2 = UpSampleBN3(
                skip_input=self.feature_1_4 + self.feature_1_2 + self.depth_feature, output_features=self.feature_1_2*2
            )
            self.up1 = UpSampleBN(
                skip_input=self.feature_1_2*2 + self.depth_feature, output_features=self.feature_1_1
            )
            
        else:
            self.resize_output_1_1 = nn.Conv2d(3, out_feature, kernel_size=1)
            self.resize_output_1_2 = nn.Conv2d(32, out_feature * 2, kernel_size=1)
            self.resize_output_1_4 = nn.Conv2d(48, out_feature * 4, kernel_size=1)

    def forward(self, features, depth_features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[3],
            features[2],
            features[1],
            features[0],
        )
        #print(x_block0.shape, x_block1.shape, x_block2.shape, x_block3.shape, x_block4.shape)
        d_block0, d_block1, d_block2, d_block3 = (
            depth_features[("disp", 3)],
            depth_features[("disp", 2)],
            depth_features[("disp", 1)],
            depth_features[("disp", 0)],
        )
        #print(d_block0.shape, d_block1.shape, d_block2.shape, d_block3.shape)

        bs = x_block0.shape[0]

        x_d0 = self.conv2(x_block0)

        x_1_16 = self.up16(x_d0, x_block1)
        x_1_8 = self.up8(x_1_16, x_block2, d_block0)
        x_1_4 = self.up4(x_1_8, x_block3, d_block1)
        x_1_2 = self.up2(x_1_4, x_block4, d_block2)
        x_1_1 = self.up1(x_1_2, d_block3)
        
        return {
            "1_1": x_1_1,
            "1_2": x_1_2,
            "1_4": x_1_4,
            "1_8": x_1_8,
        }


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            # x = self.convs[("upconv", i, 0)](x)
            x = self.decoder[self.decoder_keys[("upconv", i, 0)]](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                if x[0].shape[2] > input_features[i - 1].shape[2]:
                    x[0] = x[0][:, :, :input_features[i - 1].shape[2], :]
                if x[0].shape[3] > input_features[i - 1].shape[3]:
                    x[0] = x[0][:, :, :, :input_features[i - 1].shape[3]]
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            #x = self.convs[("upconv", i, 1)](x)
            x = self.decoder[self.decoder_keys[("upconv", i, 1)]](x)

            self.outputs[("features", i)] = x

            if i in self.scales:
                #self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                self.outputs[("disp", i)] = self.sigmoid(self.decoder[self.decoder_keys[("dispconv", i)]](x))

        return self.outputs


class Decoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec=None, d_out=1, scales=range(4), use_skips=True):
        super(Decoder, self).__init__()

        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        if num_ch_dec is None:
            self.num_ch_dec = np.array([128, 128, 256, 256, 512])
        else:
            self.num_ch_dec = num_ch_dec
        self.d_out = d_out
        self.scales = scales

        self.num_ch_dec = [max(self.d_out, chns) for chns in self.num_ch_dec]

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.d_out)

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        with profiler.record_function("encoder_forward"):
            self.outputs = {}

            # decoder
            x = input_features[-1]
            for i in range(4, -1, -1):
                x = self.decoder[self.decoder_keys[("upconv", i, 0)]](x)

                x = [F.interpolate(x, scale_factor=(2, 2), mode="nearest")]

                if self.use_skips and i > 0:
                    feats = input_features[i - 1]

                    if x[0].shape[2] > feats.shape[2]:
                        x[0] = x[0][:, :, :feats.shape[2], :]
                    if x[0].shape[3] > feats.shape[3]:
                        x[0] = x[0][:, :, :, :feats.shape[3]]
                    x += [feats]
                x = torch.cat(x, 1)

                x = self.decoder[self.decoder_keys[("upconv", i, 1)]](x)

                self.outputs[("features", i)] = x

                if i in self.scales:
                    self.outputs[("disp", i)] = self.decoder[self.decoder_keys[("dispconv", i)]](x)

        return self.outputs


class SemanticSegmentor(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
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

        # decoder
        self.decoder = Decoder(num_ch_enc=self.num_ch_enc, d_out=self.d_out, num_ch_dec=num_ch_dec, scales=self.scales)
        self.num_ch_dec = self.decoder.num_ch_dec

        self.latent_size = self.d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

        if freeze:
            for p in self.parameters(True):
                p.requires_grad = False

        # semantic decoder
        self.sem_decoder = DecoderBN(out_feature=d_out,
                                     use_decoder=True,
                                     bottleneck_features=2048,
                                     num_features=2048,
        )
        self.sem_decoder2 = DecoderBN2(out_feature=d_out,
                                     use_decoder=True,
                                     bottleneck_features=2048,
                                     num_features=2048,
        )

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
