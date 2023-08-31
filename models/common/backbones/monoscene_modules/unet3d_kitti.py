# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.backbones.monoscene_modules.modules import SegmentationHead, DensityHead
from models.common.backbones.monoscene_modules.CRP3D import CPMegaVoxels
from models.common.backbones.monoscene_modules.modules import Process, Upsample, Downsample, UpsamplewoN, ProcesswoN, DownsamplewoN

from positional_encodings.torch_encodings import PositionalEncodingPermute3D
import time

class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size,
        feature,
        project_scale,
        context_prior=None,
        bn_momentum=0.1,
        beta=0.0,
        fourier_feats=39,
    ):
        super(UNet3D, self).__init__()
        self.business_layer = []
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature
        self.fourier_feats = fourier_feats

        size_l1 = (
            int(self.full_scene_size[0] / project_scale),
            int(self.full_scene_size[1] / project_scale),
            int(self.full_scene_size[2] / project_scale),
        )
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        #### >
        dilations = [1]
        self.process_l1 = nn.Sequential(
            ProcesswoN(self.feature, dilations=dilations),
            DownsamplewoN(self.feature),
        )
        ####
        self.process_l2 = nn.Sequential(
            ProcesswoN(self.feature * 2, dilations=dilations),
            DownsamplewoN(self.feature * 2),
        )

        self.up_13_l2 = UpsamplewoN(
            self.feature * 4, self.feature * 2
        )
        ##### >
        self.up_12_l1 = UpsamplewoN(
            self.feature * 2, self.feature
        )
        self.up_l1_lfull = UpsamplewoN(
            self.feature, self.feature // 2
        )
        #####
        '''self.up_l1_lfull = Upsample(
            self.feature, self.feature // 2, norm_layer, bn_momentum
        )'''
        '''self.up_l1_lfull = Upsample(
            self.feature, self.feature, norm_layer, bn_momentum
        )'''

        '''self.ssc_head = SegmentationHead(
            self.feature // 2, self.feature // 2, class_num, dilations
        )'''

        '''self.ssc_head = SegmentationHead(
            self.feature, self.feature // 2, class_num, dilations
        )'''
        #### >
        '''self.d_head = DensityHead(
            self.feature // 2 + self.feature // 2, self.feature // 2, 1, dilations, beta
        )'''
        ####
        '''self.context_prior = context_prior
        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature * 4, size_l3, bn_momentum=bn_momentum
            )'''
        #### >
        #self.p_enc_3d = PositionalEncodingPermute3D(self.feature // 2)
        ####

        # for tiny 3dnet
        '''self.conv1 = nn.ConvTranspose3d(self.feature,
                                        self.feature // 2,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        dilation=1,
                                        output_padding=1,
                                        )
        self.conv2 = nn.Conv3d(self.feature,
                               1,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               )
        self.act = nn.ReLU()'''

    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"]
        
        x3d_l2 = self.process_l1(x3d_l1)

        x3d_l3 = self.process_l2(x3d_l2)

        '''if self.context_prior:
            ret = self.CP_mega_voxels(x3d_l3)
            x3d_l3 = ret["x"]
            for k in ret.keys():
                res[k] = ret[k]'''

        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        ###########ssc_logit_full = self.ssc_head(x3d_up_lfull)

        ###########res["ssc_logit"] = ssc_logit_full

        # add pos encoding
        #p_encodings = self.p_enc_3d(torch.zeros_like(x3d_up_lfull))
        #x3d_up_lfull = torch.cat((x3d_up_lfull, p_encodings), dim=1)
        
        #d_full = self.d_head(x3d_up_lfull)
        
        # tiny 3dnet
        '''d_full = self.act(self.conv1(x3d_l1))
        p_encodings = self.p_enc_3d(torch.zeros_like(d_full))
        d_full = torch.cat((d_full, p_encodings), dim=1)
        d_full = self.conv2(d_full)'''
        ############
        
        return x3d_up_lfull
