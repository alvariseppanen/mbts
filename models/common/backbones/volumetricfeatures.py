import argparse
import mmcv
import os
import torch
import torch.nn as nn
import warnings
from torch import profiler
#from mmcv import Config, DictAction
#from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
#from mmdet3d.models import build_model
from mmdet3d.models.backbones import ResNet
from mmdet3d.models.necks import FPN
from surroundocc_modules import *
#from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

class VolumetricFeatures(nn.Module):
    """
    3D voxel image encoder
    """

    def __init__(
        self
    ):
        super().__init__()
        
        point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
        occ_size = [200, 200, 16]
        use_semantic = True
        _dim_ = [128, 256, 512]
        _ffn_dim_ = [256, 512, 1024]
        volume_h_ = [100, 50, 25]
        volume_w_ = [100, 50, 25]
        volume_z_ = [8, 4, 2]
        _num_points_ = [2, 4, 8]
        _num_layers_ = [1, 3, 6]

        self.img_backbone = ResNet(depth=101,
                                   num_stages=4,
                                   out_indices=(1,2,3),
                                   frozen_stages=1,
                                   norm_cfg=dict(type='BN2d', requires_grad=False),
                                   norm_eval=True,
                                   style='caffe',
                                   #with_cp=True, # using checkpoint to save GPU memory
                                   dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
                                   stage_with_dcn=(False, False, True, True))
        
        self.img_neck = FPN(in_channels=[512, 1024, 2048],
                            out_channels=512,
                            start_level=0,
                            add_extra_convs='on_output',
                            num_outs=3,
                            relu_before_extra_convs=True)
        
        self.deformable_attention=MSDeformableAttention3D(embed_dims=_dim_,
                                                          num_points=_num_points_,
                                                          num_levels=1)
        
        self.spatial_cross_attention = SpatialCrossAttention(pc_range=point_cloud_range,
                                                             deformable_attention=self.deformable_attention,
                                                             embed_dims=_dim_)
        
        self.transformerlayers=OccLayer(attn_cfgs=[self.spatial_cross_attention],
                                        feedforward_channels=_ffn_dim_,
                                        ffn_dropout=0.1,
                                        embed_dims=_dim_,
                                        conv_num=2,
                                        operation_order=('cross_attn', 'norm',
                                                        'ffn', 'norm', 'conv'))
        
        self.encoder=OccEncoder(num_layers=_num_layers_,
                                pc_range=point_cloud_range,
                                return_intermediate=False,
                                transformerlayers=self.transformerlayers)
        
        self.transformer_template=PerceptionTransformer(embed_dims=_dim_,
                                                        encoder=self.encoder)
        
        self.pts_bbox_head=OccHead(volume_h=volume_h_,
                                   volume_w=volume_w_,
                                   volume_z=volume_z_,
                                   num_query=900,
                                   num_classes=17,
                                   conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
                                   conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
                                   out_indices=[0, 2, 4, 6],
                                   upsample_strides=[1,2,1,2,1,2,1],
                                   embed_dims=_dim_,
                                   img_channels=[512, 512, 512],
                                   use_semantic=use_semantic,
                                   transformer_template=self.transformer_template)
        
        self.model = SurroundOcc(use_grid_mask=True,
                                 use_semantic=use_semantic,
                                 img_backbone=self.img_backbone,
                                 img_neck=self.img_neck,
                                 pts_bbox_head=self.pts_bbox_head)
        

    def forward(self, x, img_params):
        """
        For extracting voxel features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W, D)
        """
        with profiler.record_function("encoder_forward"):
            x = torch.cat([x * .5 + .5], dim=1)
            
            outputs = self.model(return_loss=True, rescale=True, **x)

            x = [outputs[("disp", i)] for i in self.scales]

        return x

if __name__ == '__main__':
    v = VolumetricFeatures()