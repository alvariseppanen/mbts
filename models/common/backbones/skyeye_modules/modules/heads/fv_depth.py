import torch
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import ABN
from collections import OrderedDict

# TODO: We do not use skip connections yet. Might not make sense with the backbone anyways...
# ToDo: How to set the depth limit? (BEV up to 50, but depth map up to 100m)
class DepthHeadVoxel(nn.Module):
    def __init__(self, front_H_in=None, front_W_in=None, feat_scale=None, norm_act=ABN): # ToDo: Use feat scale img_scale here...
        super(DepthHeadVoxel, self).__init__()

        self.front_H_in = int(front_H_in * feat_scale)
        self.front_W_in = int(front_W_in * feat_scale)

        # Set up the up convolutions
        self.upconvs_0 = nn.ModuleDict()
        self.upconvs_1 = nn.ModuleDict()
        self.num_ch_dec = torch.tensor([16, 32, 64])

        self.upsample_mode = 'nearest'
        self.scales = range(4) # ToDo: First interp
        for i in range(len(self.num_ch_dec)-1, -1, -1):
            num_ch_in = 64 if i == len(self.num_ch_dec)-1 else self.num_ch_dec[i+1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs_0[str(i)] = nn.Sequential(nn.Conv2d(int(num_ch_in), int(num_ch_out), 3, stride=1, padding=1), nn.ELU(inplace=True))
            self.upconvs_1[str(i)] = nn.Sequential(nn.Conv2d(int(num_ch_out), int(num_ch_out), 3, stride=1, padding=1), nn.ELU(inplace=True))

        # "Scaling" convolutions after each layer, #ToDo: No multi-scale is used here, not necessary imo tbh
        self.dispconv = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(int(self.num_ch_dec[0]), int(1), 3), nn.ELU(inplace=True))

        self.flattener = nn.Sequential(nn.Conv3d(88, 1, 3, 1, 1), norm_act(1)) # ToDo: Keep voxel position, sum (on height dimension? 96 is hard coded here

        self.sigmoid = nn.Sigmoid()

    def disp_to_depth(self, disp, min_depth=0.1, max_depth=100):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def forward(self, feat_voxel):
        B, C, D, H, W = feat_voxel.shape
        feat_voxel = feat_voxel.permute(0, 2, 1, 3, 4).contiguous() # Shape: B, D, C, H, W # ToDo: Avoid permutation to keep voxel position...

        # Flatten the features
        feat_voxel = self.flattener(feat_voxel) # Shape: B, 1, C, H, W
        feat_voxel = feat_voxel.squeeze(1) # Shape: B, C, H, W

        # Here we resize the voxel features to the downscaled (feat_scale) FV image size again as the depth decoder should
        # output a full-scale depth map
        feat_voxel = F.interpolate(feat_voxel, size=(self.front_H_in, self.front_W_in), mode="bilinear", align_corners=False)

        feat = feat_voxel
        for i in range(len(self.num_ch_dec)-1, -1, -1):
            feat = self.upconvs_0[str(i)](feat)
            feat = [F.interpolate(feat, scale_factor=2, mode="nearest")]
            feat = torch.cat(feat, 1)
            feat = self.upconvs_1[str(i)](feat)

        feat = self.dispconv(feat)
        disp_map = self.sigmoid(feat)
        depth_map = self.disp_to_depth(disp_map)

        return depth_map


class FvDepthHeadDirect(nn.Module):
    def __init__(self, in_channels=160, hidden_channels=128, dilation=6, feat_scale=None, min_level=0, levels=4, norm_act=ABN):
        super(FvDepthHeadDirect, self).__init__()

        # Set up the up convolutions
        self.upconvs_0 = nn.ModuleDict()
        self.upconvs_1 = nn.ModuleDict()
        self.num_ch_dec = torch.tensor([512, 256, 128]) # ToDo: More layers maybe?

        self.upsample_mode = 'nearest'
        self.scales = range(4) # ToDo: First interp
        for i in range(len(self.num_ch_dec)):     # ToDo: Revert this
            num_ch_in = 512 if i == 0 else self.num_ch_dec[i-1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs_0[str(i)] = nn.Sequential(nn.Conv2d(int(num_ch_in), int(num_ch_out), 3, stride=1, padding=1), nn.ELU(inplace=True))
            self.upconvs_1[str(i)] = nn.Sequential(nn.Conv2d(int(num_ch_out), int(num_ch_out), 3, stride=1, padding=1), nn.ELU(inplace=True))

        # "Scaling" convolutions after each layer, #ToDo: No multi-scale is used here, not necessary imo tbh
        self.depth = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(int(self.num_ch_dec[-1]), 1, 3))
        self.uncer = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(int(self.num_ch_dec[-1]), 1, 3))

        self.sigmoid = nn.Sigmoid()

        # Set an initial bias for the depth_uncer
        nn.init.constant_(self.depth[1].bias, -2)

    def disp_to_depth(self, disp, min_depth=0.1, max_depth=100):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def scale_depth(self, depth_pred, min_depth=0.1, max_depth=100):
        depth_scaled = min_depth + (max_depth - min_depth) * depth_pred
        return depth_scaled

    def forward(self, feat_2d, return_disparity):
        # ToDO: NG: Order of layers: Interp->Conv rather than Conv->Interp
        feat = feat_2d
        for i in range(len(self.num_ch_dec)):
            feat = self.upconvs_0[str(i)](feat)
            feat = F.interpolate(feat, scale_factor=2, mode="nearest")
            feat = self.upconvs_1[str(i)](feat)

        depth_feat = self.depth(feat)
        depth_map = self.sigmoid(depth_feat)

        uncer_feat = self.uncer(feat)
        uncer_map = self.sigmoid(uncer_feat).clamp(1e-5, 1)

        # depth_map = self.disp_to_depth(disp_map)
        depth_map = self.scale_depth(depth_map)
        disp_map = 1.0 / depth_map

        if return_disparity:
            return depth_map, disp_map
        return depth_map
