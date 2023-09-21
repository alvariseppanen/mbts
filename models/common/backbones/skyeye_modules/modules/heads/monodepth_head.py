from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np


# ToDo: This is a slightly cleaned up version of the Monodpeth implementation. However, we could've
#  kept the initial version. We should do tests to make sure that the implementations are the same OR JUST
#
class MonodepthHead(nn.Module):

    def __init__(self, num_ch_enc, use_skips: bool):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.use_skips = use_skips
        self.scales = range(4)

        # Set up the up convolutions
        self.upconvs_0, self.upconvs_1, self.dispconvs = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()

        for i in range(4, -1, -1):
            # Upconv 0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs_0[str(i)] = nn.Sequential(
                nn.Conv2d(int(num_ch_in), int(num_ch_out), 3, stride=1, padding=1),
                nn.ELU(inplace=True))

            # Upconv 1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs_1[str(i)] = nn.Sequential(
                nn.Conv2d(int(num_ch_in), int(num_ch_out), 3, stride=1, padding=1),
                nn.ELU(inplace=True))

        self.depth = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(int(self.num_ch_dec[0]), 1, 3))
        nn.init.constant_(self.depth[1].bias, -2)

        # for s in self.scales:
        #     self.dispconvs[str(s)] = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(int(self.num_ch_dec[s]), 1, 3))
        # nn.init.constant_(self.dispconvs[-1][1].bias, -2)

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def disp_to_depth(disp: Tensor, min_depth: float = 0.1, max_depth: float = 100) -> Tensor:
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def scale_depth(self, depth_pred, min_depth=0.1, max_depth=100):
        depth_scaled = min_depth + (max_depth - min_depth) * depth_pred
        return depth_scaled

    def forward(self,
                in_feats: Tensor,
                return_disparity: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        self.outputs = {}

        # decoder
        x = in_feats[-1]
        for i in range(4, -1, -1):
            x = self.upconvs_0[str(i)](x)
            x = [F.interpolate(x, scale_factor=2, mode="nearest")]
            if self.use_skips and i > 0:
                x += [in_feats[i - 1]]
            x = torch.cat(x, 1)
            x = self.upconvs_1[str(i)](x)

        depth_feat = self.depth(x)
        depth_map = self.sigmoid(depth_feat)

        depth_map = self.scale_depth(depth_map)
        disp_map = 1.0 / depth_map

        if return_disparity:
            return depth_map, disp_map
        return depth_map
