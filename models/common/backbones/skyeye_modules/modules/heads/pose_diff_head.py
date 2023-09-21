import torch
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import ABN
from collections import OrderedDict

class PoseDiffHead(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, stride=1):
        super(PoseDiffHead, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        self.num_frames_to_predict_for = 2

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(512, 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        pose_diff = self.transformation_from_parameters(axisangle[:, 0], translation[:, 0])

        return pose_diff

    def transformation_from_parameters(self, axisangle, translation):
        """Convert the network's (axisangle, translation) output into a 4x4 matrix
        """
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()

        if invert:
            R = R.transpose(1, 2)
            t *= -1

        T = self.get_translation_matrix(t)

        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)

        return M

    def rot_from_axisangle(self, vec):
        """Convert an axisangle rotation into a 4x4 transformation matrix
        Input 'vec' has to be Bx1x3
        """
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)

        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca

        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)

        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC

        rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device)

        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1

        return rot

    def get_translation_matrix(self, translation_vector):
        """
        Convert a translation vector into a 4x4 transformation matrixF
        """
        T = torch.zeros(translation_vector.shape[0], 4, 4, device=translation_vector.device)

        t = translation_vector.contiguous().view(-1, 3, 1)

        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t

        return T

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

