import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.data_util import make_test_dataset
from models.common.render import semNeRFRenderer
from models.bts.model.models_sbts import SBTSNet
from models.bts.model.ray_sampler import ImageRaySampler
from utils.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from utils.projection_operations import distance_to_z

import cv2
from PIL import Image

import time

IDX = 0
EPS = 1e-4

# The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that.
cam_incl_adjust = torch.tensor(
    [  [1.0000000,  0.0000000,  0.0000000, 0],
       [0.0000000,  0.9961947,  0.0871557, 0],
       [0.0000000, -0.0871557,  0.9961947, 0],
       [0.0000000,  000000000,  0.0000000, 1]
    ],
    dtype=torch.float32
).view(1, 1, 4, 4)

'''cam_incl_adjust = torch.tensor(
    [  [1.0000000,  0.0000000,  0.0000000, 0],
       [0.0000000,  1.0000000,  0.0000000, 0],
       [0.0000000,  0.0000000,  1.0000000, 0],
       [0.0000000,  000000000,  0.0000000, 1]
    ],
    dtype=torch.float32).view(1, 1, 4, 4)'''


def get_pts(x_range, y_range, z_range, ppm, ppm_y, y_res=None):
    x_res = abs(int((x_range[1] - x_range[0]) * ppm))
    if y_res is None:
        y_res = abs(int((y_range[1] - y_range[0]) * ppm_y))
    z_res = abs(int((z_range[1] - z_range[0]) * ppm))
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    if y_res == 1:
        y = torch.tensor([y_range[0] * .5 + y_range[1] * .5]).view(y_res, 1, 1).expand(-1, z_res, x_res)
    else:
        y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1)

    return xyz, (x_res, y_res, z_res)


# This function takes all points between min_y and max_y and projects them into the x-z plane.
# To avoid cases where there are no points at the top end, we consider also points that are beyond the maximum z distance.
# The points are then converted to polar coordinates and sorted by angle.

def get_lidar_slices(point_clouds, velo_poses, y_range, y_res, max_dist):
    slices = []
    ys = torch.linspace(y_range[0], y_range[1], y_res)
    if y_res > 1:
        slice_height = ys[1] - ys[0]
    else:
        slice_height = 0
    n_bins = 360

    for y in ys:
        if y_res == 1:
            min_y = y
            max_y = y_range[-1]
        else:
            min_y = y - slice_height / 2
            max_y = y + slice_height / 2

        slice = []

        for pc, velo_pose in zip(point_clouds, velo_poses):
            pc_world = (velo_pose @ pc.T).T

            mask = ((pc_world[:, 1] >= min_y) & (pc_world[:, 1] <= max_y)) | (torch.norm(pc_world[:, :3], dim=-1) >= max_dist)

            slice_points = pc[mask, :2]

            angles = torch.atan2(slice_points[:, 1], slice_points[:, 0])
            dists = torch.norm(slice_points, dim=-1)

            slice_points_polar = torch.stack((angles, dists), dim=1)
            # Sort by angles for fast lookup
            slice_points_polar = slice_points_polar[torch.sort(angles)[1], :]

            slice_points_polar_binned = torch.zeros_like(slice_points_polar[:n_bins, :])
            bin_borders = torch.linspace(-math.pi, math.pi, n_bins+1, device=slice_points_polar.device)

            dist = slice_points_polar[0, 1]

            # To reduce noise, we bin the lidar points into bins of 1deg and then take the minimum distance per bin.
            border_is = torch.searchsorted(slice_points_polar[:, 0], bin_borders)

            for i in range(n_bins):
                left_i, right_i = border_is[i], border_is[i+1]
                angle = (bin_borders[i] + bin_borders[i+1]) * .5
                if right_i > left_i:
                    dist = torch.min(slice_points_polar[left_i:right_i, 1])
                slice_points_polar_binned[i, 0] = angle
                slice_points_polar_binned[i, 1] = dist

            slice_points_polar = slice_points_polar_binned

            # Append first element to last to have full 360deg coverage
            slice_points_polar = torch.cat(( torch.tensor([[slice_points_polar[-1, 0] - math.pi * 2, slice_points_polar[-1, 1]]], device=slice_points_polar.device), slice_points_polar, torch.tensor([[slice_points_polar[0, 0] + math.pi * 2, slice_points_polar[0, 1]]], device=slice_points_polar.device)), dim=0)

            slice.append(slice_points_polar)

        slices.append(slice)

    return slices


def check_occupancy(pts, slices, velo_poses, min_dist=3):
    is_occupied = torch.ones_like(pts[:, 0])
    is_visible = torch.zeros_like(pts[:, 0], dtype=torch.bool)

    thresh = (len(slices[0]) - 2) / len(slices[0])

    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)

    world_to_velos = torch.inverse(velo_poses)

    step = pts.shape[0] // len(slices)

    for i, slice in enumerate(slices):
        for j, (lidar_polar, world_to_velo) in enumerate(zip(slice, world_to_velos)):
            pts_velo = (world_to_velo @ pts[i*step: (i+1)*step, :].T).T

            # Convert query points to polar coordinates in velo space
            angles = torch.atan2(pts_velo[:, 1], pts_velo[:, 0])
            dists = torch.norm(pts_velo, dim=-1)

            indices = torch.searchsorted(lidar_polar[:, 0].contiguous(), angles)

            left_angles = lidar_polar[indices-1, 0]
            right_angles = lidar_polar[indices, 0]

            left_dists = lidar_polar[indices-1, 1]
            right_dists = lidar_polar[indices, 1]

            interp = (angles - left_angles) / (right_angles - left_angles)
            surface_dist = left_dists * (1 - interp) + right_dists * interp

            is_occupied_velo = (dists > surface_dist) | (dists < min_dist)

            is_occupied[i*step: (i+1)*step] += is_occupied_velo.float()

            if j == 0:
                is_visible[i*step: (i+1)*step] |= ~is_occupied_velo

    is_occupied /= len(slices[0])

    is_occupied = is_occupied > thresh

    return is_occupied, is_visible


def project_into_cam(pts, proj, pose):
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)
    cam_pts = (proj @ (torch.inverse(pose).squeeze()[:3, :] @ pts.T)).T
    cam_pts[:, :2] /= cam_pts[:, 2:3]
    dist = cam_pts[:, 2]
    return cam_pts, dist


def plot(pts, xd, yd, zd):
    pts = pts.reshape(yd, zd, xd).cpu().numpy()

    rows = math.ceil(yd / 2)
    fig, axs = plt.subplots(rows, 2)

    for y in range(yd):
        r = y // 2
        c = y % 2

        if rows > 1:
            axs[r][c].imshow(pts[y], interpolation="none")
        else:
            axs[c].imshow(pts[y], interpolation="none")
    plt.show()


def plot_sperical(polar_pts):
    polar_pts = polar_pts.cpu()
    angles = polar_pts[:, 0]
    dists = polar_pts[:, 1]

    max_dist = dists.mean() * 2
    dists = dists.clamp(0, max_dist) / max_dist

    x = -torch.sin(angles) * dists
    y = torch.cos(angles) * dists

    plt.plot(x, y)
    plt.show()


def save(name, pts, xd, yd, zd):
    pts = pts.reshape(yd, zd, xd).cpu().numpy()[0]
    plt.imsave(name, pts)


def save_all(f, is_occupied, is_occupied_pred, images, xd, yd, zd):
    save(f"{f}_gt.png", is_occupied, xd, yd, zd)
    save(f"{f}_pred.png", is_occupied_pred, xd, yd, zd)
    plt.imsave(f"{f}_input.png", images[0, 0].permute(1, 2, 0).cpu().numpy() * .5 + .5)


class BTSWrapper(nn.Module):
    def __init__(self, renderer, config, dataset) -> None:
        super().__init__()

        self.renderer = renderer

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.query_batch_size = config.get("query_batch_size", 50000)
        self.occ_threshold = 0.5

        self.x_range = (-9, 9) # (-4, 4)
        self.y_range = (0.5, -1.5) # (0, .75)
        self.z_range = (21, 3) # (20, 4)
        self.ppm = 16
        self.ppm_y = 16

        self.y_res = 32

        self.sampler = ImageRaySampler(self.z_near, self.z_far, channels=3)

        self.dataset = dataset
        self.aggregate_timesteps = 20

        self.enc_type = config["encoder"]["type"]
        self.project_scale = 2
        if self.enc_type == "volumetric":
            self.project_scale = config["encoder"]["project_scale"]

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)                           # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)                 # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)                           # n, v, 4, 4 (-1, 1)
        index = data["index"].item()

        # added 
        projected_pix = data["projected_pix_{}".format(self.project_scale)] # n, h*w, 2    (only for single input image)
        fov_mask = data["fov_mask_{}".format(self.project_scale)]           # n, h*w       (only for single input image)
        gt_bev = torch.stack(data["bev"], dim=1)                           # n, v, 4, 4 (-1, 1)

        print(gt_bev.shape)
        print(gt_bev.min(), gt_bev.max())

        '''label_id = 6
        img_id = 0
        f_seg = gt_bev[0,img_id,...].clone()
        f_seg[f_seg == 255] = 0
        f_seg[f_seg == label_id] = 255
        #cv2.imshow('seg', f_seg.cpu().numpy().astype(np.uint8))
        #cv2.waitKey(2000)
        img2 = Image.fromarray(np.ones((3,200,200)), 'RGB')
        img2.show()'''

        seq, id, is_right = self.dataset._datapoints[index]
        seq_len = self.dataset._img_ids[seq].shape[0]

        n, v, c, h, w = images.shape
        device = images.device

        T_velo_to_pose = torch.tensor(self.dataset._calibs["T_velo_to_pose"], device=device)

        # Our coordinate system is at the same position as cam0, but rotated 5deg up along the x axis to adjust for camera inclination.
        # Consequently, the xz plane is parallel to the street.
        world_transform = torch.inverse(poses[:, :1, :, :])
        world_transform = cam_incl_adjust.to(device) @ world_transform
        poses = world_transform @ poses

        self.sampler.height = h
        self.sampler.width = w

        rays, _ = self.sampler.sample(None, poses[:, :1, :, :], projs[:, :1, :, :])

        #st = time.time()
        ids_encoder = [0]
        self.renderer.net.compute_grid_transforms(projs[:, ids_encoder], poses[:, ids_encoder])
        if self.enc_type == "volumetric":
            self.renderer.net.volume_encode(images, projected_pix, fov_mask, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder, images_alt=images * .5 + .5)
        else:
            self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder, images_alt=images * .5 + .5)
        self.renderer.net.set_scale(0)
        render_dict = self.renderer(rays, want_weights=True, want_alphas=True)
        #print("model inference time: ", time.time() - st)
        if "fine" not in render_dict:
            render_dict["fine"] = dict(render_dict["coarse"])
        render_dict = self.sampler.reconstruct(render_dict)
        pred_depth = distance_to_z(render_dict["coarse"]["depth"], projs[:1, :1])

        # Get pts
        q_pts, (xd, yd, zd) = get_pts(self.x_range, self.y_range, self.z_range, self.ppm, self.ppm_y, self.y_res)
        q_pts = q_pts.to(images.device).view(-1, 3)

        #print(xd,yd,zd)

        batch_size = 50000
        if q_pts.shape[1] > batch_size:
            sigmas = []
            invalid = []
            sems = []
            l = q_pts.shape[1]
            for i in range(math.ceil(l / batch_size)):
                f = i * batch_size
                t = min((i + 1) * batch_size, l)
                q_pts_ = q_pts[:, f:t, :]
                _, invalid_, sigmas_, sems_ = self.renderer.net.forward(q_pts_.unsqueeze(0))
                sigmas.append(sigmas_)
                invalid.append(invalid_)
                sems.append(sems_)
            sigmas = torch.cat(sigmas, dim=1)
            invalid = torch.cat(invalid, dim=1)
            sems = torch.cat(sems, dim=1)
        else:
            _, invalid, sigmas, sems = self.renderer.net.forward(q_pts.unsqueeze(0))

        #print(sems.shape)
        pred_class = sems.argmax(dim=2, keepdim=True)

        sigmas[torch.any(invalid, dim=-1)] = 1
        pred_class[torch.any(invalid, dim=-1)] = -1

        occupied_mask = sigmas > 0.5

        pred_class = pred_class.reshape(yd, xd, zd)
        occupied_mask = occupied_mask.reshape(yd, xd, zd)

        grid = torch.from_numpy(np.indices((yd, xd, zd))).cuda()
        ranking_grid = grid[0] 
        ranking_grid[~occupied_mask] = 1000
        _, first_occupied = torch.min(ranking_grid, dim=0, keepdim=True)
        pred_bev = torch.take_along_dim(pred_class, first_occupied.cuda(), dim=0).squeeze()


        print(pred_bev.shape)

        data["o_acc"] = 0
        data["o_rec"] = 0
        data["o_prec"] = 0
        data["ie_acc"] = 0
        data["ie_rec"] = 0
        data["ie_prec"] = 0
        data["ie_r"] = 0
        data["t_ie"] = 0
        data["t_no_nop_nv"] = 0

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        globals()["IDX"] += 1

        return data


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
    test_dataset = make_test_dataset(config["data"])
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=config["num_workers"], shuffle=False, drop_last=False)

    return test_loader


def get_metrics(config, device):
    names = ["o_acc", "o_prec", "o_rec", "ie_acc", "ie_prec", "ie_rec", "t_ie", "t_no_nop_nv"]
    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "SBTSNet")
    net = globals()[arch](config["model_conf"])
    renderer = semNeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = BTSWrapper(
        renderer,
        config["model_conf"],
        make_test_dataset(config["data"])
    )

    return model


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    pass