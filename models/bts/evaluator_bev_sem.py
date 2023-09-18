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

def get_pts2(x_range, y_range, z_range, x_res, y_res, z_res, cam_incl_adjust=None):
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that.
    if cam_incl_adjust is not None:
        xyz = xyz.view(-1, 3)
        xyz_h = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1)
        xyz_h = (cam_incl_adjust.squeeze() @ xyz_h.mT).mT
        xyz = xyz_h[:, :3].view(y_res, z_res, x_res, 3)

    return xyz


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

        self.x_range = (-28.57, 28.57) # (-26.19, 26.19) # (-28.57, 28.57)
        self.y_range = (0.5, -1.5) # (0, .75)
        self.z_range = (52.38, 0) # (57.14, 0) # (52.38, 0)
        self.ppm = 13.45
        self.ppm_y = 10

        self.y_res = 16

        self.sampler = ImageRaySampler(self.z_near, self.z_far, channels=3)

        self.dataset = dataset
        self.aggregate_timesteps = 20

        self.enc_type = config["encoder"]["type"]
        self.project_scale = 2
        if self.enc_type == "volumetric":
            self.project_scale = config["encoder"]["project_scale"]

        self.ignore_index = 255
        self.num_classes = config["mlp_class"]["n_classes"]
        self.num_classes = 8

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]
    
    def _confusion_matrix(self, sem_pred, sem):
        confmat = sem[0].new_zeros(self.num_classes * self.num_classes, dtype=torch.float)

        for sem_pred_i, sem_i in zip(sem_pred, sem):
            valid = sem_i != self.ignore_index
            if valid.any():
                sem_pred_i = sem_pred_i.numpy()
                sem_i = sem_i.numpy()
                valid = valid.numpy()

                sem_pred_i = sem_pred_i[valid]
                sem_i = sem_i[valid]

                sem_pred_i = torch.from_numpy(sem_pred_i)
                sem_i = torch.from_numpy(sem_i)
                valid = torch.from_numpy(valid)

                #print(confmat.new_ones(sem_i.numel()).shape[0], len(sem_i.view(-1) * self.num_classes + sem_pred_i.view(-1)))

                confmat.index_add_(0, sem_i.view(-1) * self.num_classes + sem_pred_i.view(-1), confmat.new_ones(sem_i.numel()))

        return confmat.view(self.num_classes, self.num_classes)

    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)                           # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)                 # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)                           # n, v, 4, 4 (-1, 1)
        index = data["index"].item()

        # added 
        projected_pix = data["projected_pix_{}".format(self.project_scale)] # n, h*w, 2    (only for single input image)
        fov_mask = data["fov_mask_{}".format(self.project_scale)]           # n, h*w       (only for single input image)
        gt_bev = torch.stack(data["bev"], dim=1).squeeze()                           # n, v, 4, 4 (-1, 1)

        #print(gt_bev.shape)
        #print(set(gt_bev.cpu().numpy().reshape(-1)))

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

        ids_encoder = [0]
        if self.enc_type == "volumetric":
            self.renderer.net.volume_encode(images, projected_pix, fov_mask, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder, images_alt=images * .5 + .5)
        else:
            self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder, images_alt=images * .5 + .5)
        self.renderer.net.set_scale(0)

        x_res = int((abs(self.x_range[0]) + abs(self.x_range[1])) * self.ppm)
        y_res = int((abs(self.y_range[0]) + abs(self.y_range[1])) * self.ppm_y)
        z_res = int((abs(self.z_range[0]) + abs(self.z_range[1])) * self.ppm)
        q_pts = get_pts2(self.x_range, self.y_range, self.z_range, x_res, y_res, z_res)
        q_pts = q_pts.to(device).view(1, -1, 3)

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
                _, invalid_, sigmas_, sems_ = self.renderer.net(q_pts_)
                sigmas.append(sigmas_)
                invalid.append(invalid_)
                sems.append(sems_)
            sigmas = torch.cat(sigmas, dim=1)
            invalid = torch.cat(invalid, dim=1)
            sems = torch.cat(sems, dim=1)
        else:
            _, invalid, sigmas, sems = self.renderer.net(q_pts)

        #print(sems.shape)
        pred_class = sems.argmax(dim=2, keepdim=True)

        sigmas[torch.any(invalid, dim=-1)] = 1
        pred_class[torch.any(invalid, dim=-1)] = -1
        
        occupied_mask = sigmas > self.occ_threshold

        pred_class = pred_class.reshape(y_res, z_res, x_res)
        occupied_mask = occupied_mask.reshape(y_res, z_res, x_res)

        grid = torch.from_numpy(np.indices((y_res, z_res, x_res))).cuda()
        ranking_grid = grid[0] 
        ranking_grid[~occupied_mask] = 1000
        _, first_occupied = torch.min(ranking_grid, dim=0, keepdim=True)
        pred_bev = torch.take_along_dim(pred_class, first_occupied.cuda(), dim=0).squeeze()

        pred_bev = torch.rot90(pred_bev, k=-1, dims=[0,1])

        # remap predictions
        new_pred_bev = torch.ones_like(pred_bev)*255
        new_pred_bev[pred_bev == 0] = 0
        new_pred_bev[pred_bev == 1] = 1
        new_pred_bev[pred_bev == 2] = 2
        new_pred_bev[pred_bev == 5] = 3
        new_pred_bev[pred_bev == 6] = 4
        new_pred_bev[pred_bev == 9] = 5
        new_pred_bev[pred_bev == 7] = 6
        new_pred_bev[pred_bev == 8] = 7
        new_pred_bev[pred_bev == -1] = 255

        gt_bev[new_pred_bev == 255] = 255

        '''ttt = torch.ones((20,20))
        ttt[0:10, :] = 0
        print(ttt)'''

        # crop area of interest
        '''crop_from_side = 400
        crop_from_top = 300
        gt_bev[0:crop_from_top, :] = 255
        gt_bev[gt_bev.shape[0]-crop_from_top:gt_bev.shape[0], :] = 255
        gt_bev[0:crop_from_top, :] = 255
        gt_bev[:, gt_bev.shape[1]-crop_from_side:gt_bev.shape[1]] = 255'''

        #print(pred_bev.shape)
        #print(set(pred_bev.cpu().numpy().reshape(-1)))
        '''if torch.count_nonzero(gt_bev == 4) > 0:
            v_pred_bev = new_pred_bev.clone()
            v_gt_bev = gt_bev.clone()
            v_pred_bev[v_pred_bev == 255] = -1
            v_gt_bev[v_gt_bev == 255] = -1
            color_lut = torch.tensor([[128, 64,128],
                                [244, 35,232],
                                [ 70, 70, 70],
                                [153,153,153],
                                [107,142, 35],
                                [ 70,130,180],
                                [220, 20, 60],
                                [  0,  0,142],
                                [  0,  0, 70],
                                [  0,  0,230],
                                [  0,  0,  0]]).cuda()
            r_profile = color_lut[:, 0][v_pred_bev.long()][:,:, None]
            g_profile = color_lut[:, 1][v_pred_bev.long()][:,:, None]
            b_profile = color_lut[:, 2][v_pred_bev.long()][:,:, None]
            pred_profile = torch.cat((r_profile, g_profile, b_profile), dim=2).squeeze().cpu().numpy()
            r_profile = color_lut[:, 0][v_gt_bev.long()][:,:, None]
            g_profile = color_lut[:, 1][v_gt_bev.long()][:,:, None]
            b_profile = color_lut[:, 2][v_gt_bev.long()][:,:, None]
            gt_profile = torch.cat((r_profile, g_profile, b_profile), dim=2).squeeze().cpu().numpy()
            print(pred_profile.shape, gt_profile.shape)
            plt.imshow(gt_profile)
            plt.show()
            plt.imshow(pred_profile)
            plt.show()'''
        #cv2.imwrite("/home/seppanen/test/" + f"{index:010d}_pred.png", cv2.cvtColor((pred_profile * 1).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        #cv2.imwrite("/home/seppanen/test/" + f"{index:010d}_gt.png", cv2.cvtColor((gt_profile * 1).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        #bev_ignore_classes = varargs['bev_ignore_classes']
        bev_ignore_classes = []
        bev_num_classes_ignore = self.num_classes - len(bev_ignore_classes)
        bev_num_classes = self.num_classes

        bev_semantic_conf_mat = self._confusion_matrix(new_pred_bev.squeeze().cpu(), gt_bev.squeeze().cpu())
        bev_semantic_conf_mat = bev_semantic_conf_mat.to(device)
        #if not varargs['debug']:
        #    distributed.all_reduce(bev_semantic_conf_mat, distributed.ReduceOp.SUM)
        bev_semantic_conf_mat = bev_semantic_conf_mat.cpu()[:bev_num_classes, :]
        # Remove specific rows
        bev_keep_matrix = torch.ones_like(bev_semantic_conf_mat, dtype=torch.bool)
        bev_keep_matrix[bev_ignore_classes, :] = False
        bev_keep_matrix[:, bev_ignore_classes] = False
        bev_semantic_conf_mat = bev_semantic_conf_mat[bev_keep_matrix].view(bev_keep_matrix.shape[0] - len(bev_ignore_classes),
                                                                    bev_keep_matrix.shape[1] - len(bev_ignore_classes))
        bev_sem_intersection = bev_semantic_conf_mat.diag()
        bev_sem_union = ((bev_semantic_conf_mat.sum(dim=1) + bev_semantic_conf_mat.sum(dim=0)[:bev_num_classes_ignore] - bev_semantic_conf_mat.diag()) + 1e-8)
        bev_sem_miou = bev_sem_intersection / bev_sem_union
        bev_sem_miou[bev_sem_miou < 0.00001] = float('nan')

        data["road"] = bev_sem_miou[0].item()
        data["sidewalk"] = bev_sem_miou[1].item()
        data["building"] = bev_sem_miou[2].item()
        data["terrain"] = bev_sem_miou[3].item()
        data["person"] = bev_sem_miou[4].item()
        data["2-wheeler"] = bev_sem_miou[5].item()
        data["car"] = bev_sem_miou[6].item()
        data["truck"] = bev_sem_miou[7].item()

        #data["z_near"] = torch.tensor(self.z_near, device=images.device)
        #data["z_far"] = torch.tensor(self.z_far, device=images.device)

        globals()["IDX"] += 1

        return data


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
    test_dataset = make_test_dataset(config["data"])
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=config["num_workers"], shuffle=False, drop_last=False)

    return test_loader


def get_metrics(config, device):
    names = ["road", "sidewalk", "building", "terrain", "person", "2-wheeler", "car", "truck"]
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