import math
from copy import copy

import ignite.distributed as idist
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import profiler
from torchvision.utils import make_grid
import skimage.metrics

import lpips

from datasets.data_util import make_datasets, make_test_dataset
from datasets.kitti_odom.kitti_odometry_dataset import KittiOdometryDataset
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset
from models.common.model.scheduler import make_scheduler
from models.common.render import NeRFRenderer, semNeRFRenderer
from models.bts.model.image_processor import make_image_processor, RGBProcessor
from models.bts.model.loss import ReconstructionLoss, compute_errors_l1ssim
from models.bts.model.sem_loss import SemReconstructionLoss, compute_errors_l1ssim
from models.bts.model.models_bts import BTSNet
from models.bts.model.models_sbts import SBTSNet
from models.bts.model.ray_sampler import ImageRaySampler, PatchRaySampler, RandomRaySampler
from utils.base_trainer import base_training
from utils.metrics import MeanMetric
from utils.plotting import color_tensor
from utils.projection_operations import distance_to_z

from models.bts.evaluator_lidar import get_pts, get_lidar_slices, check_occupancy, project_into_cam
import time
import numpy as np
import os
import cv2

class BTSWrapper(nn.Module):
    def __init__(self, renderer, config, eval_nvs=False) -> None:
        super().__init__()

        self.renderer = renderer

        # added
        _, self.val_dataset = make_datasets(config["data"])
        #self.val_dataset = make_test_dataset(config["data"])
        config = config["model_conf"]
        self.enc_type = config["encoder"]["type"]
        self.project_scale = config["encoder"].get("project_scale", 2)
        self.dataset_type = "kitti_360"
        if self.enc_type == "volumetric":
            self.dataset_type = config["encoder"]["dataset_type"]
        if self.dataset_type == "kitti_360":
            # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that.
            self.cam_incl_adjust = torch.tensor(
                [  [1.0000000,  0.0000000,  0.0000000, 0],
                [0.0000000,  0.9961947,  0.0871557, 0],
                [0.0000000, -0.0871557,  0.9961947, 0],
                [0.0000000,  000000000,  0.0000000, 1]
                ],
                dtype=torch.float32
            ).view(1, 1, 4, 4)
        self.x_range = (-4, 4) # (-9, 9) # (-4, 4)
        self.y_range = (0, .75) # (0, 1.6) # (0, .75)
        self.z_range = (20, 4) # (25.6, 3) # (20, 4)
        self.ppm = 10
        self.ppm_y = 4
        self.y_res = 1
        self.aggregate_timesteps = 20
        self.occ_threshold = 0.5 # 0.5
        self.query_batch_size = 50000

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.ray_batch_size = config["ray_batch_size"]
        frames_render = config.get("n_frames_render", 2)
        self.frame_sample_mode = config.get("frame_sample_mode", "default")
        self.loss_from_single_img = config.get("loss_from_single_img", False)

        self.sample_mode = config.get("sample_mode", "random")
        self.patch_size = config.get("patch_size", 16)
        self.use_scales = config.get("use_scales", False)
        self.use_automasking = config.get("use_automasking", False)

        self.prediction_mode = config.get("prediction_mode", "multiscale")

        self.alternating_ratio = config.get("alternating_ratio", None)

        cfg_ip = config.get("image_processor", {})
        self.train_image_processor = make_image_processor(cfg_ip)
        self.val_image_processor = RGBProcessor()

        if type(frames_render) == int:
            self.frames_render = list(range(frames_render))
        else:
            self.frames_render = frames_render
        self.frames = self.frames_render
        
        if self.sample_mode == "random":
            self.train_sampler = RandomRaySampler(self.ray_batch_size, self.z_near, self.z_far, channels=self.train_image_processor.channels)
        elif self.sample_mode == "patch":
            self.train_sampler = PatchRaySampler(self.ray_batch_size, self.z_near, self.z_far, self.patch_size, channels=self.train_image_processor.channels)
        elif self.sample_mode == "image":
            self.train_sampler = ImageRaySampler(self.z_near, self.z_far, channels=self.train_image_processor.channels)
        else:
            raise NotImplementedError

        if self.use_automasking:
            self.train_sampler.channels += 1

        self.val_sampler = ImageRaySampler(self.z_near, self.z_far) 

        self.eval_nvs = eval_nvs
        if self.eval_nvs:
            self.lpips = lpips.LPIPS(net="alex")

        self._counter = 0

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def forward(self, data):
        #self.training = False
        #st = time.time()
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)                           # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)                           # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)                           # n, v, 4, 4 (-1, 1)

        # added 
        projected_pix = data["projected_pix_{}".format(self.project_scale)] # n, h*w, 2    (only for single input image)
        fov_mask = data["fov_mask_{}".format(self.project_scale)]           # n, h*w       (only for single input image)
        
        #if self.training == False: segs = None
        if self.enc_type != "semantic": segs = None 
        else: segs = torch.stack(data["segs"], dim=1)

        '''label_id = 7
        img_id = 7
        f_seg = segs[0,img_id,...].clone()
        f_seg[f_seg == 255] = 0
        f_seg[f_seg == label_id] = 255
        cv2.imshow('seg', f_seg.cpu().numpy().astype(np.uint8))
        cv2.imshow('img', images[0,img_id,...].permute(1,2,0).cpu().numpy())
        cv2.waitKey(2000)'''

        n, v, c, h, w = images.shape
        device = images.device

        # Use first frame as keyframe
        '''if self.enc_type == "volumetric" and self.dataset_type == "kitti_360":
            # Our coordinate system is at the same position as cam0, but rotated 5deg up along the x axis to adjust for camera inclination.
            # Consequently, the xz plane is parallel to the street. This makes the depth valuation metrics worse.
            world_transform = torch.inverse(poses[:, :1, :, :])
            world_transform = self.cam_incl_adjust.to(device) @ world_transform
            poses = world_transform.expand(-1, v, -1, -1) @ poses
        else:'''
        to_base_pose = torch.inverse(poses[:, :1, :, :])
        poses = to_base_pose.expand(-1, v, -1, -1) @ poses

        if self.training and self.alternating_ratio is not None:
            step = self._counter % (self.alternating_ratio + 1)
            if step < self.alternating_ratio:
                for params in self.renderer.net.encoder.parameters(True):
                    params.requires_grad_(True)
                for params in self.renderer.net.mlp_coarse.parameters(True):
                    params.requires_grad_(False)
            else:
                for params in self.renderer.net.encoder.parameters(True):
                    params.requires_grad_(False)
                for params in self.renderer.net.mlp_coarse.parameters(True):
                    params.requires_grad_(True)

        if self.training:
            frame_perm = torch.randperm(v)
        else:
            frame_perm = torch.arange(v)

        ids_encoder = [0]
        ids_render = torch.sort(frame_perm[[i for i in self.frames_render if i < v]]).values

        combine_ids = None

        if self.training:
            if self.frame_sample_mode == "only":
                ids_loss = [0]
                ids_render = ids_render[ids_render != 0]
            elif self.frame_sample_mode == "not":
                frame_perm = torch.randperm(v-1) + 1
                ids_loss = torch.sort(frame_perm[[i for i in self.frames_render if i < v-1]]).values
                ids_render = [i for i in range(v) if i not in ids_loss]
            elif self.frame_sample_mode == "stereo":
                if frame_perm[0] < v // 2:
                    ids_loss = list(range(v // 2))
                    ids_render = list(range(v // 2, v))
                else:
                    ids_loss = list(range(v // 2, v))
                    ids_render = list(range(v // 2))
            elif self.frame_sample_mode == "mono":
                split_i = v // 2
                if frame_perm[0] < v // 2:
                    ids_loss = list(range(0, split_i, 2)) + list(range(split_i+1, v, 2))
                    ids_render = list(range(1, split_i, 2)) + list(range(split_i, v, 2))
                else:
                    ids_loss = list(range(1, split_i, 2)) + list(range(split_i, v, 2))
                    ids_render = list(range(0, split_i, 2)) + list(range(split_i + 1, v, 2))
            elif self.frame_sample_mode == "kitti360-mono":
                steps = v // 4
                start_from = 0 if frame_perm[0] < v // 2 else 1

                ids_loss = []
                ids_render = []

                for cam in range(4):
                    ids_loss += [cam * steps + i for i in range(start_from, steps, 2)]
                    ids_render += [cam * steps + i for i in range(1 - start_from, steps, 2)]
                    start_from = 1 - start_from
            elif self.frame_sample_mode.startswith("waymo"):
                num_views = int(self.frame_sample_mode.split("-")[-1])
                steps = v // num_views
                split = steps // 2

                # Predict features from half-left, center, half-right
                ids_encoder = [0, steps, steps * 2]

                # Combine all frames half-left, center, half-right for efficiency reasons
                combine_ids = [(i, steps + i, steps * 2 + i) for i in range(steps)]

                if self.training:
                    step_perm = torch.randperm(steps)
                else:
                    step_perm = torch.arange(steps)
                step_perm = step_perm.tolist()

                ids_loss = sum([[i + j * steps for j in range(num_views)] for i in step_perm[:split]], [])
                ids_render = sum([[i + j * steps for j in range(num_views)] for i in step_perm[split:]], [])

            elif self.frame_sample_mode == "default":
                ids_loss = frame_perm[[i for i in range(v) if frame_perm[i] not in ids_render]]
            else:
                raise NotImplementedError
        else:
            ids_loss = torch.arange(v)
            ids_render = [0]

            if self.frame_sample_mode.startswith("waymo"):
                num_views = int(self.frame_sample_mode.split("-")[-1])
                steps = v // num_views
                split = steps // 2
                # Predict features from half-left, center, half-right
                ids_encoder = [0, steps, steps * 2]
                ids_render = [0, steps, steps * 2]
                combine_ids = [(i, steps + i, steps * 2 + i) for i in range(steps)]

        if self.loss_from_single_img:
            ids_loss = ids_loss[:1]

        ip = self.train_image_processor if self.training else self.val_image_processor

        images_ip = ip(images)
        if self.training and self.use_automasking:
            with profiler.record_function("trainer_automasking"):
                reference_imgs = images_ip.permute(0, 1, 3, 4, 2).view(n, v, h, w, 1, c).expand(-1, -1, -1, -1, len(ids_render), -1) * .5
                render_imgs = images_ip[:, ids_loss].permute(0, 3, 4, 1, 2).view(n, 1, h, w, len(ids_render), c).expand(-1, v, -1, -1, -1, -1) * .5
                errors = compute_errors_l1ssim(reference_imgs, render_imgs).mean(-2).squeeze(-1).unsqueeze(2)
                images_ip = torch.cat((images_ip, errors), dim=2)
        
        with profiler.record_function("trainer_encode-grid"):
            if self.enc_type == "volumetric":
                self.renderer.net.compute_grid_transforms(projs[:, ids_encoder], poses[:, ids_encoder])
                self.renderer.net.volume_encode(images, projected_pix, fov_mask, projs, poses, ids_encoder=ids_encoder, ids_render=ids_render, images_alt=images_ip, combine_ids=combine_ids)
            else:
                self.renderer.net.compute_grid_transforms(projs[:, ids_encoder], poses[:, ids_encoder])
                self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_render, images_alt=images_ip, combine_ids=combine_ids)
        
        sampler = self.train_sampler if self.training else self.val_sampler
        #st = time.time()

        # add semantic labels
        if segs is not None:
            images_ip = torch.cat((images_ip, segs.unsqueeze(2)), dim=2)
        
        with profiler.record_function("trainer_sample-rays"):
            all_rays, all_rgb_gt = sampler.sample(images_ip[:, ids_loss] , poses[:, ids_loss], projs[:, ids_loss])
        #print("ray sampling: ", time.time() - st)
        data["fine"] = []
        data["coarse"] = []
        
        if self.prediction_mode == "multiscale": # false
            for scale in self.renderer.net.encoder.scales:
                self.renderer.net.set_scale(scale)

                using_fine = self.renderer.renderer.using_fine
                if scale != 0 and using_fine:
                    self.renderer.renderer.using_fine = False
                render_dict = self.renderer(all_rays, want_weights=True, want_alphas=True, want_rgb_samps=True)
                if scale != 0 and using_fine:
                    self.renderer.renderer.using_fine = True

                if "fine" not in render_dict:
                    render_dict["fine"] = dict(render_dict["coarse"])

                render_dict["rgb_gt"] = all_rgb_gt
                render_dict["rays"] = all_rays

                render_dict = sampler.reconstruct(render_dict)

                data["fine"].append(render_dict["fine"])
                data["coarse"].append(render_dict["coarse"])
                data["rgb_gt"] = render_dict["rgb_gt"]
                data["rays"] = render_dict["rays"]
        else:
            with profiler.record_function("trainer_render"):
                render_dict = self.renderer(all_rays, want_weights=True, want_alphas=True, want_rgb_samps=True)

            if "fine" not in render_dict:
                render_dict["fine"] = dict(render_dict["coarse"])

            render_dict["rgb_gt"] = all_rgb_gt
            render_dict["rays"] = all_rays

            with profiler.record_function("trainer_reconstruct"):
                render_dict = sampler.reconstruct(render_dict)

            data["fine"].append(render_dict["fine"])
            data["coarse"].append(render_dict["coarse"])
            data["rgb_gt"] = render_dict["rgb_gt"]
            data["rays"] = render_dict["rays"]

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        if self.training is False:
            data["coarse"][0]["depth"] = distance_to_z(data["coarse"][0]["depth"], projs)
            data["fine"][0]["depth"] = distance_to_z(data["fine"][0]["depth"], projs)

            if len(data["depths"]) > 0:
                data.update(self.compute_depth_metrics(data))
            if self.eval_nvs: # False
                data.update(self.compute_nvs_metrics(data))
            #if True:
            #    data.update(self.compute_occ_metrics(data, sampler))

        if self.training:
            self._counter += 1
        #print("full: ", time.time() - st)
        return data
    
    def compute_occ_metrics(self, data, sampler):
        # added
        #self.val_dataset = make_test_dataset(self.config["data"])

        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)                           # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)                 # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)                           # n, v, 4, 4 (-1, 1)
        index = data["index"].item()
        #print("occ: ", index)

        # added 
        projected_pix = data["projected_pix_{}".format(self.project_scale)] # n, h*w, 2    (only for single input image)
        fov_mask = data["fov_mask_{}".format(self.project_scale)]           # n, h*w       (only for single input image)

        seq, id, is_right = self.val_dataset._datapoints[index] # self.dataset undefined
        seq_len = self.val_dataset._img_ids[seq].shape[0]

        n, v, c, h, w = images.shape
        #print(images.shape) # 1, 8, 3, h, w
        device = images.device

        T_velo_to_pose = torch.tensor(self.val_dataset._calibs["T_velo_to_pose"], device=device)

        # Our coordinate system is at the same position as cam0, but rotated 5deg up along the x axis to adjust for camera inclination.
        # Consequently, the xz plane is parallel to the street.
        world_transform = torch.inverse(poses[:, :1, :, :])
        world_transform = self.cam_incl_adjust.to(device) @ world_transform
        poses = world_transform @ poses

        sampler.height = h
        sampler.width = w

        # Load lidar pointclouds
        points_all = []
        velo_poses = []
        for id in range(id, min(id + self.aggregate_timesteps, seq_len)):
            points = np.fromfile(os.path.join(self.val_dataset.data_path, "data_3d_raw", seq, "velodyne_points", "data", f"{self.val_dataset._img_ids[seq][id]:010d}.bin"), dtype=np.float32).reshape(-1, 4)
            points[:, 3] = 1.0
            points = torch.tensor(points, device=device)
            velo_pose = world_transform.squeeze() @ torch.tensor(self.val_dataset._poses[seq][id], device=device) @ T_velo_to_pose
            points_all.append(points)
            velo_poses.append(velo_pose)

        velo_poses = torch.stack(velo_poses, dim=0)

        rays, _ = sampler.sample(None, poses[:, :1, :, :], projs[:, :1, :, :])

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
        render_dict = sampler.reconstruct(render_dict)
        pred_depth = distance_to_z(render_dict["coarse"]["depth"], projs[:1, :1])

        # Get pts
        q_pts, (xd, yd, zd) = get_pts(self.x_range, self.y_range, self.z_range, self.ppm, self.ppm_y, self.y_res)
        q_pts = q_pts.to(images.device).view(-1, 3)

        # is visible? Check whether point is closer than the computed pseudo depth
        cam_pts, dists = project_into_cam(q_pts, projs[0, 0], poses[0, 0])
        pred_dist = F.grid_sample(pred_depth.view(1, 1, h, w), cam_pts[:, :2].view(1, 1, -1, 2), mode="nearest", padding_mode="border", align_corners=True).view(-1)
        is_visible_pred = dists <= pred_dist

        # Query the density of the query points from the density field
        densities = []
        for i_from in range(0, len(q_pts), self.query_batch_size):
            i_to = min(i_from + self.query_batch_size, len(q_pts))
            q_pts_ = q_pts[i_from:i_to]
            _, _, densities_ = self.renderer.net(q_pts_.unsqueeze(0), only_density=True)
            densities.append(densities_.squeeze(0))
        densities = torch.cat(densities, dim=0).squeeze()
        #print(densities.shape, densities.max(), densities.min())
        is_occupied_pred = densities > self.occ_threshold

        # is occupied?
        slices = get_lidar_slices(points_all, velo_poses, self.y_range, yd, (self.z_range[0] ** 2 + self.x_range[0] ** 2) ** .5)
        is_occupied, is_visible = check_occupancy(q_pts, slices, velo_poses)

        is_visible |= is_visible_pred

        # Only not visible points can be occupied
        is_occupied &= ~is_visible

        is_occupied_acc = (is_occupied_pred == is_occupied).float().mean()
        is_occupied_prec = is_occupied[is_occupied_pred].float().mean()
        is_occupied_rec = is_occupied_pred[is_occupied].float().mean()

        not_occupied_not_visible_ratio = ((~is_occupied) & (~is_visible)).float().mean()

        total_ie = ((~is_occupied) & (~is_visible)).float().sum()

        ie_acc = (is_occupied_pred == is_occupied)[(~is_visible)].float().mean()
        ie_prec = (~is_occupied)[(~is_occupied_pred) & (~is_visible)].float().mean()
        ie_rec = (~is_occupied_pred)[(~is_occupied) & (~is_visible)].float().mean()
        total_no_nop_nv = ((~is_occupied) & (~is_occupied_pred))[(~is_visible) & (~is_occupied)].float().sum()

        '''
        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)'''

        #globals()["IDX"] += 1

        metrics_dict = {
            "o_acc": torch.nan_to_num(is_occupied_acc.view(1), nan=0.0),
            "o_rec": torch.nan_to_num(is_occupied_rec.view(1), nan=0.0),
            "o_prec": torch.nan_to_num(is_occupied_prec.view(1), nan=0.0),
            "ie_acc": torch.nan_to_num(ie_acc.view(1), nan=0.0),
            "ie_rec": torch.nan_to_num(ie_rec.view(1), nan=0.0),
            "ie_prec": torch.nan_to_num(ie_prec.view(1), nan=0.0),
            "ie_r": torch.nan_to_num(not_occupied_not_visible_ratio.view(1), nan=0.0),
            "t_ie": torch.nan_to_num(total_ie.view(1), nan=0.0),
            "t_no_nop_nv": torch.nan_to_num(total_no_nop_nv.view(1), nan=0.0)
        }
        #print(metrics_dict)
        return metrics_dict

    def compute_depth_metrics(self, data):
        # TODO: This is only correct for batchsize 1!
        depth_gt = data["depths"][0]
        depth_pred = data["fine"][0]["depth"][:, :1]

        depth_pred = F.interpolate(depth_pred, depth_gt.shape[-2:])
        # TODO: Maybe implement median scaling

        depth_pred = torch.clamp(depth_pred, 1e-3, 80)
        mask = depth_gt != 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
        a1 = (thresh < 1.25).to(torch.float).mean()
        a2 = (thresh < 1.25 ** 2).to(torch.float).mean()
        a3 = (thresh < 1.25 ** 3).to(torch.float).mean()

        rmse = (depth_gt - depth_pred) ** 2
        rmse = rmse.mean() ** .5

        rmse_log = (torch.log(depth_gt) - torch.log(depth_pred)) ** 2
        rmse_log = rmse_log.mean() ** .5

        abs_rel = torch.mean(torch.abs(depth_gt - depth_pred) / depth_gt)

        sq_rel = torch.mean(((depth_gt - depth_pred) ** 2) / depth_gt)

        #print(abs_rel.view(1).shape, sq_rel.view(1).shape, rmse.view(1).shape, rmse_log.view(1).shape, a1.view(1).shape, a2.view(1).shape, a3.view(1).shape)
        metrics_dict = {
            "abs_rel": abs_rel.view(1),
            "sq_rel": sq_rel.view(1),
            "rmse": rmse.view(1),
            "rmse_log": rmse_log.view(1),
            "a1": a1.view(1),
            "a2": a2.view(1),
            "a3": a3.view(1)
        }
        return metrics_dict

    def compute_nvs_metrics(self, data):
        # TODO: This is only correct for batchsize 1!
        # Following tucker et al. and others, we crop 5% on all sides

        # idx of stereo frame (the target frame is always the "stereo" frame).
        sf_id = data["rgb_gt"].shape[1] // 2

        imgs_gt = data["rgb_gt"][:1, sf_id:sf_id+1]
        imgs_pred = data["fine"][0]["rgb"][:1, sf_id:sf_id+1]

        imgs_gt = imgs_gt.squeeze(0).permute(0, 3, 1, 2)
        imgs_pred = imgs_pred.squeeze(0).squeeze(-2).permute(0, 3, 1, 2)

        n, c, h, w = imgs_gt.shape
        y0 = int(math.ceil(0.05 * h))
        y1 = int(math.floor(0.95 * h))
        x0 = int(math.ceil(0.05 * w))
        x1 = int(math.floor(0.95 * w))

        imgs_gt = imgs_gt[:, :, y0:y1, x0:x1]
        imgs_pred = imgs_pred[:, :, y0:y1, x0:x1]

        imgs_gt_np = imgs_gt.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        imgs_pred_np = imgs_pred.detach().squeeze().permute(1, 2, 0).cpu().numpy()

        ssim_score = skimage.metrics.structural_similarity(imgs_pred_np, imgs_gt_np, multichannel=True, data_range=1)
        psnr_score = skimage.metrics.peak_signal_noise_ratio(imgs_pred_np, imgs_gt_np, data_range=1)
        lpips_score = self.lpips(imgs_pred, imgs_gt, normalize=False).mean()

        metrics_dict = {
            "ssim": torch.tensor([ssim_score], device=imgs_gt.device),
            "psnr": torch.tensor([psnr_score], device=imgs_gt.device),
            "lpips": torch.tensor([lpips_score], device=imgs_gt.device)
        }
        return metrics_dict


def training(local_rank, config):
    return base_training(local_rank, config, get_dataflow, initialize, get_metrics, visualize)


def get_dataflow(config, logger=None):
    # - Get train/test datasets
    if idist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the dataset
        idist.barrier()

    mode = config.get("mode", "depth")

    train_dataset, test_dataset = make_datasets(config["data"])
    vis_dataset = copy(test_dataset)

    # Change eval dataset to only use a single prediction and to return gt depth.
    test_dataset.frame_count = 1 if isinstance(train_dataset, KittiRawDataset) or isinstance(train_dataset, KittiOdometryDataset) else 2
    test_dataset._left_offset = 0
    test_dataset.return_stereo = mode == "nvs"
    test_dataset.return_depth = True
    test_dataset.length = min(256, test_dataset.length)

    # Change visualisation dataset
    vis_dataset.length = 1
    vis_dataset._skip = 12 if isinstance(train_dataset, KittiRawDataset) or isinstance(train_dataset, KittiOdometryDataset) else 50
    vis_dataset.return_depth = True

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True, drop_last=True)
    test_loader = idist.auto_dataloader(test_dataset, batch_size=1, num_workers=config["num_workers"], shuffle=False)
    vis_loader = idist.auto_dataloader(vis_dataset, batch_size=1, num_workers=config["num_workers"], shuffle=False)

    return train_loader, test_loader, vis_loader


def get_metrics(config, device):
    names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    #names = ["o_acc", "o_rec", "o_prec", "ie_acc", "ie_rec", "ie_prec", "ie_r", "t_ie", "t_no_nop_nv", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    #names = ["o_acc", "o_rec", "o_prec", "ie_acc", "ie_rec", "ie_prec", "ie_r", "t_ie", "t_no_nop_nv"]
             
    if config.get("mode", "depth") == "nvs":
        names += ["ssim", "psnr", "lpips"]

    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "BTSNet")
    net = globals()[arch](config["model_conf"])
    if arch == "SBTSNet":
        renderer = semNeRFRenderer.from_conf(config["renderer"])
    else:
        renderer = NeRFRenderer.from_conf(config["renderer"])

    renderer = renderer.bind_parallel(net, gpus=None).eval()

    mode = config.get("mode", "depth")

    model = BTSWrapper(
        renderer,
        config,
        mode == "nvs"
    )

    model = idist.auto_model(model)
    #model = idist.auto_model(model, find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = make_scheduler(config.get("scheduler", {}), optimizer)

    if arch == "SBTSNet":
        criterion = SemReconstructionLoss(config["loss"], config["model_conf"].get("use_automasking", False))
    else:
        criterion = ReconstructionLoss(config["loss"], config["model_conf"].get("use_automasking", False))

    return model, optimizer, criterion, lr_scheduler


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    print("Visualizing")

    data = engine.state.output["output"]
    writer = logger.writer

    images = torch.stack(data["imgs"], dim=1).detach()[0]
    recon_imgs = data["fine"][0]["rgb"].detach()[0]
    recon_depths = [f["depth"].detach()[0] for f in data["fine"]]

    # depth_profile = data["coarse"][0]["weights"].detach()[0]
    depth_profile = data["coarse"][0]["alphas"].detach()[0]
    alphas = data["coarse"][0]["alphas"].detach()[0]
    invalids = data["coarse"][0]["invalid"].detach()[0]

    z_near = data["z_near"]
    z_far = data["z_far"]

    take_n = min(images.shape[0], 6)

    _, c, h, w = images.shape
    nv = recon_imgs.shape[0]

    images = images[:take_n]
    images = images * .5 + .5

    recon_imgs = recon_imgs.view(nv, h, w, -1, c)
    recon_imgs = recon_imgs[:take_n]
    # Aggregate recon_imgs by taking the mean
    recon_imgs = recon_imgs.mean(dim=-2).permute(0, 3, 1, 2)

    recon_mse = (((images - recon_imgs) ** 2) / 2).mean(dim=1).clamp(0, 1)
    recon_mse = color_tensor(recon_mse, cmap="plasma").permute(0, 3, 1, 2)

    recon_depths = [(1 / d[:take_n] - 1 / z_far) / (1 / z_near - 1 / z_far) for d in recon_depths]
    recon_depths = [color_tensor(d.squeeze(1).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2) for d in recon_depths]

    depth_profile = depth_profile[:take_n][:, [h//4, h//2, 3*h//4], :, :].view(take_n*3, w, -1).permute(0, 2, 1)
    depth_profile = depth_profile.clamp_min(0) / depth_profile.max()
    depth_profile = color_tensor(depth_profile, cmap="plasma").permute(0, 3, 1, 2)

    alphas = alphas[:take_n]

    alphas += 1e-5

    ray_density = alphas / alphas.sum(dim=-1, keepdim=True)
    ray_entropy = -(ray_density * torch.log(ray_density)).sum(-1) / (math.log2(alphas.shape[-1]))
    ray_entropy = color_tensor(ray_entropy, cmap="plasma").permute(0, 3, 1, 2)

    alpha_sum = (alphas.sum(dim=-1) / alphas.shape[-1]).clamp(-1)
    alpha_sum = color_tensor(alpha_sum, cmap="plasma").permute(0, 3, 1, 2)

    invalids = invalids[:take_n]
    invalids = invalids.mean(-2).mean(-1)
    invalids = color_tensor(invalids, cmap="plasma").permute(0, 3, 1, 2)

    # Write images
    nrow = int(take_n ** .5)

    images_grid = make_grid(images, nrow=nrow)
    recon_imgs_grid = make_grid(recon_imgs, nrow=nrow)
    recon_depths_grid = [make_grid(d, nrow=nrow) for d in recon_depths]
    depth_profile_grid = make_grid(depth_profile, nrow=nrow)
    ray_entropy_grid = make_grid(ray_entropy, nrow=nrow)
    alpha_sum_grid = make_grid(alpha_sum, nrow=nrow)
    recon_mse_grid = make_grid(recon_mse, nrow=nrow)
    invalids_grid = make_grid(invalids, nrow=nrow)

    writer.add_image(f"{tag}/input_im", images_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/recon_im", recon_imgs_grid.cpu(), global_step=step)
    for i, d in enumerate(recon_depths_grid):
        writer.add_image(f"{tag}/recon_depth_{i}", d.cpu(), global_step=step)
    writer.add_image(f"{tag}/depth_profile", depth_profile_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/ray_entropy", ray_entropy_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/alpha_sum", alpha_sum_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/recon_mse", recon_mse_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/invalids", invalids_grid.cpu(), global_step=step)
