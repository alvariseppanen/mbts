import math
import os

#import matplotlib.pyplot as plt
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

#import open3d as o3d

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

def getAngleBetweenVectors(vec1, vec2):
    vec1 = vec1.reshape(-1,2)
    vec2 = vec2.reshape(-1,2)
    dot_product = np.sum(vec1 * vec2, axis=1)
    cos_angle = dot_product / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
    angle = np.arccos(cos_angle)
    angle[np.isclose(cos_angle, 1.0)] = 0.0 # avoid nan at 1.0
    return angle

def getGroundTruth(groundTruthListFile, rootPath, poses, eval_every=1):
    '''if 'KITTI360_DATASET' in os.environ:
        rootPath = os.environ['KITTI360_DATASET']
    else:
        rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')'''

    '''if not os.path.isdir(rootPath):
        printError("Could not find a result root folder. Please read the instructions of this method.")

    if not os.path.isfile(groundTruthListFile):
        printError("Could not open %s. Please read the instructions of this method." % groundTruthListFile)'''

    with open(groundTruthListFile, 'r') as f:
        lines = f.read().splitlines()

    groundTruthFiles = []
    for i,line in enumerate(lines):
        if i % eval_every == 0:
            accumulatedPcdFile = os.path.join(rootPath, line.split(' ')[0])
            groundTruthFile = os.path.join(rootPath, line.split(' ')[1])
            if os.path.isfile(os.path.join(rootPath, groundTruthFile)):
                groundTruthFiles.append([groundTruthFile, accumulatedPcdFile])
            else:
                if not os.path.isfile(accumulatedPcdFile):
                    printError('Could not find the accumulated point cloud %s' % accumulatedPcdFile)
                if not generateCroppedGroundTruth(rootPath, accumulatedPcdFile, groundTruthFile):
                    printError("Could not open %s. Please read the instructions of this method." % groundTruthFile)
    return groundTruthFiles

# Crop the accumulate point cloud as the ground truth for semantic completion at a given frame
# The gruond truth is within a corridor of 30m around the vehicle poses of a 100m trajectory (50m in each direction).
# If the forward direction of one pose deviates more than 45 degree compared to the heading angle of the given center, 
# it is eliminated from the neighboring poses.
def generateCroppedGroundTruth(rootPath, accumulatedPcdFile, outputFile, disThres=50.0, angleThres=45.0):
    print("Creating %s from %s" % (outputFile, accumulatedPcdFile))
   
    # load the full accumulated window
    groundTruthPcd = read_ply(accumulatedPcdFile)
    groundTruthNpWindow = np.vstack((groundTruthPcd['x'], 
                                    groundTruthPcd['y'],
                                    groundTruthPcd['z'])).T
    groundTruthFullLabel = groundTruthPcd['semantic']
    groundTruthLabelWindow = np.zeros_like(groundTruthFullLabel)
    groundTruthColorWindow = np.zeros((groundTruthFullLabel.shape[0], 3))
    groundTruthConfWindow = groundTruthPcd['confidence']
    # Convert to trainId for evaluation, as some classes should be merged 
    # during evaluation, e.g., building+garage -> building
    for i in np.unique(groundTruthFullLabel):
        groundTruthLabelWindow[groundTruthFullLabel==i] = id2label[i].trainId
        groundTruthColorWindow[groundTruthFullLabel==i] = id2label[i].color
    groundTruthWindowTree = KDTree(groundTruthNpWindow, leaf_size=args.leafSize)

    # load the poses to determine the cropping region
    poseFile = os.path.join(rootPath, 'data_poses', '2013_05_28_drive_%04d_sync' % csWindow.sequenceNb, 'poses.txt')
    poses = np.loadtxt(poseFile)
    frameNb = int(os.path.splitext(os.path.basename(outputFile))[0])
    frameIdx = np.where(poses[:,0]==frameNb)[0]
    pose = poses[frameIdx]
    pose = np.reshape(pose[0,1:], (3,4)) 
    center = pose[:,3]

    # find neighbor points within the same window with the following conditions
    # 1) distance to the current frame within 40meters
    posesNeighbor = poses[np.logical_and(poses[:,0]>=csWindow.firstFrameNb, poses[:,0]<=csWindow.lastFrameNb)]
    dis_to_center = np.linalg.norm(posesNeighbor[:,1:].reshape(-1,3,4)[:,:,3] - center, axis=1)
    posesNeighbor = posesNeighbor[dis_to_center<disThres]
    # 2) curvature smaller than a given threshold (ignore potential occluded points due to large orientation)
    centerPrev = poses[frameIdx-1,1:].reshape(3,4)[:,3]
    centerNext = poses[frameIdx+1,1:].reshape(3,4)[:,3]
    posesPrevIdx = np.where(posesNeighbor[:,0]<=frameNb)[0]
    posesNextIdx = np.where(posesNeighbor[:,0]>=frameNb)[0]
    posesPrevLoc = posesNeighbor[posesPrevIdx,1:].reshape(-1,3,4)[:,:2,3]
    posesNextLoc = posesNeighbor[posesNextIdx,1:].reshape(-1,3,4)[:,:2,3]
    anglePrev = getAngleBetweenVectors(centerPrev[:2]-center[:2], posesPrevLoc[:-1]-posesPrevLoc[1:])
    angleNext = getAngleBetweenVectors(centerNext[:2]-center[:2], posesNextLoc[1:]-posesNextLoc[:-1])
    posesValid = np.concatenate((posesNeighbor[posesPrevIdx[:-1][anglePrev<angleThres]],
                                     posesNeighbor[posesNeighbor[:,0]==frameNb,:],
                                     posesNeighbor[posesNextIdx[1:][angleNext<angleThres]]), axis=0)

    idx_all = []
    for i in range(posesValid.shape[0]):
        idx = groundTruthWindowTree.query_radius(posesValid[i, 1:].reshape(3,4)[:3,3].reshape(1,3), args.radius)
        idx_all.append(idx[0])
    idx_all = np.unique(np.concatenate(idx_all))
    groundTruthNp = groundTruthNpWindow[idx_all,:]
    groundTruthColor = groundTruthColorWindow[idx_all,:]
    groundTruthLabel = groundTruthLabelWindow[idx_all]
    groundTruthConf = groundTruthConfWindow[idx_all]
    os.makedirs(os.path.dirname(outputFile), exist_ok=True)
    np.savez(outputFile, posesValid=posesValid, groundTruthNp=groundTruthNp, groundTruthColor=groundTruthColor, 
             groundTruthLabel=groundTruthLabel, groundTruthConf=groundTruthConf)

    return True

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


class BTSWrapper(nn.Module):
    def __init__(self, renderer, config, dataset) -> None:
        super().__init__()

        self.renderer = renderer

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.query_batch_size = config.get("query_batch_size", 50000)
        self.occ_threshold = 0.5

        self.x_range = (-25.6, 25.6) # (-26.19, 26.19) # (-28.57, 28.57)
        self.y_range = (3.2, -3.2) # (0, .75)
        self.z_range = (51.2, 0) # (57.14, 0) # (52.38, 0)
        self.ppm = 5
        self.ppm_y = 5

        self.y_res = 32

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
        #gt_bev = torch.stack(data["bev"], dim=1).squeeze()
        gt_vox = torch.stack(data["vox"], dim=1).squeeze()                  # 256, 256, 32
        gt_vox = torch.flip(gt_vox, dims=[0])

        vx, vy, vz = gt_vox.shape

        seq, id, is_right = self.dataset._datapoints[index]
        seq_len = self.dataset._img_ids[seq].shape[0]

        n, v, c, h, w = images.shape
        device = images.device

        if torch.count_nonzero(gt_vox != 255) == 0: 
            data["road"] = float('nan')
            data["sidewalk"] = float('nan')
            data["building"] = float('nan')
            data["terrain"] = float('nan')
            data["person"] = float('nan')
            data["2-wheeler"] = float('nan')
            data["car"] = float('nan')
            data["truck"] = float('nan')
            #print("skipping: ", id)
            #print(" ")
            #globals()["IDX"] += 1
            return data
        
        #print("evaluating: ", id)
        #print(" ")

        #gt_vox[gt_vox == 0] = 255

        #gt_vox[gt_vox == 0] = 255

        '''grid = torch.from_numpy(np.indices((gt_vox.shape))).cuda()
        x_i = grid[0][...,None] # x indices
        y_i = grid[1][...,None] # y indices
        z_i = grid[2][...,None] # z indices
        voxel_coordinates = torch.cat((x_i, y_i, z_i), dim=3)'''

        # crop area of interest
        crop_x = 90
        crop_z = 160
        gt_vox[0:crop_z, :, :] = 255
        gt_vox[:, 0:crop_x, :] = 255
        gt_vox[:, gt_vox.shape[1]-crop_x:gt_vox.shape[1], :] = 255

        #occupied_mask = torch.zeros_like(gt_vox).float().cuda()
        #occupied_mask[gt_vox != 255] = 1
        #occupied_mask[gt_vox == 6] = 1

        #gt_vox = gt_vox[occupied_mask] # N, 1
        #voxel_coordinates = voxel_coordinates[occupied_mask, :] # N, 3
        #print(voxel_coordinates.shape, gt_vox.shape)
        
        # down-sample for vis
        avgp = torch.nn.AvgPool3d(2)
        #occupied_mask = avgp(occupied_mask.unsqueeze(0)).squeeze()
        #occupied_mask = avgp(occupied_mask.unsqueeze(0)).squeeze()

        #print(occupied_mask.shape)

        #x, y, z = np.indices((8, 64, 64))
        #cube1 = (x < 3) & (y < 3) & (z < 3)
        #cube2 = (x >= 5) & (y >= 5) & (z >= 5)
        #link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

        # combine the objects into a single boolean array
        #voxelarray = cube1 | cube2 | link

        # set the colors of each object
        #colors = np.empty(voxelarray.shape, dtype=object)
        #colors[link] = 'red'
        #colors[cube1] = 'blue'
        #colors[cube2] = 'green'

        '''pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_coordinates[:,:3].cpu().numpy())
        #pcd.colors = o3d.utility.Vector3dVector(voxel_grid[:,3:6].numpy())
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.00)
        o3d.visualization.draw_geometries([voxel_grid])'''
        
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

        #T_velo_to_pose = torch.tensor(self.dataset._calibs["T_velo_to_pose"], device=device)

        # load gt
        #groundTruthListFile = os.path.join(self.dataset.data_path, 'data_3d_semantics', 'train', '2013_05_28_drive_val_frames.txt')
        #groundTruthImgList = getGroundTruth(groundTruthListFile, self.dataset.data_path, poses)

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

        sigmas[torch.any(invalid, dim=-1)] = 0
        pred_class[torch.any(invalid, dim=-1)] = -1
        
        occupied_mask = sigmas > self.occ_threshold

        pred_class = pred_class.reshape(y_res, z_res, x_res).permute(1,2,0)
        occupied_mask = occupied_mask.reshape(y_res, z_res, x_res).permute(1,2,0)

        '''occupied_mask[gt_vox == 255] = 0
        occupied_mask = avgp(occupied_mask.float().unsqueeze(0)).squeeze()
        occupied_mask = avgp(occupied_mask.unsqueeze(0)).squeeze()
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(occupied_mask.bool().cpu().numpy())
        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        plt.show()'''

        # remap predictions
        '''new_pred_bev = torch.ones_like(pred_bev)*255
        new_pred_bev[pred_bev == 0] = 0
        new_pred_bev[pred_bev == 1] = 1
        new_pred_bev[pred_bev == 2] = 2
        new_pred_bev[pred_bev == 5] = 3
        new_pred_bev[pred_bev == 6] = 4
        new_pred_bev[pred_bev == 9] = 5
        new_pred_bev[pred_bev == 7] = 6
        new_pred_bev[pred_bev == 8] = 7
        new_pred_bev[pred_bev == -1] = 255'''

        gt_vox[pred_class == -1] = 255
        #pred_class[gt_vox == 255] = 255
        
        # down-sample for vis
        '''gt_occupied_mask = torch.zeros_like(gt_vox).float().cuda()
        gt_occupied_mask[gt_vox != 255] = 1
        gt_occupied_mask = avgp(gt_occupied_mask.unsqueeze(0)).squeeze()
        gt_occupied_mask = avgp(gt_occupied_mask.unsqueeze(0)).squeeze()
        ax = plt.figure().add_subplot(projection='3d')
        #ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
        ax.voxels(gt_occupied_mask.bool().cpu().numpy())
        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        plt.show()'''

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

        bev_semantic_conf_mat = self._confusion_matrix(pred_class.reshape(vx*vy, vz).squeeze().cpu(), gt_vox.reshape(vx*vy, vz).squeeze().cpu())
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
        #print(bev_sem_miou)
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