import copy
import time
from collections import OrderedDict
import kornia
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from po_bev_unsupervised.algos.voxel_grid import VoxelGridAlgo
from po_bev_unsupervised.utils.sequence import pad_packed_images
from po_bev_unsupervised.algos.fv_depth import get_dynamic_weight_map, get_depth_weight_map

TRANSFORMER_ORDER = ['front', 'left', 'right', 'rear']
NETWORK_INPUTS = ["img", "fv_msk", "fv_cat", "fv_iscrowd", "bev_msk", "bev_cat", "bev_iscrowd", "fv_intrinsics", "ego_pose"]


class UnsupervisedDepthNet(nn.Module):
    def __init__(self,
                 body,
                 body_depth,
                 voxel_grid_depth,
                 fv_sem_head,
                 fv_depth_head,
                 bev_sem_head,
                 rgb_reconstruction_head,
                 voxel_grid_algo,
                 fv_sem_algo,
                 fv_depth_algo,
                 bev_sem_algo,
                 rgb_reconstruction_algo,
                 bev_rgb_reconstruction_algo,
                 dataset,
                 bev_to_bev_warper,
                 fv_to_bev_warper,
                 pointcloud_to_fv_warper=None,
                 fv_depth_to_bev_warper=None,
                 fv_pointcloud_to_bev=None,
                 fv_classes=None,
                 bev_classes=None,
                 front_vertical_classes=None,  # In the frontal view
                 front_flat_classes=None,  # In the frontal view
                 front_dynamic_classes=None,  # In the frontal view
                 bev_vertical_classes=None,  # In the BEV
                 bev_flat_classes=None,  # In the BEV
                 fv_ignore_classes=None,
                 bev_ignore_classes=None,
                 fv_sky_index=None,
                 fv_veg_index=None):
        super(UnsupervisedDepthNet, self).__init__()

        # Backbone
        self.body = body
        self.body_depth = body_depth

        # Transformer
        self.voxel_grid_depth = voxel_grid_depth

        # Modules
        self.fv_sem_head = fv_sem_head
        self.fv_depth_head = fv_depth_head
        self.bev_sem_head = bev_sem_head
        self.rgb_reconstruction_head = rgb_reconstruction_head

        # Algorithms
        self.voxel_grid_algo = voxel_grid_algo
        self.fv_sem_algo = fv_sem_algo
        self.fv_depth_algo = fv_depth_algo
        self.bev_sem_algo = bev_sem_algo
        self.rgb_reconstruction_algo = rgb_reconstruction_algo
        self.bev_rgb_reconstruction_algo = bev_rgb_reconstruction_algo
        self.fv_to_bev_warper = fv_to_bev_warper
        self.fv_depth_to_bev_warper = fv_depth_to_bev_warper
        self.fv_pointcloud_to_bev = fv_pointcloud_to_bev
        self.pointcloud_to_fv_warper = pointcloud_to_fv_warper
        self.bev_to_bev_warper = bev_to_bev_warper

        # Params
        self.dataset = dataset
        self.fv_num_classes = fv_classes['total']
        self.fv_num_stuff = fv_classes["stuff"]
        self.bev_num_classes = bev_classes['total']
        self.bev_num_stuff = bev_classes['stuff']
        self.front_vertical_classes = front_vertical_classes
        self.front_flat_classes = front_flat_classes
        self.front_dynamic_classes = front_dynamic_classes
        self.bev_vertical_classes = bev_vertical_classes
        self.bev_flat_classes = bev_flat_classes
        self.fv_ignore_classes = fv_ignore_classes
        self.bev_ignore_classes = bev_ignore_classes
        self.fv_sky_index = fv_sky_index
        self.fv_veg_index = fv_veg_index

    def _makeRegionMask(self, msk, rgb_cameras):
        if (self.bev_vertical_classes is None) or (self.bev_flat_classes is None):
            return

        B = len(msk)
        W, Z = msk[0].shape[0], msk[0].shape[1]
        v_region_msk = torch.zeros((B, 1, W, Z), dtype=torch.long).to(msk[0].device)
        f_region_msk = torch.zeros((B, 1, W, Z), dtype=torch.long).to(msk[0].device)

        for b in range(B):
            for c in self.bev_vertical_classes:
                v_region_msk[b, 0, msk[b] == int(c)] = 1
            for c in self.bev_flat_classes:
                f_region_msk[b, 0, msk[b] == int(c)] = 1

        v_region_msk_out = []
        f_region_msk_out = []
        for cam_idx in range(len(rgb_cameras)):
            v_region_msk_out.append(v_region_msk)
            f_region_msk_out.append(f_region_msk)

        return v_region_msk_out, f_region_msk_out

    def _makeVFMask(self, msk):
        # This masks the FV semantics without the Sky label, i.e., they are in terms of the BEV label system.
        if (self.bev_vertical_classes is None) or (self.bev_flat_classes is None):
            return

        v_msk, f_msk = [], []
        for msk_b in msk:
            H, W = msk_b.shape[0], msk_b.shape[1]
            v_msk_b = torch.zeros((H, W), dtype=torch.long).to(msk_b.device)
            f_msk_b = torch.zeros((H, W), dtype=torch.long).to(msk_b.device)

            sem_msk = msk_b.detach().clone()
            sem_msk[sem_msk >= 1000] = torch.div(sem_msk[sem_msk >= 1000], 1000, rounding_mode="floor")

            for c in self.bev_vertical_classes:
                v_msk_b[sem_msk == int(c)] = 1
            for c in self.bev_flat_classes:
                f_msk_b[sem_msk == int(c)] = 1

            v_msk.append(v_msk_b)
            f_msk.append(f_msk_b)

        return v_msk, f_msk

    # Todo: Clean this up though...
    def _erodeVRegions(self, msk, label_img):
        label_img_tensor = torch.stack(label_img).unsqueeze(1)
        fv_sem_vertical_gt_curr = label_img_tensor.clone()
        flat_mask_curr = torch.stack(msk, dim=0).unsqueeze(1)
        fv_sem_vertical_gt_curr[flat_mask_curr == 1] = 255

        # Morphological operations on the vertical classes (FV)
        fv_erosion_kernel = torch.ones((9, 9), device=fv_sem_vertical_gt_curr.device)

        fv_sem_vertical_eroded_gt_curr = torch.ones_like(fv_sem_vertical_gt_curr) * 255
        for b in range(fv_sem_vertical_gt_curr.shape[0]):
            v_labels_b = torch.unique(fv_sem_vertical_gt_curr[b])
            for label_b in v_labels_b:
                if label_b == 255:
                    continue
                binary_mask_b = fv_sem_vertical_gt_curr[b] == label_b
                eroded_mask_b = kornia.morphology.erosion(binary_mask_b.unsqueeze(0).type(torch.uint8),
                                                          fv_erosion_kernel, max_val=255)
                fv_sem_vertical_eroded_gt_curr[b, eroded_mask_b.squeeze(0) > 0] = label_b

        return fv_sem_vertical_eroded_gt_curr

    def _getFlatFvLabels(self, msk, label_img):
        label_img_tensor = torch.stack(label_img).unsqueeze(1)
        fv_sem_vertical_gt_curr = label_img_tensor.clone()
        flat_mask_curr = torch.stack(msk, dim=0).unsqueeze(1)
        fv_sem_vertical_gt_curr[flat_mask_curr == 0] = 255

        return fv_sem_vertical_gt_curr


    def _prepare_inputs(self, msk, cat, iscrowd, front=True):
        if front:
            num_stuff = self.fv_num_stuff
        else:
            num_stuff = self.bev_num_stuff

        cat_out, iscrowd_out, bbx_out, ids_out, sem_out, sem_wo_sky_out, po_out, po_vis_out = [], [], [], [], [], [], [], []
        for msk_i, cat_i, iscrowd_i in zip(msk, cat, iscrowd):
            msk_i = msk_i.squeeze(0)
            thing = (cat_i >= num_stuff) & (cat_i != 255)
            valid = thing & ~(iscrowd_i > 0)

            if valid.any().item():
                cat_out.append(cat_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                ids_out.append(None)

            if iscrowd_i.any().item():
                iscrowd_i = (iscrowd_i > 0) & thing
                iscrowd_out.append(iscrowd_i[msk_i].type(torch.uint8))
            else:
                iscrowd_out.append(None)

            sem_msk_i = cat_i[msk_i]
            sem_out.append(sem_msk_i)

            # Get the FV image in terms of the BEV labels. This basically eliminates sky in the FV image
            if front:
                sem_wo_sky_veg_i = copy.deepcopy(sem_msk_i)
                sem_wo_sky_veg_i[sem_wo_sky_veg_i == self.fv_sky_index] = 255
                sem_wo_sky_veg_i[sem_wo_sky_veg_i == self.fv_veg_index] = 255
                for lbl in torch.unique(sem_wo_sky_veg_i):
                    decr_ctr = 0
                    if (lbl > self.fv_sky_index) and (lbl != 255):
                        decr_ctr += 1
                    if (lbl > self.fv_veg_index) and (lbl != 255):
                        decr_ctr += 1
                    sem_wo_sky_veg_i[sem_wo_sky_veg_i == lbl] = lbl - decr_ctr
                sem_wo_sky_out.append(sem_wo_sky_veg_i)

        if front:
            return cat_out, iscrowd_out, ids_out, sem_out, sem_wo_sky_out
        else:
            return cat_out, iscrowd_out, ids_out, sem_out

    def forward(self, img, bev_msk=None, fv_msk=None, bev_weights_msk=None, bev_cat=None, bev_iscrowd=None, fv_cat=None,
                fv_iscrowd=None, bev_bbx=None, fv_bbx=None, fv_intrinsics=None, ego_pose=None,
                transform_status=None, rgb_cameras=None, do_loss=False, do_prediction=False,cam_name=None, total_window_size=None, fvsem_window_size=None,
                fvsem_step_size=None, depth_offset=None, enforce_voxel_consistency=True, generate_bev_prediction=True):
        result = OrderedDict()
        loss = OrderedDict()
        stats = OrderedDict()

        # Get the index for the data at the current time step. This is exactly in the middle of the list
        idx_curr = len(img) // 2

        fv_depth_rec_loss, fv_depth_sem_loss, fv_depth_smth_loss, fv_depth_pred = [], [], [], []
        fv_img_shape = pad_packed_images(img[idx_curr])[0].shape[-2:]

        # ***** FV + BEV SEGMENTATION *****
        # Iterate through only the future frames for the FV segmentation.
        # We cannot reconstruct the past frames as we don't have the information visible at this time step
        # All the statements in the following section take place only during training.
        if do_loss:
            for i in range(idx_curr, idx_curr + fvsem_window_size + 1, fvsem_step_size):
                # Get/Prepare the input data and ground truth labels
                img_i, _ = pad_packed_images(img[i])
                fv_intrinsics_i = fv_intrinsics[i]

                # ToDo: Change this back later as it doesnt use the depth features (this was done for debugging only)
                # feat_depth_i = self.body_depth(img_i)
                ms_feat_i = self.body_depth(img_i)
                _, feat_depth_i, _ = self.voxel_grid_depth(ms_feat_i, fv_intrinsics_i, fv_img_shape=fv_img_shape)

                ############################# DEPTH NETWORK - ONLY FOR CURR_FRAME ##############################
                if i == idx_curr:
                    # ToDo: Create intrinsics_dict and use it because the intrinsics might change between frames
                    depth_pred_curr, disp_pred_curr = self.fv_depth_head(feat_depth_i, return_disparity=True)  # ToDo: This has to be padded too, otherwise there can be a size mismatch

                    for offset in depth_offset:
                        rgb_dict, sem_dict, T_dict, K_dict, vertical_mask_dict = dict(), dict(), dict(), dict(), dict()
                        offset_triple = [0, -offset, +offset]
                        for frame_id in offset_triple:
                            rgb_dict[frame_id], _ = pad_packed_images(img[idx_curr + frame_id])
                            T_dict[frame_id] = VoxelGridAlgo.compute_relative_transformation(
                                ego_pose[idx_curr + frame_id], ego_pose[idx_curr], transform_status)

                        # ToDo: Here, we may directly check which features to use for the depth head (backbone for dep
                        #  thhead vs voxel features for depthheadvoxel?                                                                                                      sem_dict=sem_dict)
                        fv_depth_loss_curr, fv_smth_loss_curr, fv_sem_silh_loss_curr, fv_depth_pred_curr = self.fv_depth_algo.training(depth_pred_curr,
                                                                                                                                       disp_pred_curr,
                                                                                                                                       None,
                                                                                                                                       rgb_dict=rgb_dict,T_dict=T_dict,
                                                                                                                                       intrinsics=fv_intrinsics[idx_curr],
                                                                                                                                       vertical_mask=None,
                                                                                                                                       sem_dict=None)

                        fv_depth_rec_loss.append(fv_depth_loss_curr)
                        fv_depth_sem_loss.append(fv_sem_silh_loss_curr)
                        fv_depth_smth_loss.append(fv_smth_loss_curr)
                        fv_depth_pred.append(fv_depth_pred_curr)

        else:
            fv_depth_rec_loss, fv_depth_sem_loss, fv_depth_smth_loss, fv_depth_pred = None, None, None, None

        # Accumulate the values.
        fv_depth_rec_loss_net = sum(fv_depth_rec_loss) / len(fv_depth_rec_loss)
        fv_depth_sem_loss_net = sum(fv_depth_sem_loss) / len(fv_depth_sem_loss)
        fv_depth_smth_loss_net = sum(fv_depth_smth_loss) / len(fv_depth_smth_loss)

        # LOSSES
        loss['fv_depth_rec_loss'] = fv_depth_rec_loss_net
        loss['fv_depth_sem_loss'] = fv_depth_sem_loss_net
        loss['fv_depth_smth_loss'] = fv_depth_smth_loss_net

        # RESULTS
        result["fv_depth_pred"] = fv_depth_pred

        return loss, result, stats


