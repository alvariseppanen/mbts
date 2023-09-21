import copy
import time
from collections import OrderedDict
import kornia
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from po_bev_unsupervised.utils.sequence import pad_packed_images
from po_bev_unsupervised.utils.visualisation import save_semantic_output, save_semantic_masked_output, save_depth_output, save_semantic_output_with_rgb_overlay


TRANSFORMER_ORDER = ['front', 'left', 'right', 'rear']
NETWORK_INPUTS_TRAIN = ["img", "fv_msk", "fv_cat", "fv_iscrowd", "bev_msk", "bev_plabel","bev_cat", "bev_iscrowd", "fv_intrinsics", "ego_pose"]
NETWORK_INPUTS_VAL_KITTI = ["img", "fv_msk", "fv_cat", "fv_iscrowd", "bev_msk", "bev_plabel", "bev_cat","bev_iscrowd", "fv_intrinsics", "ego_pose"]
NETWORK_INPUTS_VAL_WAYMO = ["img", "fv_msk", "fv_cat", "fv_iscrowd", "fv_intrinsics", "ego_pose"]


class UnsupervisedBevNet(nn.Module):
    def __init__(self,
                 body,
                 body_depth,
                 voxel_grid,
                 voxel_grid_depth,
                 fv_sem_head,
                 fv_depth_head,
                 bev_sem_head,
                 voxel_grid_algo,
                 fv_sem_algo,
                 fv_depth_algo,
                 bev_sem_algo,
                 dataset,
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
        super(UnsupervisedBevNet, self).__init__()

        # Backbone
        self.body = body
        self.body_depth = body_depth

        # Transformer
        self.voxel_grid = voxel_grid
        self.voxel_grid_depth = voxel_grid_depth

        # Modules
        self.fv_sem_head = fv_sem_head
        self.fv_depth_head = fv_depth_head
        self.bev_sem_head = bev_sem_head

        # Algorithms
        self.voxel_grid_algo = voxel_grid_algo
        self.fv_sem_algo = fv_sem_algo
        self.fv_depth_algo = fv_depth_algo
        self.bev_sem_algo = bev_sem_algo

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

    def forward(self, img, bev_msk=None, bev_plabel=None, fv_msk=None, bev_weights_msk=None, bev_cat=None, bev_iscrowd=None,
                fv_cat=None, fv_iscrowd=None, fv_intrinsics=None, ego_pose=None, transform_status=None,
                rgb_cameras=None, do_loss=False, use_fv=False, use_bev=False, fvsem_window_size=None, fvsem_step_size=None,
                share_fv_kernel_with_bev=False, save_tuple=None, rgb_mean=None, rgb_std=None):
        result = OrderedDict()
        loss = OrderedDict()
        stats = OrderedDict()

        # Process plabels to have the correct format:
        if bev_plabel is not None:
            bev_plabel, bev_plabel_valid_img_size_i = pad_packed_images(bev_plabel[0])
            bev_plabel = list(torch.tensor_split(bev_plabel, bev_plabel.shape[0], dim=0))
            bev_plabel = [elem.squeeze() for elem in bev_plabel]

        # Get the index for the data at the current time step. This is exactly in the middle of the list
        if self.dataset == "Kitti360":
            idx_curr = len(img) // 2
        elif self.dataset == "Waymo":
            idx_curr = 0

        # fv_depth_rec_loss = []
        fv_img_shape = pad_packed_images(img[idx_curr])[0].shape[-2:]

        fv_sem_loss, fv_sem_conf_mat, fv_sem_pred, fv_sem_logits = [], [], [], []
        bev_sem_loss, bev_sem_conf_mat, bev_sem_pred, bev_sem_logits = [], [], [], []

        # ***** FV + BEV SEGMENTATION *****
        # Iterate through only the future frames for the FV segmentation.
        # We cannot reconstruct the past frames as we don't have the information visible at this time step
        for i in range(idx_curr, min(len(img), idx_curr + fvsem_window_size + 1), fvsem_step_size):
            # Get/Prepare the input data and ground truth labels
            img_i, _ = pad_packed_images(img[i])
            ego_pose_i = ego_pose[i]
            fv_intrinsics_i = fv_intrinsics[i]

            if (fv_msk is not None) and (fv_msk[i] is not None):
                fv_msk_i, fv_valid_size_i = pad_packed_images(fv_msk[i])
                fv_img_size_i = fv_msk_i.shape[-2:]
                fv_cat_i = fv_cat[i]
                fv_iscrowd_i = fv_iscrowd[i]

            if (bev_msk is not None) and (bev_msk[i] is not None):
                bev_msk_i, bev_valid_size_i = pad_packed_images(bev_msk[i])
                bev_img_size_i = bev_msk_i.shape[-2:]
                bev_cat_i = bev_cat[i]
                bev_iscrowd_i = bev_iscrowd[i]

            # Prepare the input data and the groundtruth labels
            if fv_msk is not None:
                fv_cat_i, fv_iscrowd_i, fv_ids_i, fv_sem_gt_i, fv_sem_wo_sky_gt_i = self._prepare_inputs(fv_msk_i, fv_cat_i, fv_iscrowd_i, front=True)
            if bev_msk is not None:
                bev_cat_i, bev_iscrowd_i, bev_ids_i, bev_sem_gt_i = self._prepare_inputs(bev_msk_i, bev_cat_i, bev_iscrowd_i, front=False)

            # Generate the voxel grid for all the frames
            ms_feat_i = self.body(img_i)
            feat_voxel_i, feat_merged_2d_i, implicit_depth_dist_i, implicit_depth_dist_unproj_i, vxl_to_fv_idx_map_i = self.voxel_grid(ms_feat_i, fv_intrinsics_i, fv_img_shape=fv_img_shape)
            del ms_feat_i

            if i == idx_curr:
                feat_voxel_curr = feat_voxel_i

            ############################# ONLY VOXEL WARPING ####################################
            if do_loss and i != idx_curr:
                feat_voxel_i_warped = self.voxel_grid_algo.ego_gt_warper(feat_voxel_curr, ego_pose[idx_curr], ego_pose_i, transform_status)
            else:
                feat_voxel_i_warped = feat_voxel_curr

            ############################# FV SEGMENTATION ##############################
            if use_fv:
                # Orthographic to perspective distortion
                feat_voxel_i_warped_persp = self.voxel_grid_algo.apply_perspective_distortion(feat_voxel_i_warped, fv_img_size_i, fv_intrinsics_i)

                if do_loss:
                    fv_sem_loss_i, fv_sem_conf_mat_i, fv_sem_pred_i, fv_sem_logits_i = self.fv_sem_algo.training_fv(self.fv_sem_head,
                                                                                                                    feat_voxel_i_warped_persp,
                                                                                                                    fv_sem_gt_i, fv_valid_size_i,
                                                                                                                    fv_img_size_i, None,
                                                                                                                    fv_intrinsics_i,
                                                                                                                    dyn_obj_msk=None,
                                                                                                                    depth_weight_map=None)
                else:
                    fv_sem_pred_i, fv_sem_logits_i = self.fv_sem_algo.inference_fv(self.fv_sem_head,
                                                                                   feat_voxel_i_warped_persp,
                                                                                   fv_valid_size_i,
                                                                                   fv_intrinsics_i)
                    fv_sem_loss_i, fv_sem_conf_mat_i = None, None

                fv_sem_loss.append(fv_sem_loss_i)
                fv_sem_conf_mat.append(fv_sem_conf_mat_i)
                fv_sem_pred.append(fv_sem_pred_i)
                fv_sem_logits.append(fv_sem_logits_i)
            else:
                fv_sem_loss = None
                fv_sem_conf_mat = None
                fv_sem_pred = None
                fv_sem_logits = None

            if use_bev:
                # Generate the BEV for the current frame only.
                if i == idx_curr:
                    if not share_fv_kernel_with_bev:
                        if do_loss and (bev_plabel is not None):  # During training
                            bev_sem_loss_i, bev_sem_pred_i, bev_sem_logits_i, bev_height_map_i = self.bev_sem_algo.training_bev(self.bev_sem_head, feat_voxel_i, bev_plabel, None, fv_intrinsics_i)
                            bev_sem_conf_i = self.bev_sem_algo.compute_bev_metrics_with_gt(bev_sem_pred_i, bev_sem_gt_i)
                        else:
                            bev_sem_pred_i , bev_sem_logits_i, bev_height_map_i = self.bev_sem_algo.inference_bev(self.bev_sem_head, feat_voxel_i, fv_intrinsics_i)
                            bev_sem_loss_i, bev_sem_conf_i = None, None

                        bev_sem_loss.append(bev_sem_loss_i)
                        bev_sem_pred.append(bev_sem_pred_i)
                        bev_sem_logits.append(bev_sem_logits_i)
                        bev_sem_conf_mat.append(bev_sem_conf_i)
                    else:
                        bev_sem_pred_i, bev_sem_logits_i = self.bev_sem_algo.generate_bev_using_fv(self.fv_sem_head, feat_voxel_curr, remove_channels=[self.fv_veg_index, self.fv_sky_index])
                        bev_sem_conf_mat_i = self.bev_sem_algo.compute_bev_metrics_with_gt(bev_sem_logits_i, bev_sem_gt_i)
                        bev_sem_pred.append(bev_sem_pred_i)
                        bev_sem_logits.append(bev_sem_logits_i)
                        bev_sem_conf_mat.append(bev_sem_conf_mat_i)
            else:
                bev_sem_loss = None
                bev_sem_pred = None
                bev_sem_conf = None
                bev_sem_logits = None

        if use_fv and (fv_sem_loss is not None) and len(fv_sem_loss) > 0:
            fv_sem_loss_count = len(fv_sem_loss)
            fv_sem_loss_weights = torch.linspace(1, 0.2, fv_sem_loss_count).tolist()
            fv_sem_loss_sum = sum([w * l for w, l in zip(fv_sem_loss_weights, fv_sem_loss)])
            fv_sem_loss_net = fv_sem_loss_sum / fv_sem_loss_count
        else:
            fv_sem_loss_net = torch.tensor(0.).to(img[idx_curr].device)

        if use_bev and (bev_sem_loss is not None) and len(bev_sem_loss) > 0:
            bev_sem_loss_net = sum(bev_sem_loss) / len(bev_sem_loss)
        else:
            bev_sem_loss_net = torch.tensor(0.).to(img[idx_curr].device)

        if use_fv and (fv_msk is not None):
            fv_sem_conf_mat_net = torch.zeros(self.fv_num_classes, self.fv_num_classes, dtype=torch.double).to(fv_sem_conf_mat[0].device)
            for conf_mat in fv_sem_conf_mat:
                fv_sem_conf_mat_net += conf_mat

        if use_bev and (self.bev_sem_algo is not None):
            bev_sem_conf_mat_net = torch.zeros(self.bev_num_classes, self.bev_num_classes, dtype=torch.double).to(bev_sem_conf_mat[0].device)
            for conf_mat in bev_sem_conf_mat:
                bev_sem_conf_mat_net += conf_mat

        # LOSSES
        if use_fv:
            loss["fv_sem_loss"] = fv_sem_loss_net
        if use_bev:
            loss['bev_sem_loss'] = bev_sem_loss_net

        # RESULTS
        if use_fv:
            result["fv_sem_pred"] = fv_sem_pred
            result['fv_sem_logits'] = fv_sem_logits
        if use_bev:
            result["bev_sem_pred"] = bev_sem_pred
            result['bev_sem_logits'] = bev_sem_logits

        # STATS
        if do_loss:
            if use_fv:
                stats['fv_sem_conf'] = fv_sem_conf_mat_net
            if use_bev:
                stats['bev_sem_conf'] = bev_sem_conf_mat_net


        # Save all the required outputs here
        if save_tuple is not None:
            if bev_sem_pred is not None:
                bev_sem_pred_unpack = [pad_packed_images(pred)[0] for pred in bev_sem_pred]
                save_semantic_output(bev_sem_pred_unpack, "bev_sem_pred", save_tuple, bev=True, dataset=self.dataset)
            if bev_msk is not None:
                bev_sem_gt_unpack = [pad_packed_images(gt)[0] for gt in bev_msk]
                bev_sem_gt_unpack = [self._prepare_inputs(bev_sem_gt_unpack[vis_ts], bev_cat[vis_ts], bev_iscrowd[vis_ts], front=False)[-1][0] for vis_ts in range(len(bev_sem_gt_unpack))]
                bev_sem_gt_unpack = [gt.unsqueeze(0) for gt in bev_sem_gt_unpack]
                save_semantic_output(bev_sem_gt_unpack, "bev_sem_gt", save_tuple, bev=True, dataset=self.dataset)
                if bev_sem_pred is not None:
                    bev_sem_pred_unpack = [pad_packed_images(pred)[0] for pred in bev_sem_pred]
                    save_semantic_masked_output(bev_sem_pred_unpack, bev_sem_gt_unpack, "bev_sem_pred_masked", save_tuple, bev=True, dataset=self.dataset)
            if fv_sem_pred is not None:
                fv_sem_pred_unpack = [pad_packed_images(pred)[0] for pred in fv_sem_pred]
                save_semantic_output(fv_sem_pred_unpack, "fv_sem_pred", save_tuple, bev=False, dataset=self.dataset)

                img_unpack = [pad_packed_images(rgb)[0] for rgb in img]
                img_unpack = [img_unpack[i] for i in range(idx_curr, min(len(img), idx_curr + fvsem_window_size + 1), fvsem_step_size)]
                save_semantic_output_with_rgb_overlay(fv_sem_pred_unpack, img_unpack, "fv_sem_pred_rgb_overlay",
                                                      save_tuple, bev=False, dataset=self.dataset,
                                                      rgb_mean=rgb_mean, rgb_std=rgb_std)
            if fv_msk is not None:
                # Without vegetation and sky
                fv_sem_gt_unpack = [pad_packed_images(gt)[0] for gt in fv_msk]
                fv_sem_gt_unpack_woskyveg = [
                    self._prepare_inputs(fv_sem_gt_unpack[vis_ts], fv_cat[vis_ts], fv_iscrowd[vis_ts], front=True)[-1][
                        0] for vis_ts in range(len(fv_sem_gt_unpack))]
                fv_sem_gt_unpack_woskyveg = [gt.unsqueeze(0) for gt in fv_sem_gt_unpack_woskyveg]
                save_semantic_output(fv_sem_gt_unpack_woskyveg, "fv_sem_woskyveg_gt", save_tuple, bev=False,
                                     woskyveg=True, dataset=self.dataset)

                # With all classes
                fv_sem_gt_unpack = [self._prepare_inputs(fv_sem_gt_unpack[vis_ts], fv_cat[vis_ts], fv_iscrowd[vis_ts], front=True)[-2][0] for vis_ts in range(len(fv_sem_gt_unpack))]
                fv_sem_gt_unpack = [gt.unsqueeze(0) for gt in fv_sem_gt_unpack]
                save_semantic_output(fv_sem_gt_unpack, "fv_sem_gt", save_tuple, bev=False, dataset=self.dataset)

            # if fv_depth_pred is not None:
            #     fv_depth_pred_unpack = [d.squeeze(1) for d in fv_depth_pred]
            #     save_depth_output(fv_depth_pred_unpack, "fv_depth_pred", save_tuple)

        return loss, result, stats


if __name__ == "__main__":
    from po_bev_unsupervised.algos.coordinate_transform import FvToBevWarper
    from po_bev_unsupervised.utils.visualisation import visualise_semantic_mask_train_id
    from po_bev_unsupervised.utils.transformer import getInitHomography
    import matplotlib.pyplot as plt

    resolution = 25 / 336
    W_out = 768
    Z_out = 704
    extents = [(W_out * resolution / 2), 0, -(W_out * resolution / 2), Z_out * resolution]

    fv_extrinsics = {"translation": (0, 0, 0), "rotation": (-5, 0, 0)}
    fv_to_bev_warper = FvToBevWarper(extents, resolution, fv_extrinsics)

    # FV Semantic Mask
    # img_path = "/home/gosalan/data/kitti360_bev_seam_poses/front_msk_trainid/front/2013_05_28_drive_0010_sync;0000000053.png"
    # fv_sem_msk_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # fv_sem_msk_np[fv_sem_msk_np > 1000] = fv_sem_msk_np[fv_sem_msk_np > 1000] // 1000
    # fv_sem_msk = fv_sem_msk_np.astype(np.uint8)
    # fv_sem_msk = torch.from_numpy(fv_sem_msk).unsqueeze(0).unsqueeze(0)
    # sem_msk = True

    # RGB image
    sem_msk = False
    img_path = "/home/gosalan/data/kitti360_dataset/data_2d_raw/2013_05_28_drive_0004_sync/image_00/data_rect/0000000053.png"
    fv_sem_msk_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    fv_sem_msk = fv_sem_msk_np.astype(np.uint8)
    fv_sem_msk = torch.from_numpy(fv_sem_msk).permute(2, 0, 1).unsqueeze(0)

    # Generate a fixed height map
    ht_map = torch.ones([1, 1, Z_out, W_out], dtype=torch.float) * 2
    intrinsics = torch.tensor([[552.554261, 0, 682.049453],
                               [0, 552.554261, 238.769549],
                               [0, 0, 1]], dtype=torch.float).unsqueeze(0)

    T_curr2i = torch.cat([torch.eye(3, 3), torch.zeros(3, 1)], dim=1).unsqueeze(0)

    fv_to_bev_coords = fv_to_bev_warper(ht_map, T_curr2i, intrinsics, (376, 1408))

    # Create a 4-dim tensor from list, warp the GT images and transform back to list
    fv_to_bev_sem_gt_i_tensor = F.grid_sample(fv_sem_msk, fv_to_bev_coords, mode="nearest", padding_mode="zeros")

    if sem_msk:
        vis_fv = visualise_semantic_mask_train_id(fv_sem_msk.squeeze(0), "Kitti360", False)
        vis_bev = visualise_semantic_mask_train_id(fv_to_bev_sem_gt_i_tensor.squeeze(0), "Kitti360", True)
    else:
        vis_fv = fv_sem_msk.squeeze(0)
        vis_bev = fv_to_bev_sem_gt_i_tensor.squeeze(0).type(torch.uint8)

    vis_fv = vis_fv.permute(1, 2, 0).cpu().numpy()
    vis_bev = vis_bev.permute(1, 2, 0).cpu().numpy()
    plt.imshow(vis_fv)
    plt.show()
    plt.imshow(vis_bev)
    plt.show()

    # IPM
    extrinsics = {"translation": (0.8, 0.3, 1.55), "rotation": (-85, 0, 180)}
    intrinsics = np.array([[552.554261, 0, 682.049453],
                               [0, 552.554261, 238.769549],
                               [0, 0, 1]])
    bev_params = {"f": 336, "cam_z": -24.1}
    H = getInitHomography(intrinsics, extrinsics, bev_params, 1, (384, 1408))
    H = H.numpy()
    warp_img = cv2.warpPerspective(fv_sem_msk_np, H, (Z_out, W_out), flags=cv2.INTER_NEAREST)
    if sem_msk:
        vis_ipm = visualise_semantic_mask_train_id(torch.from_numpy(warp_img.astype(np.uint8)).unsqueeze(0), "Kitti360", False)
        vis_ipm = vis_ipm.permute(1, 2, 0).cpu().numpy()
    else:
        vis_ipm = torch.from_numpy(warp_img.astype(np.uint8)).cpu().numpy()
    plt.imshow(vis_ipm)
    plt.show()





