import copy
import kornia
import torch
import torch.nn as nn
from po_bev_unsupervised.utils.sequence import pad_packed_images
from po_bev_unsupervised.algos.fv_depth import get_dynamic_weight_map


TRANSFORMER_ORDER = ['front', 'left', 'right', 'rear']
NETWORK_INPUTS = ["img", "fv_msk", "bev_msk", "bev_cat", "bev_iscrowd", "fv_intrinsics", "ego_pose"]


class UnsupervisedPlabelNet(nn.Module):
    def __init__(self,
                 body,
                 body_depth,
                 voxel_grid,
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
                 clusterer,
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
                 sem_plabels_accum=None,
                 sem_plabels_single=None,
                 sem_plabels_flat=None,
                 sem_plabels_cluster=None,
                 sem_plabels_cluster_eps=None,
                 sem_plabels_cluster_min_pts=None,
                 fv_sky_index=None,
                 fv_veg_index=None):
        super(UnsupervisedPlabelNet, self).__init__()

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
        self.clusterer = clusterer

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
        self.sem_plabels_accum = sem_plabels_accum
        self.sem_plabels_single = sem_plabels_single
        self.sem_plabels_flat = sem_plabels_flat
        self.sem_plabels_cluster = sem_plabels_cluster
        self.sem_plabels_cluster_eps = sem_plabels_cluster_eps
        self.sem_plabels_cluster_min_pts = sem_plabels_cluster_min_pts
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

    def _prepare_inputs_pred(self, msk):
        sem_wo_sky_out = []
        for sem_msk_i in msk:
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
        return sem_wo_sky_out

    def forward(self, img, bev_msk=None, fv_msk=None, bev_cat=None, bev_iscrowd=None,
                fv_intrinsics=None, ego_pose=None, fvsem_window_size=None, fvsem_step_size=None):

        # Get the index for the data at the current time step. This is exactly in the middle of the list
        idx_curr = len(img) // 2

        fv_img_shape = pad_packed_images(img[idx_curr])[0].shape[-2:]

        semantic_pcls = {label_idx: [] for label_idx in (self.sem_plabels_accum + self.sem_plabels_single)}

        # ***** PSEUDOLABEL GENERATION *****
        # All the statements in the following section take place only during training.
        for i in range(idx_curr, idx_curr + fvsem_window_size + 1, fvsem_step_size):
            # Get/Prepare the input data and ground truth labels
            img_i, _ = pad_packed_images(img[i])
            ego_pose_i = ego_pose[i]
            fv_intrinsics_i = fv_intrinsics[i]

            if (fv_msk is not None) and (fv_msk[i] is not None):
                fv_msk_i, fv_valid_size_i = pad_packed_images(fv_msk[i])

            if (bev_msk is not None) and (bev_msk[i] is not None):
                bev_msk_i, bev_valid_size_i = pad_packed_images(bev_msk[i])
                bev_img_size_i = bev_msk_i.shape[-2:]
                bev_cat_i = bev_cat[i]
                bev_iscrowd_i = bev_iscrowd[i]

            # Prepare the input data and the groundtruth labels
            fv_sem_wo_sky_gt_i = self._prepare_inputs_pred(fv_msk_i)
            fv_sem_wo_sky_gt_i = [fv_sem_img_j.squeeze() for fv_sem_img_j in fv_sem_wo_sky_gt_i]

            bev_cat_i, bev_iscrowd_i, bev_ids_i, bev_sem_gt_i = self._prepare_inputs(bev_msk_i, bev_cat_i, bev_iscrowd_i, front=False)
            vertical_mask_i, flat_mask_i = self._makeVFMask(fv_sem_wo_sky_gt_i)

            if i == idx_curr:
                fv_sem_wo_sky_gt_curr = fv_sem_wo_sky_gt_i

            # Create features for the depth head
            with torch.no_grad():
                ms_feat_depth_i = self.body_depth(img_i)
                _, feat_merged_depth_i, _ = self.voxel_grid_depth(ms_feat_depth_i, fv_intrinsics_i, fv_img_shape=fv_img_shape)
                depth_pred_i, disp_pred_i = self.fv_depth_head(feat_merged_depth_i.detach(), return_disparity=True)  # ToDo: This has to be # padded too, otherwise there can be a size mismatch
                del ms_feat_depth_i, feat_merged_depth_i

            # Generate the voxel grid for all the frames
            ms_feat_i = self.body(img_i)
            feat_voxel_i, feat_merged_2d_i, implicit_depth_dist_i = self.voxel_grid(ms_feat_i, fv_intrinsics_i, fv_img_shape=fv_img_shape)

            # Get pose difference and the fv sem tensor
            T_curr2i = self.voxel_grid_algo.compute_relative_transformation(ego_pose_i, ego_pose[idx_curr])
            T_i2curr = self.voxel_grid_algo.compute_relative_transformation(ego_pose[idx_curr], ego_pose_i)

            # Create a 4-dim tensor from list, warp the GT images and transform back to list
            fv_sem_wo_sky_gt_i_tensor = torch.stack(fv_sem_wo_sky_gt_i).unsqueeze(1)

            ############################ PSEUDOLABEL GENERATION ##############################
            # USING BEV HEIGHT MAP
            # Propagate labels from FV to BEV via predicted height map

            # if i == idx_curr or True:
            # Todo: Continue here, you need to replace bev_height_map_i...
            bev_height_map_i = torch.ones(depth_pred_i.shape[0], 1, self.bev_sem_head.bev_Z_out,
                                          self.bev_sem_head.bev_W_out).to(depth_pred_i.device) * 1.55
            bev_height_map_i = torch.flip(bev_height_map_i, dims=[3])

            # Create flat BEV labels
            fv_to_bev_sem_flat_ipm_gt_i = self.bev_sem_algo.create_flat_plabels(
                self.fv_to_bev_warper, fv_sem_wo_sky_gt_i_tensor, bev_height_map_i, flat_mask_i,
                T_curr2i, fv_intrinsics_i)

            # Create flat mask in the BEV space
            flat_mask_bev_i = self.bev_sem_algo.create_flat_bev_mask(fv_to_bev_sem_flat_ipm_gt_i)

            # Erode vertical semantics in the FV space
            fv_sem_vertical_eroded_gt_i = self._erodeVRegions(flat_mask_i, fv_sem_wo_sky_gt_i)

            # Create vertical BEV labels
            fv_to_bev_sem_vertical_depth_gt_i = self.fv_depth_to_bev_warper(fv_sem_vertical_eroded_gt_i,
                                                                            depth_pred_i.detach(),
                                                                            T_i2curr,
                                                                            fv_intrinsics[i],
                                                                            bev_height_map_i.shape[-2:])

            # Erode vertical plabels in the BEV space
            fv_to_bev_sem_vertical_eroded_gt_i = self.bev_sem_algo.erode_bev_plabels(fv_to_bev_sem_vertical_depth_gt_i)

            # Combine flat and vertical bev plabels with pre-computed flat mask. The rotation is for the
            bev_combined_supervision = self.bev_sem_algo.combine_vertical_flat_bev_plabels(fv_to_bev_sem_vertical_eroded_gt_i, fv_to_bev_sem_flat_ipm_gt_i, flat_mask_bev_i)

            if i == idx_curr:
                bev_sem_gt_curr = bev_sem_gt_i
                bev_sem_gt_curr_rot = [torch.rot90(bev_img, k=3, dims=[0, 1]) for bev_img in bev_sem_gt_curr]
                bev_combined_supervision_curr = bev_combined_supervision
            plabel = [torch.rot90(pseudo_map, k=1, dims=[0, 1]) for pseudo_map in bev_combined_supervision]

            plabel_map = [torch.ones_like(map_i).cpu().to(torch.uint8) * 255 for map_i in bev_combined_supervision]

            dynamic_mask_i = get_dynamic_weight_map(fv_sem_wo_sky_gt_i, fv_sem_wo_sky_gt_curr,
                                                    depth_pred_i.detach(), T_i2curr,
                                                    fv_intrinsics[i], self.front_dynamic_classes)

            # for i, dynamic_weight_map_i in enumerate(dynamic_mask_i):
            #     fv_sem_vertical_eroded_gt_i[i][dynamic_weight_map_i == 0] = 255
            # 2.) Filter point clouds by semantic label and accumulate them or get the pcl for a
            # single time step only
            if self.sem_plabels_accum is not None:
                semantic_pcls_i = self.clusterer.get_semantic_pcls(depth_pred_i,
                                                                   fv_intrinsics[i],
                                                                   fv_sem_vertical_eroded_gt_i,
                                                                   fv_sem_wo_sky_gt_i_tensor,
                                                                   T_i2curr,
                                                                   self.sem_plabels_accum,
                                                                   self.front_dynamic_classes,
                                                                   dynamic_mask_i)

                if self.sem_plabels_single is not None and i == idx_curr:
                    semantic_pcls_single_i = self.clusterer.get_semantic_pcls(depth_pred_i,
                                                                              fv_intrinsics[i],
                                                                              fv_sem_vertical_eroded_gt_i,
                                                                              fv_sem_wo_sky_gt_i_tensor,
                                                                              T_i2curr,
                                                                              self.sem_plabels_single)

                    semantic_pcls_i.update(semantic_pcls_single_i)

                if not list(semantic_pcls.values())[0]:
                    semantic_pcls = semantic_pcls_i
                else:
                    for b in range(len(depth_pred_i)):
                        for accum_label_idx in self.sem_plabels_accum:
                            semantic_pcls[accum_label_idx][b] = torch.cat((semantic_pcls[accum_label_idx][b], semantic_pcls_i[accum_label_idx][b]))

            # 3.) Translate flat semantic point clouds into the BEV space as no clustering is
            # required here and put them together into a joint map (road > sidewalk > terrain)
            if self.sem_plabels_flat is not None and i == idx_curr + fvsem_window_size:
                plabel_map = self.clusterer.create_plabels(depth_pred_i.shape[0],
                                                                semantic_pcls,
                                                                self.sem_plabels_flat,
                                                                bev_height_map_i.shape[-2:],
                                                                plabel_map)

                # apply erosion
                # plabel_map = self.bev_sem_algo.erode_bev_plabels(plabel_map)

            if self.sem_plabels_single is not None and i == idx_curr + fvsem_window_size:
                # Fixme: ATTENTION: The labels for erosion or non erosion in the following part are
                #  hard-coded
                plabel_map = self.clusterer.create_plabels(depth_pred_i.shape[0],
                                                           semantic_pcls,
                                                           (2,),
                                                           bev_height_map_i.shape[-2:],
                                                           plabel_map)

                plabel_map = self.bev_sem_algo.erode_bev_plabels(plabel_map)

                # plabel_map = self.clusterer.create_plabels(depth_pred_i.shape[0],
                #                                            semantic_pcls,
                #                                            (4,5,),
                #                                            bev_height_map_i.shape[-2:],
                #                                            plabel_map)

            # 4.) Cluster some vertical classes and fit ellipses for bounding box creation
            if self.sem_plabels_cluster is not None and i == idx_curr + fvsem_window_size:
                plabel_map = self.clusterer.create_clustered_plabels(depth_pred_i.shape[0],
                                                                     semantic_pcls,
                                                                     bev_height_map_i.shape[-2:],
                                                                     self.sem_plabels_cluster,
                                                                     self.sem_plabels_cluster_eps,
                                                                     self.sem_plabels_cluster_min_pts, 20, 3,
                                                                     plabel_map)

                stop_here = 0
                # ToDo: Visualize the clusters here

        return plabel_map, bev_sem_gt_curr, bev_sem_gt_curr_rot, bev_combined_supervision_curr
