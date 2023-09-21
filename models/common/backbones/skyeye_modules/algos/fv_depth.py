import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage.morphology import distance_transform_edt
from sklearn.cluster import DBSCAN
# from matplotlib import pyplot as plt

from po_bev_unsupervised.modules.losses import l1_pixelwise, SSIMLoss
from po_bev_unsupervised.utils.sequence import pack_padded_images
from po_bev_unsupervised.algos.coordinate_transform import _FvToPointcloud, CoordinateWarper, _PointcloudToFv, _FvPointcloudToBev


class FvDepthReconstructionLoss(nn.Module):
    def __init__(self, num_scales=6, alpha=0.85):
        super(FvDepthReconstructionLoss, self).__init__()
        self.fv_to_fv_warper = FvToFvWarper()
        self.ssim = SSIMLoss()
        self.num_scales = num_scales
        self.alpha = alpha

    def __call__(self, depth_map, rgb_dict, T_dict, intrinsics, use_automasking=True): #ToDo: Use an intrinsics dictionary here...
        loss = 0.0
        for j in range(self.num_scales):
            reconstruction_losses_s = []
            id_losses_s = []
            scale = 2 ** j

            # Get the scaled target image (curr frame)
            scaled_tgt_img = F.interpolate(rgb_dict[0], scale_factor=1 / scale, mode="bilinear", align_corners=True)

            for frame_id, _ in rgb_dict.items():
                if frame_id == 0:
                    continue

                # Warp the scaled, adjacent image (scaling is done within the function)
                warped_scaled_adjacent_img_ = self.fv_to_fv_warper(rgb_dict[frame_id], depth_map, T_dict[frame_id], intrinsics, scale)

                rec_l1_loss_s = l1_pixelwise(warped_scaled_adjacent_img_, scaled_tgt_img).mean(1, True)
                rec_ssim_loss_s = self.ssim(warped_scaled_adjacent_img_, scaled_tgt_img).mean(1, True)

                reconstruction_losses_s.append((1 - self.alpha) * rec_l1_loss_s + self.alpha * rec_ssim_loss_s)

                # ToDo: taking the minimum is not necessarily good, we should just exclude these pixels
                # Automasking
                if use_automasking:
                    # Get the scaled adjacent image first for identity losses
                    scaled_adjacent_img_ = F.interpolate(rgb_dict[frame_id], scale_factor=1 / scale, mode="bilinear", align_corners=True)

                    id_l1_loss_s = l1_pixelwise(scaled_adjacent_img_, scaled_tgt_img).mean(1, True)
                    id_ssim_loss_s = self.ssim(scaled_adjacent_img_, scaled_tgt_img).mean(1, True)
                    id_loss_s = (1 - self.alpha) * id_l1_loss_s + self.alpha * id_ssim_loss_s
                    id_loss_s += torch.randn(id_loss_s.shape, device=id_loss_s.device) * 0.00001
                    id_losses_s.append(id_loss_s)

            # Combine the identity and reconstruction losses and get the minimum out of it for handling
            # disocclusions/occlusions (reconstruction) and sequences where the scene is static or objects are moving
            # with the same velocity
            reconstruction_losses_s = torch.cat(reconstruction_losses_s, 1)

            # Adapt the photometric reconstruction loss with the uncertainty map
            reconstruction_losses_s = reconstruction_losses_s

            if use_automasking:
                id_losses_s = torch.cat(id_losses_s, 1)
                combined_losses_s = torch.cat((reconstruction_losses_s, id_losses_s), dim=1)
            else:
                combined_losses_s = reconstruction_losses_s

            loss_per_pixel_s, _ = torch.min(combined_losses_s, dim=1)
            mean_loss_s = loss_per_pixel_s.mean()

            loss += mean_loss_s / scale
        return loss

class FvDepthSemanticSilhouetteLoss(nn.Module):
    def __init__(self):
        super(FvDepthSemanticSilhouetteLoss, self).__init__()
        self.fv_to_fv_warper = FvToFvWarper()

    def __call__(self, depth_map, T_dict, intrinsics, vertical_mask, use_automasking=True): #ToDo: Use an intrinsics dictionary here...

        reference_mask = torch.stack(vertical_mask[0]).unsqueeze(1)

        # Flip vertical mask, move to cpu, compute distance transform, transform back to tensor and move back to GPU
        distance_img = torch.tensor(distance_transform_edt(1 - reference_mask.detach().cpu().numpy())).to(depth_map.device)

        cost_maps = []
        for frame_id, _ in vertical_mask.items():
            if frame_id == 0:
                continue

            adjacent_mask = torch.stack(vertical_mask[frame_id]).unsqueeze(1)
            #  ToDo: Penalize only vertical stuff,
            warped_adjacent_sem_img_ = self.fv_to_fv_warper(adjacent_mask, depth_map, T_dict[frame_id], intrinsics, 1, interp_mode="nearest", padding_mode="border", upsampling_mode="nearest")

            cost_vals = warped_adjacent_sem_img_ * distance_img
            cost_maps.append(cost_vals)

        # Get number of valid pixels for noramlization
        combined_silh_losses = torch.cat(cost_maps, 1)

        loss_per_pixel_s = torch.sum(combined_silh_losses, dim=1)

        loss = torch.sum(loss_per_pixel_s) / torch.sum(loss_per_pixel_s > 0)

        return loss


class FvDepthAlgo:
    def __init__(self, loss_rec, loss_smth, loss_sem_silh):
        self.loss_rec = loss_rec
        self.loss_smth = loss_smth
        self.loss_sem_silh = loss_sem_silh

    def _resize_rgb(self, rgb, img_size):
        rgb = F.interpolate(rgb, size=img_size, mode="bilinear", align_corners=False)
        return rgb

    @staticmethod
    def _pack_logits(depth_pred, valid_size, img_size):
        depth_pred = F.interpolate(depth_pred, size=img_size, mode="bilinear", align_corners=False)
        return pack_padded_images(depth_pred, valid_size)

    def _depth_dist_to_metric(self, implicit_depth_dist, z_max_voxel):
        # ToDo IMP: Starting with 0 or with 1, talk to Nikhil what convention he uses!
        depth_bins = torch.arange(0, implicit_depth_dist.shape[1]).to(implicit_depth_dist.device)
        metric_depth_bins = depth_bins * z_max_voxel / implicit_depth_dist.shape[1]
        metric_depth_bins = metric_depth_bins.unsqueeze(0).repeat(implicit_depth_dist.shape[0], 1).unsqueeze(2).unsqueeze(3)

        return torch.sum(implicit_depth_dist*metric_depth_bins, dim=1, keepdim=True)

    def training(self, depth_pred, disp_pred=None, z_max_voxel=None,
                 rgb_dict=None, T_dict=None, intrinsics=None, vertical_mask=None, sem_dict=None,
                 do_loss=True):
        if do_loss:
            # ToDo: Use an intrinsics
            # dictionary here for different intrinsics
            valid_size = [depth_pred.shape[-2:] for i in range(depth_pred.shape[0])]
            img_size = depth_pred.shape[-2:]

            # Resize all rgb iamges
            for frame_id, rgb in rgb_dict.items():
                rgb_dict[frame_id] = self._resize_rgb(rgb, img_size)

            # Go through all scales and compute the corresponding reconstruction loss
            rec_loss = self.loss_rec(depth_pred, rgb_dict, T_dict, intrinsics)

            # Attention: We apply the smoothness loss directly on the sigmoid output to avoid that the depth range has an
            # impact on the smoothness loss (e.g. bigger range also scales the derivatives and hence the smoothness loss)
            disp_pred_mean = disp_pred.mean(2, True).mean(3, True)
            disp_pred_norm = disp_pred / (disp_pred_mean + 1e-7)

            smth_loss = self.loss_smth(disp_pred_norm, rgb_dict[0])
            # sem_silh_loss = self.loss_sem_silh(depth_pred, T_dict, intrinsics, vertical_mask)
            sem_silh_loss = torch.tensor(0.).to(depth_pred.device)

            return rec_loss, smth_loss, sem_silh_loss, depth_pred
        else:
            rec_loss = torch.abs(depth_pred - depth_pred)

            return rec_loss

    def training_implicit_depth(self,  depth_pred, implicit_depth_dist=None, z_max_voxel=None):
        # Get metric continuous depth (implicit) from distribution map
        if implicit_depth_dist is not None:
            implicit_metric_depth = self._depth_dist_to_metric(implicit_depth_dist, z_max_voxel)
            # Interpolate implicit depth to the FV size to compare with the predicted depth, i.e. the implicit depth will be very coarse
            implicit_metric_depth_up = F.interpolate(implicit_metric_depth, size=depth_pred.size()[2:], mode='nearest')

            msk = (depth_pred <= z_max_voxel).detach()
            implicit_depth_loss = torch.abs(implicit_metric_depth_up[msk] - depth_pred.detach()[msk]).mean()
        else:
            implicit_depth_loss = torch.tensor(0.).to(depth_pred.device)

        return implicit_depth_loss

class FvDepthToBevWarper(nn.Module):
    def __init__(self, extents, resolution):
        super(FvDepthToBevWarper, self).__init__()
        self.fv_to_pointcloud = _FvToPointcloud()
        self.coordinate_warper = CoordinateWarper()
        self.fv_pointcloud_to_bev = _FvPointcloudToBev(extents, resolution)

    def forward(self, fv_label_img, depth_map, T, intrinsics, bev_image_shape):
        assert depth_map.dim() == 4, 'The input batch of source images has {} dimensions which is != 4'.format(depth_map.dim())

        # Get batch size
        batch_size = depth_map.shape[0]

        # Transform T to homogeneous coordinates by extending it by an additional row of 0, 0, 0, 1
        hom_row = torch.tensor([0, 0, 0, 1]).unsqueeze(0)
        hom_row = hom_row.repeat(batch_size, 1, 1).to(depth_map.device)
        T = torch.cat([T, hom_row], 1)

        # Transform FV image into camera coordinate system using the depth map
        fv_as_pointcloud = self.fv_to_pointcloud(depth_map, intrinsics)

        # Apply pose transformation on the pointcloud
        transformed_bev_pointcloud = self.coordinate_warper(fv_as_pointcloud, T)

        # Propagate fv_img labels to the BEV space
        bev_label_img = self.fv_pointcloud_to_bev(fv_label_img, transformed_bev_pointcloud, bev_image_shape)

        return bev_label_img


 # ToDo: We can make this static actually...
class FvToFvWarper(nn.Module):
    """ Associates each FV pixel with FV pixels of a selected camera inside of the window

    """
    def  __init__(self):
        super(FvToFvWarper, self).__init__()
        self.fv_to_pointcloud = _FvToPointcloud()
        self.coordinate_warper = CoordinateWarper()
        self.pointcloud_to_fv = _PointcloudToFv()


    def forward(self, fv_img_adjacent, depth_map, T, intrinsics, scale, interp_mode="bilinear", padding_mode="zeros", upsampling_mode="bilinear"):
        """Associates each FV pixel with FV pixels of a selected camera inside of the window

        Parameters
        ----------
        bev_height_map : torch.Tensor
            A tensor with shape B x 1 x H x W containing the height of each pixel in the bev image.

        Returns
        -------
        pixel_coordinates : torch.Tensor
            A tensor with shape B x H x W x 2 containing the pixel coordinates for each frontal view image
        """
        assert depth_map.dim() == 4, 'The input batch of source images has {} dimensions which is != 4'.format(depth_map.dim())
        assert depth_map.shape[1] == 1, 'The input batch of bev height maps has {} channels which is != 1'.format(depth_map.dim())

        # Scale images and intrinsics to the correct scale
        intrinsics_scale = intrinsics[:, :3, :3] / scale
        intrinsics_scale[:, 2, 2] = 1
        depth_map = F.interpolate(depth_map, scale_factor=1 / scale, mode="bilinear", align_corners=True)
        fv_img_adjacent = F.interpolate(fv_img_adjacent, scale_factor=1 / scale, mode=upsampling_mode).type(torch.float)
        fv_img_shape = fv_img_adjacent.shape[2:]

        # Get batch size
        batch_size = depth_map.shape[0]

        # Transform T to homogeneous coordinates by extending it by an additional row of 0, 0, 0, 1
        hom_row = torch.tensor([0, 0, 0, 1]).unsqueeze(0)
        hom_row = hom_row.repeat(batch_size, 1, 1).to(depth_map.device)
        T = torch.cat([T, hom_row], 1)

        # Transform FV image into camera coordinate system using the depth map
        fv_as_pointcloud = self.fv_to_pointcloud(depth_map, intrinsics_scale)

        transformed_bev_pointcloud = self.coordinate_warper(fv_as_pointcloud, T)

        # Reproject transformed BEV pointcloud into the image
        pixel_coordinates = self.pointcloud_to_fv(transformed_bev_pointcloud, intrinsics_scale, fv_img_shape)

        warped_img = F.grid_sample(fv_img_adjacent, pixel_coordinates, mode=interp_mode, padding_mode=padding_mode, align_corners=False)

        return warped_img

class PclWarper(nn.Module):
    def  __init__(self):
        super(PclWarper, self).__init__()
        self.fv_to_pointcloud = _FvToPointcloud()
        self.coordinate_warper = CoordinateWarper()

    def forward(self, depth_map, T, intrinsics):

        assert depth_map.dim() == 4, 'The input batch of source images has {} dimensions which is != 4'.format(depth_map.dim())
        assert depth_map.shape[1] == 1, 'The input batch of bev height maps has {} channels which is != 1'.format(depth_map.dim())

        # Get batch size
        batch_size = depth_map.shape[0]

        # Transform T to homogeneous coordinates by extending it by an additional row of 0, 0, 0, 1
        hom_row = torch.tensor([0, 0, 0, 1]).unsqueeze(0)
        hom_row = hom_row.repeat(batch_size, 1, 1).to(depth_map.device)
        T = torch.cat([T, hom_row], 1)

        # Transform FV image into camera coordinate system using the depth map
        fv_as_pointcloud = self.fv_to_pointcloud(depth_map, intrinsics)

        transformed_pointcloud = self.coordinate_warper(fv_as_pointcloud, T)

        return transformed_pointcloud

def get_dynamic_weight_map(sem_tn, sem_t0, depth_pred, depth_dist_unproj, vxl_to_fv_idx_map, T, fv_intrinsics, fv_dynamic_classes, scale=1):
    fv_to_fv_warper = FvToFvWarper()

    consistent_map_list = []
    for idx, (sem_tn_i, sem_t0_i) in enumerate(zip(sem_tn, sem_t0)):
        warped_t0_image_i = fv_to_fv_warper(sem_t0_i.unsqueeze(0).unsqueeze(0), depth_pred[idx].unsqueeze(0).detach(),
                                            T[idx].unsqueeze(0), fv_intrinsics[idx].unsqueeze(0),1, interp_mode="nearest",
                                            upsampling_mode="nearest",padding_mode="border")
        error_map_i = abs(warped_t0_image_i - sem_tn_i.unsqueeze(0))
        potential_dynamic_mask_i = torch.zeros_like(sem_tn_i.unsqueeze(0), dtype=torch.bool)
        for c in fv_dynamic_classes:
            potential_dynamic_mask_i[sem_tn_i.unsqueeze(0) == c] = True
        error_map_i[~potential_dynamic_mask_i.unsqueeze(0)] = 0
        consistent_map_i = (error_map_i == 0)
        consistent_map_list.append(consistent_map_i.type(torch.float).squeeze(0))

    # Generate the dynamic mask for the voxel grid using the error or consistency map and the depth_pred
    # Step1: Lift the consistency map from 2D to 3D
    consistent_map = torch.stack(consistent_map_list, dim=0).detach().clone()  # Should be between 0 and 1 now
    # Scale the consistent map to match the dimensions of the FV features. We are trying to replicate the lifting here
    consistent_map = F.interpolate(consistent_map, scale_factor=0.125, mode="bilinear", align_corners=True)
    B, D, H, W, _ = vxl_to_fv_idx_map.shape
    consistent_map_3d = consistent_map.unsqueeze(2)
    consistent_voxel = F.grid_sample(consistent_map_3d, vxl_to_fv_idx_map, padding_mode="zeros")

    # Step2: Mask the lifted consistency map with the depth distribution
    consistent_voxel = consistent_voxel * depth_dist_unproj

    return consistent_map_list, consistent_voxel

def get_depth_weight_map(depth_pred_adj, T, intrinsics, voxel_max_z=55):
    pcl_warper = PclWarper()
    transformed_pcl = pcl_warper(depth_pred_adj, T, intrinsics)
    z_vals = transformed_pcl[:,2,:,:].unsqueeze(1).type(torch.float)
    outlier_mask = z_vals > voxel_max_z
    depth_weight_map = torch.ones(depth_pred_adj.shape, dtype=torch.float).to(depth_pred_adj.device)
    depth_weight_map[outlier_mask] = abs(z_vals[outlier_mask] - voxel_max_z) / 255.0

class Clusterer():
    def __init__(self):
        self.fv_to_pointcloud = _FvToPointcloud()

    def forward(self, depth_map, intrinsics, filtered_fv_semantics, eps, min_pts):
        depth_as_pcl = self.fv_to_pcl(depth_map, intrinsics)
        depth_arr = torch.flatten(depth_as_pcl, 2)
        msk_arr = torch.flatten(filtered_fv_semantics, 2).repeat(1, 3, 1)
        sem_labels = torch.unique(filtered_fv_semantics)
        filtered_pcl = depth_arr[msk_arr != 255].view(depth_arr.shape[0], depth_arr.shape[1], -1).permute(0, 2, 1)

        in_db = filtered_pcl[0].detach().cpu().numpy()
        db = DBSCAN(eps=0.2, min_samples=5).fit(in_db)

        return db

    # def output_result(self, db):
    #     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #     core_samples_mask[db.core_sample_indices_] = True
    #     labels = db.labels_
    #     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #     n_noise_ = list(labels).count(-1)

    # def visualize_cluster(self, db, in_db, cluster_idx):
    #     cluster_0 = in_db[db.labels_ == 0, :]
    #     cluster_1 = in_db[db.labels_ == 1, :]
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(cluster_0[:, 0], cluster_0[:, 2], cluster_0[:, 1])
    #     plt.show()

