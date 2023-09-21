import torch
import torch.nn as nn
import torch.nn.functional as F

from po_bev_unsupervised.algos.voxel_grid import VoxelGridAlgo
from po_bev_unsupervised.modules.losses import SSIMLoss
from po_bev_unsupervised.utils.transformer import computeExtrinsicMatrix
from po_bev_unsupervised.algos.coordinate_transform import _BevToPointcloud, CoordinateWarper, _PointcloudToFv, _PointcloudToBev


# Consistency between the RGB images
class BevReconstructionLoss(nn.Module):
    def __init__(self, fv_to_bev_warper, num_scales=4, alpha=0.85):
        super(BevReconstructionLoss, self).__init__()
        self.fv_to_bev_warper = fv_to_bev_warper
        self.ssim = SSIMLoss()
        self.num_scales = num_scales
        self.alpha = alpha
        self.ssim = SSIMLoss()


    def __call__(self, bev_ht_map, img_curr, img_i, ego_pose_curr, ego_pose_i, intrinsics, alpha=0.85):
        bev_recon_loss = []
        for j in range(self.num_scales):
            scale = 2 ** j

            # BEV segmentation, warp the i-th frontal view image into the BEV image and compare them
            T_neutral = VoxelGridAlgo.compute_relative_transformation(ego_pose_curr, ego_pose_curr)
            T_curr2i = VoxelGridAlgo.compute_relative_transformation(ego_pose_i, ego_pose_curr)

            # warp curr_idx-th rgb image
            warped_rgb_curr = self.fv_to_bev_warper(img_curr, bev_ht_map, T_neutral, intrinsics, scale=scale)

            # warp i-th rgb image
            warped_rgb_i = self.fv_to_bev_warper(img_i, bev_ht_map, T_curr2i, intrinsics, scale=scale)

            # Compute losses
            ssim_loss = self.ssim(warped_rgb_i, warped_rgb_curr).mean(1, True).mean()
            l1_loss = (warped_rgb_i - warped_rgb_curr).abs().mean(1, True).mean()

            bev_recon_loss_s = (1 - alpha) * l1_loss + alpha * ssim_loss
            bev_recon_loss.append(bev_recon_loss_s / scale)

        # Get the mean loss from all scaled losses
        bev_recon_loss_mean = sum(bev_recon_loss) / len(bev_recon_loss)
        return bev_recon_loss_mean


class BevRGBReconstructionAlgo:
    def __init__(self, loss):
        self.loss = loss
        self.num_scales = loss.num_scales

    def training(self, bev_ht_map, img_curr, img_i, ego_pose_curr, ego_pose_i, intrinsics):
        # ToDo: Output warped image maybe for visulalization purposes?
        bev_rgb_rec_loss = self.loss(bev_ht_map, img_curr, img_i, ego_pose_curr, ego_pose_i, intrinsics)
        return bev_rgb_rec_loss


class FvToBevWarper(nn.Module):
    """ Associates each BEV cell with FV pixels of a selected camera inside of the window

    Parameters
    ----------
    intrinsics: dict
        Pinhole camera intrinsics

    """
    def  __init__(self, extents, resolution, fv_extrinsics):
        super(FvToBevWarper, self).__init__()
        self.bev_to_pointcloud = _BevToPointcloud(extents, resolution)
        self.coordinate_warper = CoordinateWarper()
        self.pointcloud_to_fv = _PointcloudToFv()
        self.fv_extrinsics = fv_extrinsics

    def forward(self, fv_img, bev_height_map, T, intrinsics, scale, warp_mode="bilinear"):
        """Associates each BEV pixel with FV pixels of a selected camera inside of the window

        Parameters
        ----------
        bev_height_map : torch.Tensor
            A tensor with shape B x 1 x H x W containing the height of each pixel in the bev image.

        Returns
        -------
        pixel_coordinates : torch.Tensor
            A tensor with shape B x H x W x 2 containing the pixel coordinates for each frontal view image
        """
        assert bev_height_map.dim() == 4, 'The input batch of source images has {} dimensions which is != 4'.format(bev_height_map.dim())
        assert bev_height_map.shape[1] == 1, 'The input batch of bev height maps has {} channels which is != 1'.format(bev_height_map.dim())

        batch_size = bev_height_map.shape[0]

        # Rescale everything according to the input scale
        if scale != 1:
            intrinsics = intrinsics[:, :3, :3] / scale
            intrinsics[:, 2, 2] = 1
            fv_img = F.interpolate(fv_img, scale_factor=1 / scale, mode="bilinear", align_corners=True)
            bev_height_map = F.interpolate(bev_height_map, scale_factor=1 / scale, mode="bilinear", align_corners=True)

        fv_img_shape = fv_img.shape[-2:]

        # Transform T to homogeneous coordinates by extending it by an additional row of 0, 0, 0, 1
        hom_row = torch.tensor([0, 0, 0, 1]).unsqueeze(0)
        hom_row = nn.Parameter(hom_row.repeat(batch_size, 1, 1), requires_grad=False).to(bev_height_map.device)
        T = torch.cat([T, hom_row], 1)

        # Transform BEV image into camera coordinate system
        bev_as_pointcloud = self.bev_to_pointcloud(bev_height_map, scale)

        # Transform the points so that they account for the extrinsics of the FV camera
        # TODO: Check the T_ext, does it transform correctly?
        T_ext = computeExtrinsicMatrix(self.fv_extrinsics['translation'], self.fv_extrinsics['rotation'])
        T_ext = torch.cat([torch.from_numpy(T_ext).unsqueeze(0)] * batch_size, dim=0).type(torch.float).to(bev_as_pointcloud.device)
        T_ext = torch.cat([T_ext, hom_row], 1)
        bev_as_pointcloud = self.coordinate_warper(bev_as_pointcloud, T_ext)

        # Transform BEV pointcloud into coordinate systems of the FV RGB camera
        transformed_bev_pointcloud = self.coordinate_warper(bev_as_pointcloud, T)

        # Reproject transformed BEV pointcloud into the image
        pixel_coordinates = self.pointcloud_to_fv(transformed_bev_pointcloud, intrinsics, fv_img_shape)

        warped_img = F.grid_sample(fv_img.float(), pixel_coordinates, mode=warp_mode, padding_mode="zeros")

        return warped_img



class BevToBevWarper(nn.Module):
    """ Warps a source bev image into the coordinate system of the target camera by using the target height predictions.
    The output can directly be compared with the target image
    """
    def __init__(self, extents, resolution):
        super(BevToBevWarper, self).__init__()
        self.bev_to_pointcloud = _BevToPointcloud(extents, resolution)
        self.coordinate_warper = CoordinateWarper()
        self.pointcloud_to_bev = _PointcloudToBev()

    def forward(self, bev_src_height_map, bev_tgt_height_map, T):
        """Associates each BEV pixel with FV pixels of a selected camera inside of the window

        Parameters
        ----------
        bev_src_height_map : torch.Tensor
            A tensor with shape B x 1 x H x W containing the height and semantic logits of the source bev image that is
            to be warped to tgt bev image.
        bev_tgt_height_map : torch.Tensor
            A tensor with shape B x 1 x H x W of which we use the height information to warp the source BEV image into
            the target BEV image
        Returns
        -------
        warped_bev_image : torch.Tensor
            A tensor with shape B x C x H x W containing the warped source bev image such that it can directly be compared
            with the target bev image
        """
        assert bev_src_height_map.dim() == 4, 'The input batch of source images has {} dimensions which is != 4'.format(bev_src_height_map.dim())
        assert bev_tgt_height_map.dim() == 4, 'The input batch of source images has {} dimensions which is != 4'.format(bev_tgt_height_map.dim())
        assert bev_src_height_map.shape[1] == 1, 'The input batch of bev height maps has {} channels which is != 1'.format(bev_height_map.dim())
        assert bev_tgt_height_map.shape[1] == 1, 'The input batch of bev height maps has {} channels which is != 1'.format(bev_height_map.dim())

        # Transform BEV image into camera coordinate system
        bev_tgt_as_pointcloud = self.bev_to_pointcloud(bev_tgt_height_map)

        # Transform BEV pointcloud into coordinate systems of the src camera
        transformed_bev_tgt_pointcloud = self.coordinate_warper(bev_tgt_as_pointcloud, T)

        # Collapse the height to get the pixel associations between the target and source BEV image
        pixel_coordinates = self.pointcloud_to_bev(transformed_bev_tgt_pointcloud)

        # Warp src->tgt bev image
        warped_bev_image = F.grid_sample(bev_src_height_map, pixel_coordinates, padding_mode="zeros")

        return warped_bev_image
