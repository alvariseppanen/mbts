"""Provide the BEV image warping layer based on a pre-defined camera model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class _BevToPointcloud(nn.Module):
    """Transforms the local BEV coordinate system into the camera coordinate system assuming that the camera sits at the
    origin of the BEV coordinate system (x_cam=x_bev, y_cam:=height, z_cam:=z_bev)

    BEV:
                z
                ^
                |
                |
                |
                |
    ----------cam-------->x,

    Parameters
    ----------
    extents :
        Metric extents of the BEV image
    resolution : float
        Resolution to sample the grid with
    """
    def __init__(self, extents, resolution):
        super(_BevToPointcloud, self).__init__()
        self.extents = extents
        self.resolution = resolution

    # Attention: bev_height_map expects the first channel of the BEV image to be the height
    def forward(self, bev_height_map, scale=1):
        """Transform the bev image into a pointcloud

        Parameters
        ----------
        bev_height_map : torch.Tensor
            A tensor with shape B x 1 x H x W containing the height (first channel) of the bev image
        Returns
        -------
        '...' : torch.Tensor
            A tensor with shape B x C_euc x H x W containing the bev as pointcloud in the local camera coordinate system
        """
        assert bev_height_map.dim() == 4, 'The input batch of BEV images has {} dimensions which is != 4'.format(bev_height_map.dim())

        bev_img_height, bev_img_width = bev_height_map.shape[2:]

        x1, z1, x2, z2 = self.extents
        x3d_bev = torch.arange(start=x1, end=x2, step=-self.resolution*scale).expand(bev_img_height, bev_img_width).to(bev_height_map.device)
        z3d_bev = torch.arange(start=z1, end=z2, step=self.resolution*scale).expand(bev_img_width, bev_img_height).t().to(bev_height_map.device)

        # Adapt x3d_bev and y3d_bev to size of the height map
        x3d_bev = x3d_bev.unsqueeze(0).unsqueeze(0).repeat(bev_height_map.shape[0], 1, 1, 1)
        z3d_bev = z3d_bev.unsqueeze(0).unsqueeze(0).repeat(bev_height_map.shape[0], 1, 1, 1)

        return torch.cat((x3d_bev, bev_height_map, z3d_bev), dim=1)

# class _FvPointcloudToBev(nn.Module):
#     def __init__(self, extents, resolution):
#         super(_FvPointcloudToBev, self).__init__()
#         self.extents = extents
#         self.resolution = resolution
#
#     # Attention: bev_height_map expects the first channel of the BEV image to be the height
#     def forward(self, fv_pointcloud, bev_image_shape, scale=1):
#         # fv_pointcloud.shape: B x 3 x fv_img_height, fv_img_width
#         # ToDo: Rename bev image size
#         bev_img_height, bev_img_width = bev_image_shape
#
#         x1, z1, x2, z2 = self.extents
#         x3d_bev = torch.arange(start=x1, end=x2, step=-self.resolution*scale).expand(bev_img_height, bev_img_width).to(fv_pointcloud.device)
#         z3d_bev = torch.arange(start=z1, end=z2, step=self.resolution*scale).expand(bev_img_width, bev_img_height).t().to(fv_pointcloud.device)
#
#         # Adapt x3d_bev and y3d_bev to size of the height map
#         x3d_bev = x3d_bev.unsqueeze(0).unsqueeze(0).repeat(fv_pointcloud.shape[0], 1, 1, 1)
#         z3d_bev = z3d_bev.unsqueeze(0).unsqueeze(0).repeat(fv_pointcloud.shape[0], 1, 1, 1)
#
#         # Create a zero height map and set the heights as predicted by the depth map
#         height_map = torch.zeros(x3d_bev.shape).to(fv_pointcloud.device)+255
#         validity_mask = torch.zeros(x3d_bev.shape, dtype=torch.bool)
#
#         # TODO: They order between idx and x3d_bev might be reverted! TAKE CARE of this
#         # Convert metric depth map coordinates (pointcloud) to bev idxs
#         x_idx = fv_pointcloud[:, 0, :, :].unsqueeze(1)*(-1) / (self.resolution * scale) + bev_img_width/2
#         z_idx = fv_pointcloud[:, 2, :, :].unsqueeze(1) / (self.resolution * scale)
#
#         x_idx_fv = torch.arange(0, fv_pointcloud.shape[-1]).expand(fv_pointcloud.shape[-2], fv_pointcloud.shape[-1]).unsqueeze(0).unsqueeze(0).repeat(fv_pointcloud.shape[0], 1, 1, 1)
#         y_idx_fv = torch.arange(0, fv_pointcloud.shape[-2]).expand(fv_pointcloud.shape[-1], fv_pointcloud.shape[-2]).t().to(fv_pointcloud.device).unsqueeze(0).unsqueeze(0).repeat(fv_pointcloud.shape[0], 1, 1, 1)
#
#         # The depth map might yield coordinates outside of the pre-defined bev map. Remove this from the idx list
#         is_valid_x = torch.logical_and(x_idx < bev_img_width, x_idx >= 0)
#         is_valid_z = torch.logical_and(z_idx < bev_img_height, z_idx >= 0)
#         is_valid = torch.logical_and(is_valid_x, is_valid_z)
#
#         # Constrain the idx matrices
#         x_idx = x_idx[is_valid].view(fv_pointcloud.shape[0], 1, -1).round().long()
#         z_idx = z_idx[is_valid].view(fv_pointcloud.shape[0], 1, -1).round().long()
#         x_idx_fv = x_idx_fv[is_valid].view(fv_pointcloud.shape[0], 1, -1)
#         y_idx_fv = y_idx_fv[is_valid].view(fv_pointcloud.shape[0], 1, -1)
#
#         # Set height map via depth prediction that lay within the pre-defined bev map
#         for b in range(height_map.shape[0]):
#             height_map_b = height_map[b].squeeze(0)
#             fv_pointcloud_b = fv_pointcloud[b, 1, :, :]
#             height_map_b[x_idx[b].squeeze(0), z_idx[b].squeeze(0)] = fv_pointcloud_b[y_idx_fv[b].squeeze(0), x_idx_fv[b].squeeze(0)]
#             height_map[b] = height_map_b
#
#         validity_mask[height_map > 254] = True
#         validity_mask[height_map <= 254] = False
#
#         return torch.cat((x3d_bev, height_map, z3d_bev), dim=1), validity_mask

class _FvPointcloudToBev(nn.Module):
    def __init__(self, extents, resolution):
        super(_FvPointcloudToBev, self).__init__()
        self.extents = extents
        self.resolution = resolution

    # Attention: bev_height_map expects the first channel of the BEV image to be the height
    def forward(self, fv_label_img, fv_pointcloud, bev_image_shape, scale=1):
        # fv_pointcloud.shape: B x 3 x fv_img_height, fv_img_width
        # ToDo: Rename bev image size
        bev_img_height, bev_img_width = bev_image_shape

        # Create a bev map with invalid labels first
        x1, z1, x2, z2 = self.extents
        x3d_bev = torch.arange(start=x1, end=x2, step=-self.resolution*scale).expand(bev_img_height, bev_img_width).to(fv_pointcloud.device)
        z3d_bev = torch.arange(start=z1, end=z2, step=self.resolution*scale).expand(bev_img_width, bev_img_height).t().to(fv_pointcloud.device)

        # Adapt x3d_bev and y3d_bev to size of the height map
        x3d_bev = x3d_bev.unsqueeze(0).unsqueeze(0).repeat(fv_pointcloud.shape[0], 1, 1, 1)
        z3d_bev = z3d_bev.unsqueeze(0).unsqueeze(0).repeat(fv_pointcloud.shape[0], 1, 1, 1)

        bev_label_img = (torch.ones(x3d_bev.shape, dtype=torch.long) * 255).to(fv_pointcloud.device)

        # Get all valid depth measurements (valid: predicted depth does not go out of the bev map extents)
        # and set the labels accordingly in the bev map

        # Convert metric depth map coordinates (pointcloud) to bev idxs
        x_idx = fv_pointcloud[:, 0, :, :].unsqueeze(1)*(-1) / (self.resolution * scale) + bev_img_width//2
        z_idx = fv_pointcloud[:, 2, :, :].unsqueeze(1) / (self.resolution * scale)
        # We also need to keep track of the fv idxs as they will also be removed partly
        x_idx_fv = torch.arange(0, fv_pointcloud.shape[-1]).expand(fv_pointcloud.shape[-2], fv_pointcloud.shape[-1]).unsqueeze(0).unsqueeze(0).repeat(fv_pointcloud.shape[0], 1, 1, 1)
        y_idx_fv = torch.arange(0, fv_pointcloud.shape[-2]).expand(fv_pointcloud.shape[-1], fv_pointcloud.shape[-2]).t().to(fv_pointcloud.device).unsqueeze(0).unsqueeze(0).repeat(fv_pointcloud.shape[0], 1, 1, 1)

        for b in range(bev_label_img.shape[0]):
            x_idx_b = x_idx[b].round().long()
            z_idx_b = z_idx[b].round().long()
            x_idx_fv_b = x_idx_fv[b]
            y_idx_fv_b = y_idx_fv[b]
            is_valid_x_b = torch.logical_and(x_idx_b >= 0, x_idx_b < bev_img_width)
            is_valid_z_b = torch.logical_and(z_idx_b >= 0, z_idx_b < bev_img_height)
            is_valid_b = torch.logical_and(is_valid_x_b, is_valid_z_b)

            # Constrain the idx matrices
            x_idx_b = x_idx_b[is_valid_b].view(1, -1)
            z_idx_b = z_idx_b[is_valid_b].view(1, -1)
            x_idx_fv_b = x_idx_fv_b[is_valid_b].view(1, -1)
            y_idx_fv_b = y_idx_fv_b[is_valid_b].view(1, -1)

            # Set the bev labels
            bev_label_img_b = bev_label_img[b].squeeze(0)
            fv_label_img_b = fv_label_img[b].squeeze(0)
            # bev_label_img_b[z_idx_b.squeeze(0), x_idx_b.squeeze(0)] = fv_label_img_b[y_idx_fv_b.squeeze(0), x_idx_fv_b.squeeze(0)]
            bev_label_img_b[z_idx_b, x_idx_b] = fv_label_img_b[y_idx_fv_b, x_idx_fv_b]
            bev_label_img[b] = bev_label_img_b

        return bev_label_img


class _FvToPointcloud(nn.Module):
    """Projects all images of the batch into the 3d world (batch of pointclouds)
    """
    def __init__(self):
        super(_FvToPointcloud, self).__init__()

    def get_viewing_ray(self, u2d, v2d, intrinsics):
        # Compute a vector that points in the direction of the viewing ray (assuming a depth of 1)
        ray_x = (u2d - intrinsics[:, 0, 2].view(intrinsics.shape[0], 1, 1, 1)) / intrinsics[:, 0, 0].view(intrinsics.shape[0], 1, 1, 1)
        ray_y = (v2d - intrinsics[:, 1, 2].view(intrinsics.shape[0], 1, 1, 1)) / intrinsics[:, 1, 1].view(intrinsics.shape[0], 1, 1, 1)
        ray_z = 1.0

        # Compute the norm of the ray vector #
        norm = torch.sqrt(ray_x ** 2 + ray_y ** 2 + ray_z ** 2)

        # Normalize the ray to obtain a unit vector
        ray_x /= norm
        ray_y /= norm
        ray_z /= norm

        return [ray_x, ray_y, ray_z]

    def forward(self, fv_depth_map, intrinsics):
        assert fv_depth_map.dim() == 4, 'The input batch of depth maps has {} dimensions which is != 4'.format(fv_depth_map.dim())
        assert fv_depth_map.size(1) == 1, 'The input batch of depth maps has {} channels which is != 1'.format(fv_depth_map.size(1))

        # Define a grid of pixel coordinates for the corresponding image size. Each entry defines specific grid
        # pixel coordinates for which the viewing ray is to be computed
        fv_img_height, fv_img_width = fv_depth_map.shape[2:]
        u2d_vals = torch.arange(start=0, end=fv_img_width).expand(fv_img_height, fv_img_width).float().to(fv_depth_map.device)
        v2d_vals = torch.arange(start=0, end=fv_img_height).expand(fv_img_width, fv_img_height).t().float().to(fv_depth_map.device)

        # Adapt u2d_vals and v2d_vals to size of the height map
        u2d_vals = u2d_vals.unsqueeze(0).unsqueeze(0).repeat(fv_depth_map.shape[0], 1, 1, 1)
        v2d_vals = v2d_vals.unsqueeze(0).unsqueeze(0).repeat(fv_depth_map.shape[0], 1, 1, 1)

        rays_x, rays_y, rays_z = self.get_viewing_ray(u2d_vals, v2d_vals, intrinsics)

        x3d = rays_x / abs(rays_z) * fv_depth_map
        y3d = rays_y / abs(rays_z) * fv_depth_map
        z3d = rays_z / abs(rays_z) * fv_depth_map

        return torch.cat((x3d, y3d, z3d), dim=1)


class _PointcloudToBev(nn.Module):
    """Transforms the BEV pointcloud back into local BEV coordinates
    """
    def __init__(self):
        super(_PointcloudToBev, self).__init__()

    # Attention: bev_image expects the first channel of the BEV image to be the height
    def forward(self, bev_pointcloud):
        """Transform the bev pointcloud back into a BEV image

        Parameters
        ----------
        bev_pointcloud : torch.Tensor
            A tensor with shape B x 3 x H x W containing the euclidean coordinates of the BEV pointcloud

        Returns
        -------
        '...' : torch.Tensor
            A tensor with shape B x 3 x H x W containing the bev as pointcloud in the local camera coordinate system
        """
        assert bev_pointcloud.dim() == 4, 'The input batch of BEV images has {} dimensions which is != 4'\
            .format(bev_pointcloud.dim())

        batch_size = bev_pointcloud.size(0)
        bev_img_height, bev_img_width = bev_pointcloud[2:]

        # Get the BEV image coordinates (just collapse height and associate coordinates accordingly)
        x_bev = bev_pointcloud[:, 0, :, :]
        y_bev = bev_pointcloud[:, 2, :, :]

        # Normalize the coordinates to [-1, +1] as required for grid_sample
        x_bev_norm = (x_bev / (bev_img_width -1) - 0.5) * 2
        y_bev_norm = (y_bev / (bev_img_height -1) - 0.5) * 2

        # Put the coordinates together and reshape them
        pixel_coordinates = torch.stack([x_bev_norm, y_bev_norm], dim=2)
        pixel_coordinates = pixel_coordinates.view(batch_size, bev_img_height, bev_img_width, 2)

        return pixel_coordinates


class CoordinateWarper(nn.Module):
    """ Transforms the bev pointcloud that is projected into the camera coordinate system of another camera
    """
    def __init__(self):
        super(CoordinateWarper, self).__init__()

    def forward(self, bev_image_as_pointcloud, T):
        """Transform the bev pointcloud into the camera coordinate system of another camera

        Parameters
        ----------
        bev_image_as_pointcloud : torch.Tensor
            A tensor with shape B x C_euc x H x W containing the euclidean coordinates of the bev pointcloud, where
            C_euc = 3
        T :
            3x3 matrix including both the rotation and translation between the cameras

        Returns
        -------
        transformed_pointcloud : torch.Tensor
            A tensor with shape B x C_euc x H x W containing the transformed bev pointcloud
        """

        bev_img_height, bev_img_width = bev_image_as_pointcloud.shape[2:]

        # Transform the pointcloud into homogeneous coordinates
        ones = torch.ones(bev_image_as_pointcloud.size(0), 1, bev_img_height, bev_img_width, dtype=torch.float, requires_grad=True).to(bev_image_as_pointcloud.device)
        image_as_pointcloud_homogeneous = torch.cat([bev_image_as_pointcloud, ones], 1)

        # Transform the obtained pointcloud into the local coordinate system of the target camera pose (homogeneous)
        transformed_pointcloud = torch.bmm(torch.inverse(T), image_as_pointcloud_homogeneous.view(bev_image_as_pointcloud.size(0), 4, -1))
        transformed_pointcloud = transformed_pointcloud.view(-1, 4, bev_img_height, bev_img_width)

        # Transform back to Euclidean coordinates
        transformed_pointcloud = transformed_pointcloud[:, :3, :, :] / transformed_pointcloud[:, 3, :, :].unsqueeze(1)

        return transformed_pointcloud


class _PointcloudToFv(nn.Module):
    """Reprojects all pointclouds of the batch into the image and returns a new batch of the correspponding 2d image points
    As such, this associates each BEV point with a 2D coordinate in the FV image

    Parameters
    ----------
    intrinsics: dict
        Pinhole camera intrinsics
    """

    def __init__(self):
        super(_PointcloudToFv, self).__init__()

    def forward(self, pointcloud, intrinsics, fv_img_shape):
        """Transform the bev pointcloud into the camera coordinate system of the FV RGB camera

        Parameters
        ----------
        pointcloud : torch.Tensor
            A tensor with shape B x C_euc x H x W containing the euclidean coordinates of the bev pointcloud, where
            C_euc = 3

        Returns
        -------
        pixel_coordinates : torch.Tensor
            A tensor with shape B x H x W x 2 containing the pixel coordinates for each frontal view image
        """
        # Fetch data from the batch of BEV images (1 dim for batch, 2 dims for BEV space dimensions and 1 for x,y,z values)
        assert pointcloud.dim() == 4, 'The input pointcloud has {} dimensions which is != 4'.format(pointcloud.dim())
        assert pointcloud.size(1) == 3, 'The input pointcloud has {} channels which is != 3 (x,y,z)'.format(pointcloud.size(1))

        bev_img_height, bev_img_width = pointcloud.shape[2:]

        batch_size = pointcloud.size(0)
        x3d = pointcloud[:, 0, :, :].view(batch_size, -1)
        y3d = pointcloud[:, 1, :, :].view(batch_size, -1)
        z3d = pointcloud[:, 2, :, :].clamp(min=1e-5).view(batch_size, -1)

        # Compute the pixel coordinates
        u2d = (x3d * intrinsics[:, 0, 0].unsqueeze(1) / z3d) + intrinsics[:, 0, 2].unsqueeze(1)
        v2d = (y3d * intrinsics[:, 1, 1].unsqueeze(1) / z3d) + intrinsics[:, 1, 2].unsqueeze(1)

        # Put the u2d and v2d vectors together and reshape them
        u2d_norm = ((u2d / (fv_img_shape[1] - 1)) - 0.5) * 2  # Normalise to [-1, 1]
        v2d_norm = ((v2d / (fv_img_shape[0] - 1)) - 0.5) * 2   # Normalise to [-1, 1]

        pixel_coordinates = torch.stack([u2d_norm, v2d_norm], dim=2) # dim: batch_size, H_bevxW_bev, 2

        pixel_coordinates = pixel_coordinates.view(batch_size, bev_img_height, bev_img_width, 2)

        return pixel_coordinates





if __name__ == "__main__":
    print("hello test")

