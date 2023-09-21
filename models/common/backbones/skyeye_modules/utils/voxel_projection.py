import torch
import torch.nn as nn

def generate_3d_to_2d_idx_mapping(intrinsics, voxel_extents, resolution, feat_2d_shape, **varargs):

    # Generate the homogeneous coordinates for each voxel in the voxel grid
    x1, x2, z1, z2, y1, y2 = voxel_extents
    zz, yy, xx = torch.arange(z1, z2, resolution).to(varargs['device']), \
                 torch.arange(y1, y2, resolution).to(varargs["device"]), \
                 torch.arange(x1, x2, resolution).to(varargs['device'])

    zz_idx, yy_idx, xx_idx = torch.arange(0, zz.shape[0], 1, dtype=torch.long).to(varargs['device']), \
                             torch.arange(0, yy.shape[0], 1, dtype=torch.long).to(varargs['device']), \
                             torch.arange(0, xx.shape[0], 1, dtype=torch.long).to(varargs['device'])

    pts_3d = torch.cartesian_prod(xx, yy, zz).to(varargs["device"])
    pts_3d = torch.transpose(pts_3d, 0, 1)
    ones_array = torch.ones([1, pts_3d.shape[1]], dtype=torch.float).to(varargs['device'])
    pts_3d = torch.cat([pts_3d, ones_array], dim=0)

    voxel_idxs = torch.cartesian_prod(xx_idx, yy_idx, zz_idx).to(varargs['device'])
    voxel_idxs = torch.transpose(voxel_idxs, 0, 1)

    voxel_dims = (zz_idx.shape[0], yy_idx.shape[0], xx_idx.shape[0])  # D, H, W

    # Project the voxel coords into the pixel space
    rot = torch.eye(3, dtype=torch.float).to(varargs['device'])
    trans = torch.zeros([3, 1], dtype=torch.float).to(varargs['device'])
    extrinsics = torch.cat([rot, trans], dim=1)

    pts_2d = []
    voxel_idxs_list = []
    for intrinsics_i in intrinsics:
        pts_2d_i = torch.mm(intrinsics_i, torch.mm(extrinsics, pts_3d))
        pts_2d_i[0, :] = pts_2d_i[0, :] / pts_2d_i[2, :]
        pts_2d_i[1, :] = pts_2d_i[1, :] / pts_2d_i[2, :]

        # Remove all the points that lie outside the 2D feature space
        valid_mask = torch.ge(pts_2d_i[0, :], 0) * torch.ge(pts_2d_i[1, :], 0) * \
                     torch.lt(pts_2d_i[0, :], feat_2d_shape[3]) * torch.lt(pts_2d_i[1, :], feat_2d_shape[2])
        pts_2d_i = pts_2d_i[:, valid_mask]
        voxel_idxs_i = voxel_idxs[:, valid_mask]

        pts_2d_i = pts_2d_i[:2, :]
        pts_2d.append(pts_2d_i)
        voxel_idxs_list.append(voxel_idxs_i)

    return pts_2d, voxel_idxs_list, voxel_dims


def convert_voxel_to_frustum(intrinsics, image_extents, resolution, **varargs):
    # Generate the coordinates of the frustum
    x1, x2, y1, y2, z1, z2 = image_extents
    zz, yy, xx = torch.arange(z1, z2, resolution, dtype=torch.float).to(varargs['device']), \
                 torch.arange(y1, y2, 1, dtype=torch.float).to(varargs["device"]), \
                 torch.arange(x1, x2, 1, dtype=torch.float).to(varargs['device'])

    zz_idx, yy_idx, xx_idx = torch.arange(0, zz.shape[0], 1, dtype=torch.long).to(varargs['device']), \
                             torch.arange(0, yy.shape[0], 1, dtype=torch.long).to(varargs['device']), \
                             torch.arange(0, xx.shape[0], 1, dtype=torch.long).to(varargs['device'])

    voxel_idxs = torch.cartesian_prod(xx_idx, yy_idx, zz_idx).to(varargs['device'])
    voxel_idxs = torch.transpose(voxel_idxs, 0, 1)

    pts_3d = []
    voxel_idxs_list = []
    for intrinsics_i in intrinsics:
        pts_3d_i = torch.cartesian_prod(xx, yy, zz).to(varargs["device"])
        pts_3d_i = torch.transpose(pts_3d_i, 0, 1)
        ones_array = torch.ones([1, pts_3d_i.shape[1]], dtype=torch.float).to(varargs['device'])
        pts_3d_i = torch.cat([pts_3d_i, ones_array], dim=0)

        # Make the voxel into a frustum
        pts_3d_i[0, :] = (pts_3d_i[0, :] - intrinsics_i[0][2]) / intrinsics_i[0][0]
        pts_3d_i[1, :] = (pts_3d_i[1, :] - intrinsics_i[1][2]) / intrinsics_i[1][1]
        pts_3d_i[:2] *= pts_3d_i[2]


        pts_3d_i = pts_3d_i[:3, :] / resolution # Fixme: Not sure about this yet, image coords vs voxel coords

        valid_mask = torch.all(torch.ge(pts_3d_i, 0), dim=0)
        valid_mask = (valid_mask *
                      torch.lt(pts_3d_i[0], xx.shape[0]) *
                      torch.lt(pts_3d_i[1], yy.shape[0]) *
                      torch.lt(pts_3d_i[2], zz.shape[0]))

        pts_3d_i = pts_3d_i[:, valid_mask]
        voxel_idxs_i = voxel_idxs[:, valid_mask]

        pts_3d.append(pts_3d_i)
        voxel_idxs_list.append(voxel_idxs_i)

    return pts_3d, voxel_idxs_list