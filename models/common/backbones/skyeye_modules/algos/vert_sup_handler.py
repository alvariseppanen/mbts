import torch
import torch.nn.functional as F
from po_bev_unsupervised.algos.voxel_grid import VoxelGridAlgo
from po_bev_unsupervised.algos.coordinate_transform import _FvToPointcloud, CoordinateWarper,


def transform_depth_to_frame(idx_frame_src, idx_frame_tgt, ego_pose, fv_intrinsics, depth_pred_src, is_img=True):
    # Get batch size
    batch_size = depth_pred_src.shape[0]

    # Compute pose difference (reverted here as we go from idx_curr+offset to idx_curr
    T = VoxelGridAlgo.compute_relative_transformation(ego_pose[idx_frame_tgt], ego_pose[idx_frame_src])

    # Transform T to homogeneous coordinates by extending it by an additional row of 0, 0, 0, 1
    hom_row = torch.tensor([0, 0, 0, 1]).unsqueeze(0)
    hom_row = hom_row.repeat(batch_size, 1, 1).to(depth_pred_src.device)
    T = torch.cat([T, hom_row], 1)

    # Transform FV image into camera coordinate system using the depth map
    fv_to_pointcloud = _FvToPointcloud()
    fv_as_pointcloud = depth_pred_src.detach()
    if is_img:
        fv_as_pointcloud = fv_to_pointcloud(depth_pred_src.detach(), fv_intrinsics)
    coordinate_warper = CoordinateWarper()
    transformed_pointcloud = coordinate_warper(fv_as_pointcloud, T)

    return transformed_pointcloud

def reject_depth_outliers(accumulated_depth, label_imgs, pointcloud_to_fv, idx_curr, ego_pose, fv_intrinsics, depth_window):
    for i in range(len(accumulated_depth)):
        depth_i = accumulated_depth[i].detach()
        score_map_i = torch.zeros(depth_i.shape, device=depth_i.device)
        for j in range(len(label_imgs)):
            label_img_j = label_imgs[j]

            depth_warped = transform_depth_to_frame(idx_curr, idx_curr + j, ego_pose, fv_intrinsics[idx_curr + j], depth_i, is_img=False)
            pixel_coordinates = pointcloud_to_fv(depth_warped, fv_intrinsics[idx_curr + j], label_img_j.shape[-2:])

            # ToDo: Car label is hard-coded here. VERY BAD!
            score_j = (F.grid_sample(label_img_j.float(), pixel_coordinates, mode="nearest", padding_mode="zeros") == 7).int()
            score_map_i += score_j.repeat(1, 3, 1, 1)
        accumulated_depth[i][score_map_i < depth_window-1] = 255

    return accumulated_depth




