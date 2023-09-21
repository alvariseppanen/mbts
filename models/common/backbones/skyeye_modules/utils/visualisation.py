import matplotlib as mpl
# matplotlib.use("Agg")
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import random
import seaborn as sns
import PIL.Image as pil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mppatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import align_tensors
from po_bev_unsupervised.utils.sequence import pad_packed_images
from po_bev_unsupervised.utils.kitti_labels_merged_bev import labels as kitti_labels_bev
from po_bev_unsupervised.utils.kitti_labels_merged_front import labels as kitti_labels_fv
from po_bev_unsupervised.utils.nuscenes_labels_mc import labels as nuscenes_labels
from po_bev_unsupervised.utils.waymo_labels_merged_bev import labels as waymo_labels_bev
from po_bev_unsupervised.utils.waymo_labels_merged_front import labels as waymo_labels_fv
from po_bev_unsupervised.utils.logging import accumulate_wandb_images
from po_bev_unsupervised.utils.panoptic import make_semantic_gt_list

THING_COLOURS = [(56, 60, 200), (168, 240, 104), (24, 20, 140), (41, 102, 116),
                 (101, 224, 99), (51, 39, 96), (5, 72, 78), (4, 236, 154)]


def generate_visualisations(sample, results, idxs, wandb_vis_dict, **varargs):
    vis_dict = {}

    # Generate the semantic mask
    if "fv_sem_pred" in results.keys():
        fv_sem_gt_list = make_semantic_gt_list(sample['fv_msk'], sample['fv_cat'])
        vis_dict['fv_sem'] = visualise_fv_semantic(sample['img'], fv_sem_gt_list, results['fv_sem_pred'],
                                                fvsem_window_size=varargs['fvsem_window_size'],
                                                fvsem_step_size=varargs['fvsem_step_size'],
                                                scale=0.25 / varargs['img_scale'], num_stuff=varargs['fv_num_stuff'],
                                                dataset=varargs['dataset'], rgb_mean=varargs['rgb_mean'],
                                                rgb_std=varargs['rgb_std'])

    if "bev_sem_pred" in results.keys():
        bev_sem_gt_list = make_semantic_gt_list(sample['bev_msk'], sample['bev_cat'])
        bev_combined_supervision = results['bev_combined_supervision'] if 'bev_combined_supervision' in results.keys() else None
        vis_dict['bev_sem'] = visualise_bev_semantic(sample['img'], bev_sem_gt_list, results['bev_sem_pred'],
                                                 bev_combined_supervision,
                                                 fvsem_window_size=varargs['fvsem_window_size'],
                                                 scale=0.25 / varargs['img_scale'], num_stuff=varargs['bev_num_stuff'],
                                                 dataset=varargs['dataset'], rgb_mean=varargs['rgb_mean'],
                                                 rgb_std=varargs['rgb_std'])

    if "fv_rgb_pred" in results.keys():
        vis_dict['fv_rgb_pred'] = visualise_rgb_reconsruction(sample['img'], results['fv_rgb_pred'],
                                                              fvsem_window_size=varargs['fvsem_window_size'],
                                                              fvsem_step_size=varargs['fvsem_step_size'],
                                                              scale=0.25 / varargs['img_scale'], num_stuff=varargs['fv_num_stuff'],
                                                              rgb_mean=varargs['rgb_mean'],
                                                              rgb_std=varargs['rgb_std'])


    if "fv_depth_pred" in results.keys():
        vis_dict['fv_depth_pred'] = visualise_depth_pred(sample['img'], results['fv_depth_pred'],
                                                         fvsem_window_size=varargs['fvsem_window_size'],
                                                         scale=0.25 / varargs['img_scale'], num_stuff=varargs['fv_num_stuff'],
                                                         dataset=varargs['dataset'],
                                                         rgb_mean=varargs['rgb_mean'],
                                                         rgb_std=varargs['rgb_std'])

    if "bev_ht_pred" in results.keys():
        bev_sem_gt_list = make_semantic_gt_list(sample['bev_msk'], sample['bev_cat'])
        vis_dict['bev_ht'] = visualise_bev_heights(bev_sem_gt_list, results['bev_ht_pred'],
                                                   fvsem_window_size=varargs['fvsem_window_size'],
                                                   scale=0.25 / varargs['img_scale'], dataset=varargs['dataset'],
                                                   min_ht=1.3, max_ht=1.8)



    # Accumulate the images
    dataset_labels = get_labels(varargs['dataset'], bev=False)
    fv_sem_class_labels = {label.trainId: label.name for label in dataset_labels if label.trainId >= 0 and label.trainId != 255}
    wandb_vis_dict = accumulate_wandb_images("fv_sem", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'],
                                                 fv_sem_class_labels=fv_sem_class_labels)
    wandb_vis_dict = accumulate_wandb_images("fv_rgb_pred", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'])
    wandb_vis_dict = accumulate_wandb_images("fv_depth_pred", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'])
    wandb_vis_dict = accumulate_wandb_images("bev_sem", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'])
    wandb_vis_dict = accumulate_wandb_images("bev_ht", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'])

    return wandb_vis_dict


def visualise_fv_semantic(img, sem_gt, sem_pred, scale=0.5, **varargs):
    vis_list = []
    fvsem_window_size = varargs['fvsem_window_size']
    fvsem_step_size = varargs['fvsem_step_size']

    if varargs['dataset'] == "Kitti360":
        idx_curr = len(sem_gt) // 2
    elif varargs['dataset'] == "Waymo":
        idx_curr = 0

    for b in range(len(sem_gt[0])):
        # Appending all the images next to each other
        img_b = []
        sem_pred_b = []
        sem_gt_b = []

        for ts_idx, ts in enumerate(range(idx_curr, min(len(img), idx_curr + fvsem_window_size + 1), fvsem_step_size)):
            img_i = pad_packed_images(img[ts])[0][b]
            sem_pred_i = pad_packed_images(sem_pred[ts_idx])[0][b]
            sem_gt_i =  sem_gt[ts][b]

            # Scale the images based on the scale
            img_i_scaled = F.interpolate(img_i.unsqueeze(0), scale_factor=scale, mode="bilinear", recompute_scale_factor=True).squeeze(0)
            sem_pred_i_scaled = F.interpolate(sem_pred_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)
            sem_gt_i_scaled = F.interpolate(sem_gt_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)

            # Restore the RGB image and generate the RGB images for the semantic mask
            img_i_scaled = (recover_image(img_i_scaled, varargs["rgb_mean"], varargs["rgb_std"]) * 255).type(torch.int)
            # sem_pred_i_scaled = visualise_semantic_mask_train_id(sem_pred_i_scaled, varargs['dataset'], bev=False)
            # sem_gt_i_scaled = visualise_semantic_mask_train_id(sem_gt_i_scaled, varargs['dataset'], bev=False)

            # Append all the RGB images and masks next to each other
            img_b.append(img_i_scaled)
            sem_pred_b.append(sem_pred_i_scaled)
            sem_gt_b.append(sem_gt_i_scaled)

        # Generate a long appended image
        img_b = torch.cat(img_b, dim=2)
        sem_pred_b = torch.cat(sem_pred_b, dim=2)
        sem_gt_b = torch.cat(sem_gt_b, dim=2)

        vis_dict_b = {"img": img_b, "sem_pred": sem_pred_b, "sem_gt": sem_gt_b}
        vis_list.append(vis_dict_b)

    return vis_list

def visualise_bev_semantic(img, sem_gt, sem_pred, supervision, scale=0.5, frontal_view=False, **varargs):
    vis_list = []
    fvsem_window_size = varargs['fvsem_window_size']

    if varargs['dataset'] == "Kitti360":
        idx_curr = len(sem_gt) // 2
    elif varargs['dataset'] == "Waymo":
        idx_curr = 0

    for b in range(len(sem_gt[0])):
        ts = idx_curr

        # This is for the BEV
        vis_b = []
        img_i = pad_packed_images(img[ts])[0][b]
        sem_pred_i = pad_packed_images(sem_pred[0])[0][b]
        sem_gt_i = sem_gt[ts][b]
        if supervision is not None:
            supervision_i = supervision[b]

        # Rotate the prediction to align it with the GT
        # sem_pred_i = torch.rot90(sem_pred_i, k=3, dims=[0, 1])
        # if supervision is not None:
        #     supervision_i = torch.rot90(supervision_i, k=1, dims=[0, 1])

        # Scale the images based on the scale
        img_i_scaled = F.interpolate(img_i.unsqueeze(0), scale_factor=scale, mode="bilinear", recompute_scale_factor=True).squeeze(0)
        sem_pred_i_scaled = F.interpolate(sem_pred_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)
        sem_gt_i_scaled = F.interpolate(sem_gt_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)
        if supervision is not None:
            supervision_i_scaled = F.interpolate(supervision_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)

        # Get the masked BEV prediction
        sem_pred_masked_i_scaled = sem_pred_i_scaled.clone()
        sem_pred_masked_i_scaled[sem_gt_i_scaled == 255] = 255

        # Restore the RGB image
        img_i_scaled = (recover_image(img_i_scaled, varargs['rgb_mean'], varargs['rgb_std']) * 255).type(torch.int)
        # Rotate the prediction by 90 degrees to align it with the GT
        sem_pred_i_scaled = visualise_semantic_mask_train_id(sem_pred_i_scaled, varargs['dataset'], bev=True)
        sem_gt_i_scaled = visualise_semantic_mask_train_id(sem_gt_i_scaled, varargs['dataset'], bev=True)
        sem_pred_masked_i_scaled = visualise_semantic_mask_train_id(sem_pred_masked_i_scaled, varargs['dataset'], bev=True)
        if supervision is not None:
            supervision_i_scaled = visualise_semantic_mask_train_id(supervision_i_scaled, varargs['dataset'], bev=True)

        # Align the images properly
        if varargs['dataset'] == "Kitti360":
            vis_row1 = torch.cat([sem_pred_i_scaled, sem_gt_i_scaled], dim=2)
            if supervision is not None:
                vis_row2 = torch.cat([sem_pred_masked_i_scaled, supervision_i_scaled], dim=2)

            if supervision is not None:
                vis_ts = torch.cat([img_i_scaled, vis_row1, vis_row2], dim=1)
            else:
                vis_ts = torch.cat([img_i_scaled, vis_row1], dim=1)
            vis_b.append(vis_ts)

        elif varargs['dataset'] == "Waymo":
            vis_ts = torch.cat([img_i_scaled, sem_pred_i_scaled, sem_gt_i_scaled], dim=1)
            vis_b.append(vis_ts)

        vis_b = torch.cat(vis_b, dim=2)
        vis_dict_b = {'bev_sem': vis_b}
        vis_list.append(vis_dict_b)

    return vis_list


def visualise_rgb_reconsruction(img_gt, img_pred, scale, **varargs):
    vis_list = []
    fvsem_window_size = varargs['fvsem_window_size']
    fvsem_step_size = varargs['fvsem_step_size']
    idx_curr = len(img_gt) // 2

    for b in range(len(img_gt[0])):
        # Appending all the images next to each other
        # img_gt_b = []
        img_pred_b = []
        for ts_idx, ts in enumerate(range(idx_curr, idx_curr + fvsem_window_size + 1, fvsem_step_size)):
            # img_gt_i = pad_packed_images(img_gt[ts])[0][b]
            img_pred_i = img_pred[ts_idx][b]

            # Scale the images based on the scale
            # img_gt_i_scaled = F.interpolate(img_gt_i.unsqueeze(0), scale_factor=scale, mode="bilinear", recompute_scale_factor=True).squeeze(0)
            img_pred_i_scaled = F.interpolate(img_pred_i.unsqueeze(0), scale_factor=scale, mode="bilinear", recompute_scale_factor=True).squeeze(0)


            # Restore the RGB image and generate the RGB images for the semantic mask
            img_pred_i_scaled = (recover_image(img_pred_i_scaled, varargs["rgb_mean"], varargs["rgb_std"]) * 255).type(torch.int)

            # Append all the RGB images and masks next to each other

            img_pred_b.append(img_pred_i_scaled)

        # Generate a long appended image
        img_pred_b = torch.cat(img_pred_b, dim=2)

        vis_dict_b = {"fv_rgb_pred": img_pred_b}
        vis_list.append(vis_dict_b)

    return vis_list


def visualise_depth_pred(img_gt, depth_pred, scale, **varargs):
    vis_list = []
    fvsem_window_size = varargs['fvsem_window_size']

    if varargs['dataset'] == "Kitti360":
        idx_curr = len(img_gt) // 2
    elif varargs['dataset'] == "Waymo":
        idx_curr = 0

    img_gt_i = pad_packed_images(img_gt[idx_curr])[0]
    for b in range(len(img_gt[0])):
        # Appending all the images next to each other
        depth_pred_b = depth_pred[0][b]
        img_gt_b = img_gt_i[b]

        # Scale the images based on the scale
        img_gt_b_scaled = F.interpolate(img_gt_b.unsqueeze(0), scale_factor=scale, mode="bilinear", recompute_scale_factor=True).squeeze(0)
        depth_pred_b_scaled = F.interpolate(depth_pred_b.unsqueeze(0), scale_factor=scale, mode="bilinear", recompute_scale_factor=True).squeeze(0)

        # Restore the RGB image and generate the RGB images for the semantic mask
        img_gt_b_scaled = (recover_image(img_gt_b_scaled, varargs["rgb_mean"], varargs["rgb_std"]) * 255).type(torch.int)

        # Apply colormap on the depth image
        disp = (1 / depth_pred_b_scaled).permute(1, 2, 0).cpu().numpy()
        vmax = np.percentile(disp, 95)
        normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
        mapper = plt.cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_img = (mapper.to_rgba(disp.squeeze(2))[:, :, :3] * 255).astype(np.uint8)
        depth_pred_visu = torch.from_numpy(colormapped_img).type(torch.int).permute(2, 0, 1).to(img_gt_b_scaled.device)

        vis_vertical = torch.cat([img_gt_b_scaled, depth_pred_visu], dim=2)
        vis_dict_b = {'fv_depth_pred': vis_vertical}
        vis_list.append(vis_dict_b)

    return vis_list


def visualise_bev_heights(bev_gt, ht_pred, scale=0.5, **varargs):
    vis_list = []
    fvsem_window_size = varargs['fvsem_window_size']
    idx_curr = len(bev_gt) // 2

    # ht_pred, _ = pad_packed_images(ht_pred)
    for b in range(len(bev_gt[0])):
        # This is for the BEV
        vis_b = []

        ht_pred_i = ht_pred[b]
        bev_gt_i = bev_gt[idx_curr][b]

        # Scale the images based on the scale
        ht_pred_i_scaled = F.interpolate(ht_pred_i.unsqueeze(0).type(torch.float), scale_factor=scale, mode="bilinear", recompute_scale_factor=True).type(torch.int).squeeze(0)
        bev_gt_i_scaled = F.interpolate(bev_gt_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)

        # Rotate the prediction by 90 degrees to align it with the GT
        ht_pred_i_scaled = torch.rot90(ht_pred_i_scaled, k=1, dims=[1, 2])
        ht_pred_i_scaled = (ht_pred_i_scaled - varargs['min_ht']) / (varargs['max_ht'] - varargs['min_ht'])
        ht_pred_i_scaled = torch.cat([ht_pred_i_scaled * 255] * 3, dim=0)
        bev_gt_i_scaled = visualise_semantic_mask_train_id(bev_gt_i_scaled, varargs['dataset'], bev=True)

        # Align the images properly
        vis_vertical = torch.cat([ht_pred_i_scaled, bev_gt_i_scaled], dim=2)
        vis_dict_b = {'bev_ht': vis_vertical}
        vis_list.append(vis_dict_b)

    return vis_list


def visualiseBEV(img, bev_gt, bev_pred, bev_pred_with_invalid, scale=0.5, semantic=False, **varargs):
    vis_list = []

    img_unpack, _ = pad_packed_images(img)
    if semantic:
        bev_pred, _ = pad_packed_images(bev_pred)
        bev_pred_with_invalid, _ = pad_packed_images(bev_pred_with_invalid)
    else:
        if "sem_pred" in varargs.keys():
            bev_pred_sem, _ = pad_packed_images(varargs['sem_pred'])
        else:
            bev_pred_sem = None

    for b in range(len(bev_gt)):
        vis = []
        if semantic:
            bev_gt_unpack = bev_gt[b]
            bev_pred_unpack = bev_pred[b]
            # bev_pred_wi_unpack = bev_pred_with_invalid[b]
        else:
            bev_gt_unpack = getPOMask(bev_gt[b], varargs['remap'], varargs['num_stuff'])
            bev_pred_unpack = getPOMask(bev_pred[b]['po_pred'], varargs['remap'], varargs['num_stuff'])
            bev_pred_sem_unpack = bev_pred_sem[b] if bev_pred_sem is not None else None

        img_count = img_unpack.shape[1] // 3

        img_small = [F.interpolate(img_unpack[b, 3*i:(3*i)+3, :, :].unsqueeze(0), scale_factor=scale, mode="bilinear").squeeze(0) for i in range(img_count)]
        bev_gt_small = F.interpolate(bev_gt_unpack.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor), scale_factor=scale, mode="nearest").type(torch.IntTensor).squeeze(0)
        bev_pred_small = F.interpolate(bev_pred_unpack.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor), scale_factor=scale, mode="nearest").type(torch.IntTensor).squeeze(0)

        if not semantic:
            bev_pred_sem_small = F.interpolate(bev_pred_sem_unpack.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor),
                                           scale_factor=scale, mode="nearest").type(torch.IntTensor).squeeze(0)
        else:
            bev_pred_sem_small = None

        # bev_pred_wi_small = F.interpolate(bev_pred_wi_unpack.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor), scale_factor=scale, mode="nearest").type(torch.IntTensor).squeeze(0)

        # Visualise BEV as RGB
        for img in img_small:
            vis.append((recoverImage(img, varargs["rgb_mean"], varargs["rgb_std"]) * 255).type(torch.IntTensor))

        if bev_gt_small.shape[2] < img_small[0].shape[2]:
            vis_horizontal = []
            if semantic:
                vis_bev_pred = visualiseSemanticMaskTrainId(bev_pred_small, varargs['dataset'])
                vis_bev_gt = visualiseSemanticMaskTrainId(bev_gt_small, varargs['dataset'])
                # vis_bev_pred_wi = visualiseSemanticMaskTrainId(bev_pred_wi_small, varargs['dataset'])
            else:
                vis_bev_pred = visualisePanopticMaskTrainId(bev_pred_small, varargs['dataset'])
                vis_bev_gt = visualisePanopticMaskTrainId(bev_gt_small, varargs['dataset'])

                if bev_pred_sem_small is not None:
                    vis_bev_pred_sem = visualiseSemanticMaskTrainId(bev_pred_sem_small, varargs['dataset'])
                else:
                    vis_bev_pred_sem = None
                # vis_bev_pred_wi = None

            # Add the error map and the masked output
            vis_bev_pred_masked = vis_bev_pred.clone()
            vis_bev_pred_masked[:, bev_gt_small.squeeze(0) == 255] = 0  # Set invalid areas to 0

            error_map = torch.zeros_like(vis_bev_pred_masked)
            if semantic:
                error_region = (bev_pred_small != bev_gt_small).squeeze(0)
            else:
                bev_pred_small_sem = bev_pred_small.clone()
                bev_gt_small_sem = bev_gt_small.clone()
                bev_pred_small_sem[bev_pred_small_sem > 1000] = bev_pred_small_sem[bev_pred_small_sem > 1000] // 1000
                bev_gt_small_sem[bev_gt_small_sem > 1000] = bev_gt_small_sem[bev_gt_small_sem > 1000] // 1000
                error_region = (bev_gt_small_sem != bev_pred_small_sem).squeeze(0)
            error_map[:, error_region] = 255
            error_map[:, bev_gt_small.squeeze(0) == 255] = 0

            if semantic:
                # Row 1 --> GT and Pred
                vis.append(torch.cat([vis_bev_gt, vis_bev_pred], dim=2))
                # Row 2 --> Masked pred and Error map
                vis.append(torch.cat([vis_bev_pred_masked, error_map], dim=2))
            else:
                # Get the bbox projection on the semantic prediction as well
                bbx_pred = varargs['bbx_pred'][b]
                bbx_pred = bbx_pred.unsqueeze(0) if bbx_pred is not None else [None]
                vis_bbx = visualiseBBoxes(bev_pred_sem_small, bbx_pred, scale=scale, dataset=varargs['dataset'])[0]
                vis_bbx = torch.from_numpy(vis_bbx).permute(2, 0, 1)

                # Row 1 --> GT and Masked Pred
                vis.append(torch.cat([vis_bev_gt, vis_bev_pred_masked], dim=2))
                # Row 2 --> Error map and BBox
                vis.append(torch.cat([vis_bbx, error_map], dim=2))

            # Add the with invalid and error output
            # if vis_bev_pred_wi is not None:
            #     error_map = torch.zeros_like(vis_bev_pred_wi)
            #     error_region = (bev_pred_wi_small != bev_gt_small).squeeze(0)
            #     error_map[:, error_region] = 255
            #     error_map[:, bev_gt_small.squeeze(0) == 255] = 0
            #     vis.append(torch.cat([vis_bev_pred_wi, error_map], dim=2))

        else:
            if semantic:
                vis_bev_pred = visualiseSemanticMaskTrainId(bev_pred_small, varargs['dataset'])
                vis_bev_gt = visualiseSemanticMaskTrainId(bev_gt_small, varargs['dataset'])
                vis.append(vis_bev_gt)
                vis.append(vis_bev_pred)
                error_region = (bev_pred_small != bev_gt_small).squeeze(0)
            else:
                vis_bev_pred = visualisePanopticMaskTrainId(bev_pred_small, varargs['dataset'])
                vis_bev_gt = visualisePanopticMaskTrainId(bev_gt_small, varargs['dataset'])

                bbx_pred = varargs['bbx_pred'][b]
                bbx_pred = bbx_pred.unsqueeze(0) if bbx_pred is not None else [None]
                vis_bbx = visualiseBBoxes(bev_pred_sem_small, bbx_pred, scale=scale, dataset=varargs['dataset'])[0]
                vis_bbx = torch.from_numpy(vis_bbx).permute(2, 0, 1)

                vis.append(vis_bev_gt)
                vis.append(vis_bev_pred)
                vis.append(vis_bbx)

                bev_pred_small_sem = bev_pred_small.clone()
                bev_gt_small_sem = bev_gt_small.clone()
                bev_pred_small_sem[bev_pred_small_sem > 1000] = bev_pred_small_sem[bev_pred_small_sem > 1000] // 1000
                bev_gt_small_sem[bev_gt_small_sem > 1000] = bev_gt_small_sem[bev_gt_small_sem > 1000] // 1000
                error_region = (bev_gt_small_sem != bev_pred_small_sem).squeeze(0)

            vis_bev_pred_masked = vis_bev_pred.clone()
            vis_bev_pred_masked[:, bev_gt_small.squeeze(0) == 255] = 0  # Set invalid areas to 0
            vis.append(vis_bev_pred_masked)

            # Add the error map and the masked output
            error_map = torch.zeros_like(vis_bev_pred_masked)
            error_map[:, error_region] = 255
            error_map[:, bev_gt_small.squeeze(0) == 255] = 0
            vis.append(error_map)

        # Append all the images together
        vis = torch.cat(vis, dim=1)
        vis_list.append(vis)

    return vis_list


def visualiseInstanceMask(bbx_pred, cls_pred, msk_pred, bev_gt, scale=0.5, **varargs):
    vis_list = []
    for bbx_i, cls_i, msk_i, bev_gt_i in zip(bbx_pred, cls_pred, msk_pred, bev_gt):
        bev_gt_unpack_i = getPOMask(bev_gt_i, varargs['remap'], varargs['num_stuff']).cpu()
        bev_gt_unpack_i[(bev_gt_unpack_i < 1000) & (bev_gt_unpack_i != 255)] = 255
        vis_bev_gt_i = visualisePanopticMaskTrainId(bev_gt_unpack_i.unsqueeze(0), dataset=varargs['dataset'])

        bev_shape = bev_gt_unpack_i.shape

        # Place the instances in the correct position
        inst_mask_i = torch.zeros((bev_gt_unpack_i.shape[0], bev_gt_unpack_i.shape[1]), dtype=torch.long)
        if (bbx_i is None) or (cls_i is None) or (msk_i is None):
            pass
        else:
            for box_id in range(cls_i.shape[0]):
                ref_box = bbx_i[box_id, :].long()
                y_min = int(bbx_i[box_id][0])
                y_max = int(bbx_i[box_id][2])
                x_min = int(bbx_i[box_id][1])
                x_max = int(bbx_i[box_id][3])
                w = max((x_max - x_min + 1), 1)
                h = max((y_max - y_min + 1), 1)

                roi_edge = msk_i.shape[2]

                # msk_ch = cat_i[box_id] - self.num_stuff  # Select the channel from the mask logits based on the class prediction
                mask = F.upsample(msk_i[box_id, :, :].view(1, 1, roi_edge, roi_edge), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
                mask = (mask.sigmoid() > 0.5).type(torch.int) * (box_id + 1)  # Set the ID of the valid region to box_id
                x_min = max(ref_box[1], 0)
                x_max = min(ref_box[3] + 1, bev_shape[1])
                y_min = max(ref_box[0], 0)
                y_max = min(ref_box[2] + 1, bev_shape[0])

                inst_mask_i[y_min:y_max, x_min:x_max] = mask[0, (y_min - ref_box[0]):(y_max - ref_box[0]), (x_min - ref_box[1]):(x_max - ref_box[1])]

        # Colour the instance masks
        vis_inst_i = torch.zeros((3, inst_mask_i.shape[0], inst_mask_i.shape[1]), dtype=torch.int32)
        unique_inst_i = torch.unique(inst_mask_i)
        unique_inst_i = unique_inst_i[unique_inst_i > 0]
        for inst_id in unique_inst_i:
            region = (inst_mask_i == inst_id)
            vis_inst_i[:, region] = torch.IntTensor(random.choice(THING_COLOURS)).unsqueeze(1)

        # Resize the visualisations by a "scale" factor
        vis_bev_gt_i = F.interpolate(vis_bev_gt_i.unsqueeze(0).type(torch.FloatTensor), scale_factor=scale, mode="bilinear").type(torch.IntTensor).squeeze(0)
        vis_inst_i = F.interpolate(vis_inst_i.unsqueeze(0).type(torch.FloatTensor), scale_factor=scale, mode="bilinear").type(torch.IntTensor).squeeze(0)
        vis = torch.cat([vis_bev_gt_i, vis_inst_i], dim=2)
        vis_list.append(vis)

    return vis_list

def getPOMask(panoptic_pred, remap, num_stuff):
    canvas = torch.ones((panoptic_pred[0].shape)).type(torch.long).to(panoptic_pred[0].device) * 255
    thing_list = []
    for idd, pred in enumerate(list(panoptic_pred[1])):
        if pred == 255:
            continue
        if panoptic_pred[3][idd] == 0:
            # If not iscrowd
            if pred < num_stuff:
                canvas[panoptic_pred[0] == idd] = pred
            else:
                canvas[panoptic_pred[0] == idd] = pred * 1000 + thing_list.count(pred)
                thing_list.append(pred)
    return canvas


def recover_image(img, rgb_mean, rgb_std):
    img = img * img.new(rgb_std).view(-1, 1, 1)
    img = img + img.new(rgb_mean).view(-1, 1, 1)
    return img


def get_labels(dataset, bev=False):
    if bev:
        if dataset == "Kitti360":
            return kitti_labels_bev
        elif dataset == "nuScenes":
            return nuscenes_labels
        elif dataset == "Waymo":
            return waymo_labels_bev
    else:
        if dataset == "Kitti360":
            return kitti_labels_fv
        elif dataset == "nuScenes":
            return nuscenes_labels
        elif dataset == "Waymo":
            return waymo_labels_fv


def visualise_semantic_mask_train_id(sem_mask, dataset, bev=False):
    dataset_labels = get_labels(dataset, bev)
    STUFF_COLOURS_TRAINID = {label.trainId: label.color for label in dataset_labels}

    sem_vis = torch.zeros((3, sem_mask.shape[1], sem_mask.shape[2]), dtype=torch.int32).to(sem_mask.device)

    # Colour the stuff
    classes = torch.unique(sem_mask)
    for stuff_label in classes:
        sem_vis[:, (sem_mask == stuff_label).squeeze()] = torch.tensor(STUFF_COLOURS_TRAINID[stuff_label.item()],
                                                                       dtype=torch.int, device=sem_mask.device).unsqueeze(1)

    return sem_vis


def visualisePanopticMaskTrainId(bev_panoptic, dataset):
    if dataset == "Kitti360":
        STUFF_COLOURS_TRAINID = {label.trainId: label.color for label in cs_labels}
    elif dataset == "nuScenes":
        STUFF_COLOURS_TRAINID = {label.trainId: label.color for label in nuscenes_labels}

    po_vis = torch.zeros((3, bev_panoptic.shape[1], bev_panoptic.shape[2]), dtype=torch.int32).to(bev_panoptic.device)

    # Colour the stuff
    stuff_mask = bev_panoptic <= 1000
    classes = torch.unique(bev_panoptic[stuff_mask])
    for stuff_label in classes:
        po_vis[:, (bev_panoptic == stuff_label).squeeze()] = torch.tensor(STUFF_COLOURS_TRAINID[stuff_label.item()], dtype=torch.int32).unsqueeze(1).to(bev_panoptic.device)

    # Colour the things
    thing_mask = (bev_panoptic > 1000)
    if torch.sum(thing_mask) > 0:
        for thing_label in torch.unique(bev_panoptic[thing_mask]):
            po_vis[:, (bev_panoptic == thing_label).squeeze()] = torch.tensor(random.choice(THING_COLOURS), dtype=torch.int32).unsqueeze(1).to(bev_panoptic.device)

    return po_vis


def plotConfusionMatrix(conf_mat, num_classes, dataset, ignore_classes, bev=False):
    labels = get_labels(dataset, bev=bev)
    ignore_classes = [255, -1] + ignore_classes

    # Get the class names
    seen_ids = []
    class_labels = []
    for l in labels:
        if (l.trainId in seen_ids) or (l.trainId in ignore_classes):
            continue
        seen_ids.append(l.trainId)
        class_labels.append(l.name)

    # Get the important part of the confusion matrix
    conf_mat_np = conf_mat[:num_classes, :num_classes]

    # Get the ratio. Row elts + Col elts - Diagonal elt (it is computed twice)
    conf_mat_np = conf_mat_np / ((conf_mat_np.sum(dim=0) + conf_mat_np.sum(dim=1) - conf_mat_np.diag()) + 1e-8)  # Small number added to avoid nan
    conf_mat_np = conf_mat_np.cpu().detach().numpy()

    # Plot the confusion matrix
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    conf_mat_plt = sns.heatmap(conf_mat_np * 100, annot=True, fmt=".2g", vmin=0.0, vmax=100., square=True,
                               xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": 7}, ax=ax)

    return conf_mat_plt


def visualiseVFMask(img, vf_logits_list, vf_mask_gt, **varargs):
    img, _ = pad_packed_images(img)
    img = recoverImage(img, varargs['rgb_mean'], varargs["rgb_std"])
    vf_mask_gt, _ = pad_packed_images(vf_mask_gt)

    if varargs['dataset'] == 'Kitti360':
        B = vf_mask_gt.shape[0]
        H, W = vf_mask_gt.shape[2], vf_mask_gt.shape[3]
        vf_msk = torch.ones((B, 1, H, W), dtype=torch.long).to(vf_mask_gt.device) * 2
        sem_msk = vf_mask_gt.detach().clone()
        sem_msk[sem_msk >= 1000] = sem_msk[sem_msk >= 1000] // 1000

        for c in varargs['vertical_classes']:
            vf_msk[sem_msk == int(c)] = 0
        for c in varargs['flat_classes']:
            vf_msk[sem_msk == int(c)] = 1

    elif varargs['dataset'] == "nuScenes":
        vf_msk = vf_mask_gt

    # for b_idx in range(B):
    #     unique_labels = torch.unique(vf_mask_gt[b_idx])
    #     for label in unique_labels:
    #         if label >= 1000:
    #             stuff_label = label // 1000
    #         else:
    #             stuff_label = label
    #
    #         if stuff_label in varargs['vertical_classes']:
    #             vf_msk[b_idx, vf_mask_gt[b_idx] == label] = 0
    #         elif stuff_label in varargs['flat_classes']:
    #             vf_msk[b_idx, vf_mask_gt[b_idx] == label] = 1
    #         else:
    #             vf_msk[b_idx, vf_mask_gt[b_idx] == label] = 2

    labels = {0: "vertical", 1: "flat", 2: "other"}

    vf_vis_list = []

    cam_idx = 0
    for b_idx in range(img.shape[0]):
        vf_batch_list = []

        for vf_logits_scale in vf_logits_list:
            scale = vf_logits_scale.shape[2] / vf_mask_gt.shape[2]
            vf_gt_scale = F.interpolate(vf_msk.type(torch.float), scale_factor=scale, mode="nearest").squeeze(1).type(torch.long)
            vf_pred_scale = torch.argmax(vf_logits_scale, dim=1)
            img_scale = F.interpolate(img[b_idx].unsqueeze(0), scale_factor=scale, mode="bilinear").squeeze(0)

            vf_batch_list.append([img_scale, vf_pred_scale[b_idx, :, :], vf_gt_scale[b_idx, :, :], labels])
            break

        vf_vis_list.append(vf_batch_list)

            # vf_img = wandb.Image(img, masks={
            #     "predictions": {
            #         "mask_data": vf_pred_scale,
            #         "class_labels": labels
            #     },
            #     "ground_truth": {
            #         "mask_data": vf_gt_scale,
            #         "class_labels": labels
            #     }
            # })

            # vf_vis_list.append(vf_img)

    return vf_vis_list


def visualiseVRegionMask(bev_gt, v_region_logits_list, **varargs):
    v_region_vis_list = []
    v_region_batch_list = []
    labels = {0: "vertical", 1: "other"}

    for b in range(len(bev_gt)):
        H, W = bev_gt[b].shape[0], bev_gt[b].shape[1]
        v_region_msk = torch.zeros((H, W), dtype=torch.long).to(bev_gt[0].device)

        for c in varargs['vertical_classes']:
            v_region_msk[bev_gt[b] == c] = 1
        # for c in varargs['flat_classes']:
        #     v_region_msk[bev_gt[b] == c] = 1

        vis_bev_gt = visualiseSemanticMaskTrainId(bev_gt[b].unsqueeze(0).cpu(), varargs['dataset'])

        for v_region_logits_scale in v_region_logits_list:
            scale = v_region_logits_scale.shape[2] / bev_gt[b].shape[0]
            v_region_gt_scale = F.interpolate(v_region_msk.type(torch.float).unsqueeze(0).unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(1).type(torch.long).squeeze(0)
            v_region_pred_scale = (torch.sigmoid(v_region_logits_scale[b]))

            # Plot the GT and the prediction
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(v_region_pred_scale.squeeze().cpu().numpy(), cmap='jet')
            ax.imshow(v_region_gt_scale.cpu().numpy(), cmap="gray", vmin=0, vmax=1, alpha=0.5)

            fig.canvas.draw()
            # buf = canvas.buffer_rgba()
            vis_numpy = np.asarray(fig.canvas.renderer._renderer)
            vis_vregion = torch.from_numpy(vis_numpy).permute(2, 0, 1)
            v_region_batch_list.append(vis_vregion)
            plt.close(fig)
            plt.clf()

            # vis_bev_gt_scale = F.interpolate(vis_bev_gt.unsqueeze(0).type(torch.float), scale_factor=scale, mode="bilinear").squeeze(0).type(torch.long)

            # v_region_batch_list.append([vis_bev_gt_scale, v_region_pred_scale, v_region_gt_scale, labels])
            break

    v_region_vis_list.append(v_region_batch_list)
    return v_region_vis_list

def visualiseFRegionMask(bev_gt, f_region_logits_list, **varargs):
    f_region_vis_list = []
    f_region_batch_list = []
    labels = {0: "flat", 1: "other"}

    for b in range(len(bev_gt)):
        H, W = bev_gt[b].shape[0], bev_gt[b].shape[1]
        f_region_msk = torch.zeros((H, W), dtype=torch.long).to(bev_gt[0].device)

        for c in varargs['flat_classes']:
            f_region_msk[bev_gt[b] == c] = 1
        # for c in varargs['flat_classes']:
        #     v_region_msk[bev_gt[b] == c] = 1

        vis_bev_gt = visualiseSemanticMaskTrainId(bev_gt[b].unsqueeze(0).cpu(), varargs['dataset'])

        for f_region_logits_scale in f_region_logits_list:
            scale = f_region_logits_scale.shape[2] / bev_gt[b].shape[0]
            f_region_gt_scale = F.interpolate(f_region_msk.type(torch.float).unsqueeze(0).unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(1).type(torch.long).squeeze(0)
            f_region_pred_scale = (torch.sigmoid(f_region_logits_scale[b]))

            # Plot the GT and the prediction
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(f_region_pred_scale.squeeze().cpu().numpy(), cmap='jet')
            ax.imshow(f_region_gt_scale.cpu().numpy(), cmap="gray", vmin=0, vmax=1, alpha=0.5)

            fig.canvas.draw()
            # buf = canvas.buffer_rgba()
            vis_numpy = np.asarray(fig.canvas.renderer._renderer)
            vis_vregion = torch.from_numpy(vis_numpy).permute(2, 0, 1)
            f_region_batch_list.append(vis_vregion)
            plt.close(fig)
            plt.clf()

            # vis_bev_gt_scale = F.interpolate(vis_bev_gt.unsqueeze(0).type(torch.float), scale_factor=scale, mode="bilinear").squeeze(0).type(torch.long)

            # v_region_batch_list.append([vis_bev_gt_scale, v_region_pred_scale, v_region_gt_scale, labels])
            break

    f_region_vis_list.append(f_region_batch_list)
    return f_region_vis_list

def visualiseSpatialUncertainty(bev_gt, spatial_uncertainty, **varargs):
    uncert_vis_list = []

    for b in range(len(bev_gt)):
        H, W = bev_gt[b].shape[0], bev_gt[b].shape[1]

        vis_bev_gt = visualiseSemanticMaskTrainId(bev_gt[b].unsqueeze(0).cpu(), varargs['dataset']).squeeze().permute(1, 2, 0)
        spat_uncert = spatial_uncertainty[b] ** 2
        spat_uncert = spat_uncert.clamp(0.1, 10)  # To make all the values positive

        # Plot the GT and the prediction
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(vis_bev_gt.cpu().numpy())
        ax.imshow(spat_uncert.permute(1, 2, 0).cpu().numpy(), cmap="jet", alpha=0.6, vmin=0, vmax=10)

        fig.canvas.draw()
        # buf = canvas.buffer_rgba()
        vis_numpy = np.asarray(fig.canvas.renderer._renderer)
        vis_vregion = torch.from_numpy(vis_numpy).permute(2, 0, 1)
        uncert_vis_list.append(vis_vregion)
        plt.close(fig)
        plt.clf()

    return uncert_vis_list

def visualiseProposals(bev_mask, proposals, **varargs):
    for bev_mask_i, proposals_i in zip(bev_mask, proposals):
        bev_mask_i = bev_mask_i.cpu()
        bev_rgb_i = visualiseSemanticMaskTrainId(bev_mask_i, varargs['dataset'])
        bev_rgb_i = bev_rgb_i.permute(1, 2, 0).numpy()
        proposals_i = proposals_i.cpu().numpy()

        fig, ax = plt.subplots()
        ax.imshow(bev_rgb_i)

        p_idxs = list(range(len(proposals_i)))
        random.shuffle(p_idxs)
        count = 0
        for p_idx in p_idxs:
            if count > 20:
                break
            count += 1

            p = proposals_i[p_idx]
            min_row, min_col, max_row, max_col = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            height = max_row - min_row
            width = max_col - min_col
            rect = mppatches.Rectangle((min_col, min_row), width, height, linewidth=1, edgecolor='b', fill=False)
            ax.add_patch(rect)
        plt.show()


def visualiseBBoxes(bev_mask, bboxes, matplotlib=False, scale=1, **varargs):
    vis_list = []
    for bev_mask_i, bboxes_i in zip(bev_mask, bboxes):
        bev_mask_i = bev_mask_i.cpu().unsqueeze(0)
        bev_rgb_i = visualiseSemanticMaskTrainId(bev_mask_i, dataset=varargs['dataset'])
        bev_rgb_i = bev_rgb_i.permute(1, 2, 0).numpy().astype(np.uint8)
        if bboxes_i is None:
            vis_list.append(bev_rgb_i)
            continue
        bboxes_i = bboxes_i.cpu().detach().numpy()

        if matplotlib:
            fig, ax = plt.subplots()
            ax.imshow(bev_rgb_i)

        b_idxs = list(range(len(bboxes_i)))
        for p_idx in b_idxs:
            p = bboxes_i[p_idx]
            p = p * scale
            min_row, min_col, max_row, max_col = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            height = max_row - min_row
            width = max_col - min_col
            if matplotlib:
                rect = mppatches.Rectangle((min_col, min_row), width, height, linewidth=1, edgecolor='c', fill=False)
                ax.add_patch(rect)
            else:
                bev_rgb_i = cv2.rectangle(cv2.UMat(bev_rgb_i), (min_col, min_row), (max_col, max_row), (240, 255, 3), thickness=2)

        if matplotlib:
            plt.show()
        else:
            vis_list.append(bev_rgb_i.get())

    return vis_list


def saveIntermediateVisualisation(sample, sample_category, save_tuple):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    save_dir = os.path.join(save_path, sample_category)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_name = os.path.join(save_dir, "{}.png".format(sample_name))
    plt.imshow(sample, cmap="jet", vmin=0., vmax=1.0)
    plt.axis("off")
    plt.savefig(img_name, bbox_inches='tight', pad_inches=0)
    plt.clf()


def save_semantic_output(sample, sample_category, save_tuple, bev=False, woskyveg=False, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        save_dir_rgb = os.path.join(save_path, cam_name, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, cam_name, sample_category)
    else:
        save_dir_rgb = os.path.join(save_path, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, sample_category)

    if not os.path.exists(save_dir_rgb):
        os.makedirs(save_dir_rgb)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_name_rgb = [os.path.join(save_dir_rgb, "{}.png".format(sample_name_i)) for sample_name_i in sample_name]
    img_name = [os.path.join(save_dir, "{}.png".format(sample_name_i)) for sample_name_i in sample_name]

    # Generate the numpy image and save the image using OpenCV
    for idx, (sample_ts, img_name_ts, img_name_rgb_ts) in enumerate(zip(sample, img_name, img_name_rgb)):
        # Save the raw version of the mask
        sample_ts_orig = sample_ts.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if bev:
            sample_ts_orig = cv2.rotate(sample_ts_orig, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(img_name_ts, sample_ts_orig)

        # Save the RGB version of the mask
        sample_ts_rgb = visualise_semantic_mask_train_id(sample_ts, varargs['dataset'], bev=bev)
        sample_ts_rgb = sample_ts_rgb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        sample_ts_rgb = cv2.cvtColor(sample_ts_rgb, cv2.COLOR_RGB2BGR)
        if bev:
            sample_ts_rgb = cv2.rotate(sample_ts_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(img_name_rgb_ts, sample_ts_rgb)


def save_semantic_output_with_rgb_overlay(sample, rgb, sample_category, save_tuple, bev=False, woskyveg=False, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        save_dir_rgb = os.path.join(save_path, cam_name, "{}_rgb".format(sample_category))
        img_save_dir = os.path.join(save_path, "img_rgb")
        stacked_save_dir = os.path.join(save_path, "stacked_pred_rgb")
    else:
        save_dir_rgb = os.path.join(save_path, "{}_rgb".format(sample_category))
        img_save_dir = os.path.join(save_path, "img_rgb")
        stacked_save_dir = os.path.join(save_path, "stacked_pred_rgb")

    if not os.path.exists(save_dir_rgb):
        os.makedirs(save_dir_rgb)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(stacked_save_dir):
        os.makedirs(stacked_save_dir)

    # Generate the numpy image and save the image using OpenCV
    stacked_images = []
    for idx, (sample_ts, rgb_ts) in enumerate(zip(sample, rgb)):
        # Save the RGB version of the mask
        sample_ts_rgb = visualise_semantic_mask_train_id(sample_ts, varargs['dataset'], bev=bev)
        sample_ts_rgb = sample_ts_rgb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        sample_ts_rgb = cv2.cvtColor(sample_ts_rgb, cv2.COLOR_RGB2BGR)

        rgb_ts_recover = recover_image(rgb_ts, varargs['rgb_mean'], varargs["rgb_std"]).squeeze(0) * 255
        rgb_ts_recover = rgb_ts_recover.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rgb_ts_recover = cv2.cvtColor(rgb_ts_recover, cv2.COLOR_RGB2BGR)

        # Save the RGB image for the first timestep
        if idx == 0:
            img_save_name = os.path.join(img_save_dir, "{}.png".format(sample_name[0]))
            cv2.imwrite(img_save_name, rgb_ts_recover)
            stacked_images.append(cv2.resize(rgb_ts_recover, dsize=(0, 0), fx=0.25, fy=0.25))

        # Blend RGB and FV Pred
        sample_ts_overlay = cv2.addWeighted(sample_ts_rgb, 0.6, rgb_ts_recover, 0.4, 0.0)

        overlay_save_name = os.path.join(save_dir_rgb, "{}_{}.png".format(sample_name[0], idx))
        cv2.imwrite(overlay_save_name, sample_ts_overlay)
        stacked_images.append(cv2.resize(sample_ts_overlay, dsize=(0, 0), fx=0.25, fy=0.25))

    stacked_img = np.concatenate(stacked_images, axis=1)
    stacked_save_name = os.path.join(stacked_save_dir, "{}.png".format(sample_name[0]))
    cv2.imwrite(stacked_save_name, stacked_img)


def save_semantic_masked_output(sample_pred, sample_gt, sample_category, save_tuple, bev=False, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        save_dir_rgb = os.path.join(save_path, cam_name, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, cam_name, sample_category)
    else:
        save_dir_rgb = os.path.join(save_path, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, sample_category)

    if not os.path.exists(save_dir_rgb):
        os.makedirs(save_dir_rgb)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_name_rgb = [os.path.join(save_dir_rgb, "{}.png".format(sample_name_i)) for sample_name_i in sample_name]
    img_name = [os.path.join(save_dir, "{}.png".format(sample_name_i)) for sample_name_i in sample_name]

    # Generate the numpy image and save the image using OpenCV
    for idx, (sample_pred_ts, sample_gt_ts, img_name_ts, img_name_rgb_ts) in enumerate(zip(sample_pred, sample_gt, img_name, img_name_rgb)):
        # Mask the prediction using the GT
        sample_pred_ts[sample_gt_ts == 255] = 255

        # Save the raw version of the mask
        sample_ts_orig_masked = sample_pred_ts.permute(1, 2, 0).cpu().numpy().astype(np.uint16)
        sample_ts_orig_masked = cv2.rotate(sample_ts_orig_masked, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(img_name_ts, sample_ts_orig_masked)

        # Save the RGB version of the mask
        sample_ts_masked_rgb = visualise_semantic_mask_train_id(sample_pred_ts, varargs['dataset'], bev=bev)
        sample_ts_masked_rgb = sample_ts_masked_rgb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        sample_ts_masked_rgb = cv2.cvtColor(sample_ts_masked_rgb, cv2.COLOR_RGB2BGR)
        sample_ts_masked_rgb = cv2.rotate(sample_ts_masked_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(img_name_rgb_ts, sample_ts_masked_rgb)


def save_depth_output(sample, sample_category, save_tuple, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        save_dir = os.path.join(save_path, cam_name, sample_category)
    else:
        save_dir = os.path.join(save_path, sample_category)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_name = [os.path.join(save_dir, "{}.png".format(sample_name_i)) for sample_name_i in sample_name]

    # Generate the numpy image and save the image using OpenCV
    for idx, (sample_ts, img_name_ts) in enumerate(zip(sample, img_name)):
        # Apply colormap on the depth image
        disp = (1 / sample_ts).permute(1, 2, 0).cpu().numpy()
        vmax = np.percentile(disp, 95)
        normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
        mapper = plt.cm.ScalarMappable(norm=normalizer, cmap='magma')
        sample_vis = (mapper.to_rgba(disp.squeeze(2))[:, :, :3] * 255).astype(np.uint8)

        # Save the raw version of the mask
        sample_vis = cv2.cvtColor(sample_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_name_ts, sample_vis)




def savePanopticOutput(sample, sample_category, save_tuple, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        save_dir_rgb = os.path.join(save_path, cam_name, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, cam_name, sample_category)
    else:
        save_dir_rgb = os.path.join(save_path, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, sample_category)

    if not os.path.exists(save_dir_rgb):
        os.makedirs(save_dir_rgb)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_name_rgb = os.path.join(save_dir_rgb, "{}.png".format(sample_name))
    img_name = os.path.join(save_dir, "{}.png".format(sample_name))

    # Generate the numpy image and save the image using OpenCV
    # Check if there are multiple elements int he sample. Then you'll have to decode it using generatePOMask function
    if len(sample) > 1:
        po_mask = getPOMask(sample, None, varargs["num_stuff"]).unsqueeze(0)
    else:
        po_mask = sample[0].unsqueeze(0)

    # Save the raw version of the mask
    po_mask_orig = po_mask.permute(1, 2, 0).cpu().numpy().astype(np.uint16)
    cv2.imwrite(img_name, po_mask_orig)

    # Get the RGB image of the po_pred
    po_mask_rgb = visualisePanopticMaskTrainId(po_mask, varargs['dataset'])
    po_mask_rgb = po_mask_rgb.permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(img_name_rgb, po_mask_rgb)


def savePanopticOutputMasked(sample_pred, sample_gt, sample_category, save_tuple, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    save_dir_rgb = os.path.join(save_path, "{}_rgb".format(sample_category))
    save_dir = os.path.join(save_path, sample_category)

    if not os.path.exists(save_dir_rgb):
        os.makedirs(save_dir_rgb)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_name_rgb = os.path.join(save_dir_rgb, "{}.png".format(sample_name))
    img_name = os.path.join(save_dir, "{}.png".format(sample_name))

    # Generate the numpy image and save the image using OpenCV
    # Check if there are multiple elements int he sample. Then you'll have to decode it using generatePOMask function
    po_mask_pred = getPOMask(sample_pred, None, varargs["num_stuff"]).unsqueeze(0)
    po_mask_gt = sample_gt[0].unsqueeze(0)

    # Mask the invalid values
    po_mask_pred[po_mask_gt == 255] = 255

    # Save the raw masked version
    po_mask_orig = po_mask_pred.permute(1, 2, 0).cpu().numpy().astype(np.uint16)
    cv2.imwrite(img_name, po_mask_orig)

    # Get the RGB image of the po_pred
    po_mask_rgb = visualisePanopticMaskTrainId(po_mask_pred, varargs['dataset'])
    po_mask_rgb = po_mask_rgb.permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(img_name_rgb, po_mask_rgb)


def savePanopticLogits(sample, sample_category, save_tuple, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        save_dir_stuff = os.path.join(save_path, cam_name, "{}_stuff".format(sample_category))
        save_dir_all = os.path.join(save_path, cam_name, "{}_all".format(sample_category))
    else:
        save_dir_stuff = os.path.join(save_path, cam_name, "{}_stuff".format(sample_category))
        save_dir_all = os.path.join(save_path, cam_name, "{}_all".format(sample_category))

    if not os.path.exists(save_dir_stuff):
        os.makedirs(save_dir_stuff)
    if not os.path.exists(save_dir_all):
        os.makedirs(save_dir_all)

    logits_name_all = os.path.join(save_dir_all, "{}.bin".format(sample_name))
    logits_name_stuff = os.path.join(save_dir_stuff, "{}.bin".format(sample_name))

    logits_stuff = sample[:varargs['num_stuff'], ...].cpu().numpy()
    logits_all = sample.cpu().numpy()

    # Save the raw version of the mask
    np.save(logits_name_all, logits_all)
    np.save(logits_name_stuff, logits_stuff)