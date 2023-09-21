import torch
import torch.distributed as distributed

# from .bbx import invert_roi_bbx
# from .misc import Empty
# from .roi_sampling import roi_sampling
import torch.nn.functional as functional
# import po_bev_unsupervised.utils.coco_ap as coco_ap
from po_bev_unsupervised.utils.sequence import pad_packed_images

import numpy as np

class PanopticPreprocessing:
    def __init__(self,
                 score_threshold=0.5,
                 overlap_threshold=0.5,
                 min_stuff_area=64 * 64):
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold
        self.min_stuff_area = min_stuff_area

    def __call__(self, sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, sem_logits, num_stuff):
        img_size = [sem_pred.size(0), sem_pred.size(1)]

        # Initialize outputs
        occupied = torch.zeros_like(sem_pred, dtype=torch.uint8)
        occupied_ = torch.zeros_like(sem_pred, dtype=torch.uint8)
        occupied2_ = torch.zeros_like(sem_pred, dtype=torch.int64)
 
        msk = torch.zeros_like(sem_pred)
        cat = [255]
        obj = [0]
        obj2 = [0]
        iscrowd = [0]
        cat2 = [255]
        iscrowd1 = [0] 
        cat1 = [255]
        j_fr = 1
        # Process thing
        try:
           if bbx_pred is None or cls_pred is None or obj_pred is None or msk_pred is None:
                raise Empty
           scale = 1.0
           img_n_size = list(sem_logits.shape[-2:])

            # Remove low-confidence instances
           keep = obj_pred > self.score_threshold
           if not keep.any():
                raise Empty
           obj_pred, bbx_pred, cls_pred, msk_pred = obj_pred[keep], bbx_pred[keep], cls_pred[keep], msk_pred[keep]

           idx = torch.argsort(obj_pred, descending=True)
 
           keep_inds = []
           mask_energy = msk_pred.new_zeros(msk_pred.shape[0], sem_logits.shape[-2], sem_logits.shape[-1])
           
           cls_pred_=[]
           obj_pred_=[]
           bbx_pred_=bbx_pred.new_zeros(bbx_pred.shape)
           idx_len = idx.shape[0]
           max_limit = 100
           i=0    
           for idx_sub in range(0,idx_len,max_limit):
            if (idx_len-idx_sub)<max_limit:
                max_limit=idx_len     

            bbx_inv1 = invert_roi_bbx(bbx_pred[idx[idx_sub:idx_sub+max_limit]]*scale, list(msk_pred[idx[idx_sub:idx_sub+max_limit]].shape[-2:]), img_n_size)
            
            bbx_idx = torch.arange(0, msk_pred[idx[idx_sub:idx_sub+max_limit]].size(0), dtype=torch.long, device=msk_pred.device)
            msk_pred1 = roi_sampling(msk_pred[idx[idx_sub:idx_sub+max_limit]].unsqueeze(1), bbx_inv1, bbx_idx, tuple(img_n_size), padding="zero")


            
            # Process instances
            
            for id, msk_f, cls_i, obj_i in zip(idx[idx_sub:idx_sub+max_limit], msk_pred1, cls_pred[idx[idx_sub:idx_sub+max_limit]], obj_pred[idx[idx_sub:idx_sub+max_limit]]):
                # Check intersection
                y0 = max(int(bbx_pred[id,0]), 0)
                y1 = min(int(bbx_pred[id,2].round()+1),img_n_size[0])
                x0 = max(int(bbx_pred[id,1]), 0)
                x1 = min(int(bbx_pred[id,3].round()+1),img_n_size[1])
                I = torch.zeros((msk_f.shape[-2:]), dtype=msk_f.dtype, device=msk_f.device)
                I[y0:y1, x0:x1] = 1
                msk_f[0,:, :] *= I
                  
                msk_i = (msk_f.sigmoid() > 0.5)[0] 
                msk_i = msk_i.type(torch.uint8)
                intersection = occupied_ & msk_i
                if intersection.float().sum() / msk_i.float().sum() > self.overlap_threshold:
                    continue

                # Add non-intersecting part to output
                msk_i = msk_i - intersection
                keep_inds.append(id.cpu().numpy())
 
                # Update occupancy mask
                msk_i = msk_i.type(torch.uint8)
                occupied_ += msk_i
                
                mask_energy[i, ...]= msk_f
                cls_pred_.append(cls_i.item())
                obj_pred_.append(obj_i.item())
                bbx_pred_[i, ...]=bbx_pred[id, ...]
                i += 1
            
           cls_pred_ = torch.tensor(cls_pred_, dtype=torch.long)
           obj_pred_ = torch.tensor(obj_pred_, dtype=torch.float)
           bbx_pred_ = bbx_pred_[:len(keep_inds)]
           mask_energy = mask_energy[:len(keep_inds)]
           del msk_pred1, cls_pred, obj_pred

           seg_inst_logits = torch.zeros((cls_pred_.shape[0], sem_logits.shape[-2], sem_logits.shape[-1]), device=sem_logits.device)

           for idx in range(cls_pred_.shape[0]):
                y0 = int(bbx_pred_[idx,0]+1*scale)
                y1 = int((bbx_pred_[idx,2]-1*scale).round()+1)
                x0 = int(bbx_pred_[idx,1]+1*scale)
                x1 = int((bbx_pred_[idx,3]-1*scale).round()+1)
                
                seg_inst_logits[idx, y0: y1, x0: x1] = sem_logits[cls_pred_[idx]+num_stuff, y0: y1, x0: x1]

           mask_energy = (((mask_energy*4).sigmoid()+seg_inst_logits.sigmoid()))*(4*mask_energy+seg_inst_logits)
           del seg_inst_logits 
           panoptic_logits = torch.cat([sem_logits[:num_stuff], (mask_energy)], dim=0)
           del mask_energy
            
        except Empty:
            panoptic_logits = sem_logits[:num_stuff]  
            pass

        
        panoptic_msk = torch.argmax(functional.softmax(panoptic_logits, dim=0),dim=0)+1
            
        for cls_i in np.unique(panoptic_msk.cpu().numpy()):
            if cls_i-1 < num_stuff:
                    a=1 # rewrite it later
            else:
                obj2.append(obj_pred_[cls_i-1-num_stuff].cpu())
                cat2.append(cls_pred_[cls_i-1-num_stuff].cpu().numpy()+num_stuff)
                occupied2_[panoptic_msk==cls_i]=j_fr
                j_fr=j_fr+1   
                iscrowd1.append(0)
      
        occupied[occupied2_>0]=1
        for cls_i in range(sem_pred.max().item() + 1):
            msk_i = sem_pred == cls_i
            # Remove occupied part and check remaining area
            msk_i = msk_i.type(torch.uint8)
            msk_i = msk_i & ~occupied
            if msk_i.float().sum() < self.min_stuff_area:
                continue
            if cls_i < num_stuff:
                occupied2_[msk_i]=j_fr
                cat2.append(cls_i) 
                iscrowd1.append(0)
                j_fr = j_fr+1  
            # Add non-intersecting part to output
            msk[msk_i] = len(cat)
            cat.append(cls_i)
            obj.append(1)
            iscrowd.append(cls_i >= num_stuff)

            # Update occupancy mask
            msk_i = msk_i.type(torch.uint8)
            occupied += msk_i

        # Wrap in tensors
        cat2 = torch.tensor(cat2, dtype=torch.long)
        iscrowd1 = torch.tensor(iscrowd1, dtype=torch.uint8)
        obj2 = torch.tensor(obj2, dtype=torch.float)

        return occupied2_.cpu(), cat2, obj2, iscrowd1


def panoptic_stats(msk_gt, cat_gt, panoptic_pred, num_classes, _num_stuff):
    # Move gt to CPU
    msk_gt, cat_gt = msk_gt.cpu(), cat_gt.cpu()
    msk_pred, cat_pred, _, iscrowd_pred = panoptic_pred
    
    # Convert crowd predictions to void
    msk_remap = msk_pred.new_zeros(cat_pred.numel())
    msk_remap[~(iscrowd_pred > 0)] = torch.arange(0, (~(iscrowd_pred>0)).long().sum().item(), dtype=msk_remap.dtype, device=msk_remap.device)

    msk_pred = msk_remap[msk_pred]
    cat_pred = cat_pred[~(iscrowd_pred>0)]

    iou = msk_pred.new_zeros(num_classes, dtype=torch.double)
    tp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fn = msk_pred.new_zeros(num_classes, dtype=torch.double)

    if cat_gt.numel() > 1:
        msk_gt = msk_gt.view(-1)
        msk_pred = msk_pred.view(-1)

        # Compute confusion matrix
        confmat = msk_pred.new_zeros(cat_gt.numel(), cat_pred.numel(), dtype=torch.double)
        confmat.view(-1).index_add_(0, msk_gt * cat_pred.numel() + msk_pred,
                                    confmat.new_ones(msk_gt.numel()))

        # track potentially valid FP, i.e. those that overlap with void_gt <= 0.5
        num_pred_pixels = confmat.sum(0)
        valid_fp = (confmat[0] / num_pred_pixels) <= 0.5

        # compute IoU without counting void pixels (both in gt and pred)
        _iou = confmat / ((num_pred_pixels - confmat[0]).unsqueeze(0) + confmat.sum(1).unsqueeze(1) - confmat)

        # flag TP matches, i.e. same class and iou > 0.5
        matches = ((cat_gt.unsqueeze(1) == cat_pred.unsqueeze(0)) & (_iou > 0.5))

        # remove potential match of void_gt against void_pred
        matches[0, 0] = 0

        _iou = _iou[matches]
        tp_i, _ = matches.max(1)
        fn_i = ~tp_i
        fn_i[0] = 0  # remove potential fn match due to void against void
        fp_i = ~matches.max(0)[0] & valid_fp
        fp_i[0] = 0  # remove potential fp match due to void against void

        # Compute per instance classes for each tp, fp, fn
        tp_cat = cat_gt[tp_i]
        fn_cat = cat_gt[fn_i]
        fp_cat = cat_pred[fp_i]

        # Accumulate per class counts
        if tp_cat.numel() > 0:
            tp.index_add_(0, tp_cat, tp.new_ones(tp_cat.numel()))
        if fp_cat.numel() > 0:
            fp.index_add_(0, fp_cat, fp.new_ones(fp_cat.numel()))
        if fn_cat.numel() > 0:
            fn.index_add_(0, fn_cat, fn.new_ones(fn_cat.numel()))
        if tp_cat.numel() > 0:
            iou.index_add_(0, tp_cat, _iou)

    # note else branch is not needed because if cat_gt has only void we don't penalize predictions
    return iou, tp, fp, fn


def performPanopticFusion(result, idx, msk, cat, iscrowd, **varargs):
    # Accumulate COCO AP and panoptic predictions
    panoptic_pred_list = []
    for i, (idy, po_pred, po_class, po_iscrowd, sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, sem_logits, msk_gt, cat_gt, iscrowd) in enumerate(
            zip(idx, result['po_pred'], result['po_class'], result['po_iscrowd'], result["sem_pred"], result["bbx_pred"], result["cls_pred"], result["obj_pred"], result["msk_pred"],
                result["sem_logits"], msk, cat, iscrowd)):
        msk_gt = msk_gt.squeeze(0)
        sem_gt = cat_gt[msk_gt]

        # Remove crowd from gt
        cmap = msk_gt.new_zeros(cat_gt.numel())
        cmap[~(iscrowd > 0)] = torch.arange(0, (~(iscrowd > 0)).long().sum().item(), dtype=cmap.dtype, device=cmap.device)
        msk_gt = cmap[msk_gt]
        cat_gt = cat_gt[~(iscrowd > 0)]

        # Compute panoptic output
        # panoptic_pred = varargs['make_panoptic'](sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, sem_logits, varargs['num_stuff'])
        panoptic_pred_list.append({"po_pred": (po_pred, po_class, None, po_iscrowd),
                                   "msk_gt": msk_gt,
                                   "sem_gt": sem_gt,
                                   "cat_gt": cat_gt,
                                   # "bbx_pred": bbx_pred,
                                   # 'cls_pred': cls_pred,
                                   # "obj_pred": obj_pred,
                                   # "msk_pred": msk_pred,
                                   "idx": idx[i]})

    return panoptic_pred_list


def confusionMatrix(gt, pred):
    conf_mat = gt.new_zeros(256 * 256, dtype=torch.float)
    conf_mat.index_add_(0, gt.view(-1) * 256 + pred.view(-1), conf_mat.new_ones(gt.numel()))
    return conf_mat.view(256, 256)


def computePanopticTestMetrics(panoptic_pred_list, panoptic_buffer, conf_mat, coco_struct, eval_mode, eval_coco, **varargs):

    for i, po_dict in enumerate(panoptic_pred_list):
        sem_gt = po_dict['sem_gt']
        msk_gt = po_dict['msk_gt']
        cat_gt = po_dict['cat_gt']
        idx = po_dict['idx']
        panoptic_pred = po_dict['po_pred']

        panoptic_buffer += torch.stack(panoptic_stats(msk_gt, cat_gt, panoptic_pred, varargs['num_classes'], varargs['num_stuff']), dim=0)

        if eval_mode == "panoptic":
            # Calculate confusion matrix on panoptic output
            sem_pred = panoptic_pred[1][panoptic_pred[0]]

            conf_mat_i = confusionMatrix(sem_gt.cpu(), sem_pred)
            conf_mat += conf_mat_i.to(conf_mat)

            # Update coco AP from panoptic output
            if eval_coco and ((panoptic_pred[1] >= varargs['num_stuff']) & (panoptic_pred[1] != 255)).any():
                coco_struct += coco_ap.process_panoptic_prediction(
                    panoptic_pred[:4], varargs['num_stuff'], idx, varargs["batch_sizes"][i], varargs["original_sizes"][i])
                img_list.append(idxs[i])

    return panoptic_buffer, conf_mat

        # elif eval_mode == "separate":
        #     # Update coco AP from detection output
        #     if eval_coco and bbx_pred is not None:
        #         coco_struct += coco_ap.process_prediction(
        #             bbx_pred, cls_pred + varargs['num_stuff'], obj_pred, msk_pred, batch_sizes[i], idx, original_sizes[i])
        #         img_list.append(idxs[i])


def getPanopticScores(panoptic_buffer, scores_out, device, num_stuff, debug):
    # Gather from all workers
    panoptic_buffer = panoptic_buffer.to(device)
    if not debug:
        distributed.all_reduce(panoptic_buffer, distributed.ReduceOp.SUM)

    # From buffers to scores
    denom = panoptic_buffer[1] + 0.5 * (panoptic_buffer[2] + panoptic_buffer[3])
    denom[denom == 0] = 1.
    scores = panoptic_buffer[0] / denom
    RQ = panoptic_buffer[1] / denom
    # print ("RQ", (RQ).mean().item(),(RQ[:num_stuff]).mean().item(),(RQ[num_stuff:]).mean().item())
    panoptic_buffer[1][panoptic_buffer[1] == 0] = 1.
    SQ = panoptic_buffer[0] / panoptic_buffer[1]
    # print ("SQ", (SQ).mean().item(),(SQ[:num_stuff]).mean().item(),(SQ[num_stuff:]).mean().item())

    scores_out["pq"] = scores.mean()
    scores_out["pq_stuff"] = scores[:num_stuff].mean()
    scores_out["pq_thing"] = scores[num_stuff:].mean()
    scores_out["sq"] = SQ.mean()
    scores_out["sq_stuff"] = (SQ[:num_stuff]).mean()
    scores_out["sq_thing"] = (SQ[num_stuff:]).mean()
    scores_out["rq"] = RQ.mean()
    scores_out["rq_stuff"] = (RQ[:num_stuff]).mean()
    scores_out["rq_thing"] = (RQ[num_stuff:]).mean()

    return scores_out


def makePanopticGTList(msk, cat, iscrowd):
    msk_up, _ = pad_packed_images(msk)

    panoptic_gt_list = []
    for b in range(msk_up.shape[0]):
        panoptic_gt_list.append((msk_up[b, 0, :, :], cat[b], [], iscrowd[b]))

    return panoptic_gt_list


def make_semantic_gt_list(msk, cat):
    sem_out = []
    for msk_timestep, cat_timestep in zip(msk, cat):
        sem_timestep_out = []
        for msk_i, cat_i in zip(msk_timestep, cat_timestep):
            msk_i = msk_i.squeeze(0)
            sem_timestep_out.append(cat_i[msk_i])
        sem_out.append(sem_timestep_out)
    return sem_out
