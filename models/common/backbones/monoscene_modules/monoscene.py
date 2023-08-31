import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.common.backbones.monoscene_modules.unet3d_nyu import UNet3D as UNet3DNYU
from models.common.backbones.monoscene_modules.unet3d_kitti import UNet3D as UNet3DKitti
from models.common.backbones.monoscene_modules.flosp import FLoSP
#from monoscene.loss.CRP_loss import compute_super_CP_multilabel_loss
#from monoscene.loss.sscMetrics import SSCMetrics
#from monoscene.loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss
import numpy as np
import torch.nn.functional as F
from models.common.backbones.monoscene_modules.unet2d import UNet2D
from models.common.backbones.monoscene_modules.monodepth2_m import Monodepth2
from torch.optim.lr_scheduler import MultiStepLR

import time

class MonoScene(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        class_names,
        feature,
        project_scale,
        full_scene_size,
        dataset,
        n_relations=4,
        context_prior=False,
        fp_loss=True,
        project_res=["1", "2", "4", "8"],
        frustum_size=4,
        relation_loss=False,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        lr=1e-4,
        weight_decay=1e-4,
        beta=0.0,
    ):
        super().__init__()
        
        self.project_res = project_res
        self.fp_loss = fp_loss
        self.dataset = dataset
        self.context_prior = context_prior
        self.frustum_size = frustum_size
        self.class_names = class_names
        self.relation_loss = relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.project_scale = project_scale
        self.class_weights = torch.zeros((2))
        self.lr = lr
        self.weight_decay = weight_decay

        self.latent_size = feature // 2

        self.projects = {}
        self.scale_2ds = [1, 2, 4, 8]  # 2D scales
        for scale_2d in self.scale_2ds:
            self.projects[str(scale_2d)] = FLoSP(
                full_scene_size, project_scale=self.project_scale, dataset=self.dataset
            )
        self.projects = nn.ModuleDict(self.projects)

        self.n_classes = n_classes
        if self.dataset == "NYU":
            self.net_3d_decoder = UNet3DNYU(
                self.n_classes,
                nn.BatchNorm3d,
                n_relations=n_relations,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        elif self.dataset == "kitti":
            self.net_3d_decoder = UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=project_scale,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
                beta=beta,
            )
        #self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)

        self.net_rgb = Monodepth2(resnet_layers=50, cp_location=None, freeze=False, num_ch_dec=[32, 32, 64, 128, 256], d_out=64, scales=range(4))

        # log hyperparameters
        #self.save_hyperparameters()

        #self.train_metrics = SSCMetrics(self.n_classes)
        #self.val_metrics = SSCMetrics(self.n_classes)
        #self.test_metrics = SSCMetrics(self.n_classes)

    def forward(self, img, projected_pixs, fov_masks):

        """
        For extracting volumetric features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W, D)
        """

        #img = batch["img"]

        projected_pix = projected_pixs[0].cuda()
        fov_mask = fov_masks[0].cuda()

        bs = img.shape[0]

        out = {}
        #st = time.time()
        x_rgb = self.net_rgb(img)
        #print("2D: ", time.time() - st) # ~6ms

        x3ds = []
        #st = time.time()
        for i in range(bs):
            x3d = None
            for scale, scale_2d in enumerate(self.project_res):

                # project features at each 2D scale to target 3D scale
                scale_2d = int(scale_2d)
                #projected_pix = batch["projected_pix_{}".format(self.project_scale)][i].cuda()
                #fov_mask = batch["fov_mask_{}".format(self.project_scale)][i].cuda()
                projected_pix = projected_pixs[i].cuda()
                fov_mask = fov_masks[i].cuda()

                # Sum all the 3D features
                if x3d is None:
                    x3d = self.projects[str(scale_2d)](
                        x_rgb[scale][i],
                        projected_pix // scale_2d,
                        fov_mask,
                    )
                else:
                    x3d += self.projects[str(scale_2d)](
                        x_rgb[scale][i],
                        projected_pix // scale_2d,
                        fov_mask,
                    )
            
            x3ds.append(x3d)
        
        input_dict = {
            "x3d": torch.stack(x3ds),
        }
        #print("projection: ", time.time() - st) # ~1ms
        #st = time.time()
        out = self.net_3d_decoder(input_dict)
        #print("3D: ", time.time() - st) # ~4ms, whole net 10ms
        
        return out

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]

        if self.context_prior:
            P_logits = out_dict["P_logits"]
            CP_mega_matrices = batch["CP_mega_matrices"]

            if self.relation_loss:
                loss_rel_ce = compute_super_CP_multilabel_loss(
                    P_logits, CP_mega_matrices
                )
                loss += loss_rel_ce
                self.log(
                    step_type + "/loss_relation_ce_super",
                    loss_rel_ce.detach(),
                    on_epoch=True,
                    sync_dist=True,
                )

        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + "/loss_ssc",
                loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.sem_scal_loss:
            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            loss += loss_sem_scal
            self.log(
                step_type + "/loss_sem_scal",
                loss_sem_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log(
                step_type + "/loss_geo_scal",
                loss_geo_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.fp_loss and step_type != "test":
            frustums_masks = torch.stack(batch["frustums_masks"])
            frustums_class_dists = torch.stack(
                batch["frustums_class_dists"]
            ).float()  # (bs, n_frustums, n_classes)
            n_frustums = frustums_class_dists.shape[1]

            pred_prob = F.softmax(ssc_pred, dim=1)
            batch_cnt = frustums_class_dists.sum(0)  # (n_frustums, n_classes)

            frustum_loss = 0
            frustum_nonempty = 0
            for frus in range(n_frustums):
                frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
                prob = frustum_mask * pred_prob  # bs, n_classes, H, W, D
                prob = prob.reshape(bs, self.n_classes, -1).permute(1, 0, 2)
                prob = prob.reshape(self.n_classes, -1)
                cum_prob = prob.sum(dim=1)  # n_classes

                total_cnt = torch.sum(batch_cnt[frus])
                total_prob = prob.sum()
                if total_prob > 0 and total_cnt > 0:
                    frustum_target_proportion = batch_cnt[frus] / total_cnt
                    cum_prob = cum_prob / total_prob  # n_classes
                    frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion)
                    frustum_loss += frustum_loss_i
                    frustum_nonempty += 1
            frustum_loss = frustum_loss / frustum_nonempty
            loss += frustum_loss
            self.log(
                step_type + "/loss_frustums",
                frustum_loss.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [("test", self.test_metrics)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print(
                "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                    stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
                )
            )
            print("class IoU: {}, ".format(classes))
            print(
                " ".join(["{:.4f}, "] * len(classes)).format(
                    *(stats["iou_ssc"] * 100).tolist()
                )
            )
            print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
            metric.reset()

    def configure_optimizers(self):
        if self.dataset == "NYU":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]
        elif self.dataset == "kitti":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]
    

    @classmethod
    def from_conf(cls, conf):
        return cls(
            n_classes=conf.get("n_classes", 2),
            class_names=conf.get("class_names", ["occupied", "empty"]),
            feature=conf.get("feature", 64),
            project_scale=conf.get("project_scale", 2),
            full_scene_size=conf.get("full_scene_size", (256, 32, 256)),
            dataset=conf.get("dataset", "kitti"),
            n_relations=conf.get("n_relations", 4),
            frustum_size=conf.get("frustum_size", 4),
            beta=conf.get("beta", 0.0)
        )
