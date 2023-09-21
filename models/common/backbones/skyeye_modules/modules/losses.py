import torch
import torch.nn as nn

from po_bev_unsupervised.utils.parallel import PackedSequence


def smooth_l1(x1, x2, sigma):
    """Smooth L1 loss"""
    sigma2 = sigma ** 2

    diff = x1 - x2
    abs_diff = diff.abs()

    mask = (abs_diff.detach() < (1. / sigma2))
    return mask * (sigma2 / 2.) * diff ** 2 + (1 - mask) * (abs_diff - 0.5 / sigma2)

def l1_pixelwise(x1, x2):
    """Pixelwise l1 loss"""
    assert (x1.dim() == x2.dim())
    return (x1-x2).abs()

def ohem_loss(loss, ohem=None):
    if isinstance(loss, torch.Tensor):
        loss = loss.view(loss.size(0), -1)
        if ohem is None:
            return loss.mean()

        top_k = min(max(int(ohem * loss.size(1)), 1), loss.size(1))
        if top_k != loss.size(1):
            loss, _ = loss.topk(top_k, dim=1)

        return loss.mean()
    elif isinstance(loss, PackedSequence):
        if ohem is None:
            return sum(loss_i.mean() for loss_i in loss) / len(loss)

        loss_out = loss.data.new_zeros(())
        for loss_i in loss:
            loss_i = loss_i.view(-1)

            top_k = min(max(int(ohem * loss_i.numel()), 1), loss_i.numel())
            if top_k != loss_i.numel():
                loss_i, _ = loss_i.topk(top_k, dim=0)

            loss_out += loss_i.mean()

        return loss_out / len(loss)

class EdgeAwareSmoothnessLoss(nn.Module):
    def __init__(self):
        super(EdgeAwareSmoothnessLoss, self).__init__()

    def forward(self, inv_depth_map, image):

        # Compute the gradients
        abs_grad_inv_depth_x = (inv_depth_map[:, :, :, 1:] - inv_depth_map[:, :, :, :-1]).abs()
        abs_grad_inv_depth_y = (inv_depth_map[:, :, 1:, :] - inv_depth_map[:, :, :-1, :]).abs()

        abs_grad_image_x = ((image[:, :, :, 1:] - image[:, :, :, :-1]).abs()).mean(1, keepdim=True)
        abs_grad_image_y = ((image[:, :, 1:, :] - image[:, :, :-1, :]).abs()).mean(1, keepdim=True)

        # Compute the final loss
        loss_x = abs_grad_inv_depth_x*torch.exp(-abs_grad_image_x)
        loss_y = abs_grad_inv_depth_y*torch.exp(-abs_grad_image_y)

        loss = loss_x.mean() + loss_y.mean()

        return loss

class SSIMLoss(nn.Module):
    def __init__(self, window_size=3):
        super(SSIMLoss, self).__init__()
        padding = window_size // 2

        self.mu_pool = nn.AvgPool2d(window_size, padding)
        self.sig_pool = nn.AvgPool2d(window_size, padding)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, src_img, tgt_img):
        x = self.refl(src_img)
        y = self.refl(tgt_img)

        mu_x = self.mu_pool(x)
        mu_y = self.mu_pool(y)

        sigma_x = self.sig_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return (torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1))
