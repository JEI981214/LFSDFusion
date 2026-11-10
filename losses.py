import torch
import torch.nn as nn

import kornia
import lpips
from loss_c import JointGrad, MSELoss


class FusionLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lambda_l2 = float(cfg["loss"]["lambda_l2"])
        self.lambda_ssim = float(cfg["loss"]["lambda_ssim"])
        self.lambda_jgrad = float(cfg["loss"]["lambda_jgrad"])
        self.lambda_lpips = float(cfg["loss"]["lambda_sem"])

        self.mse_loss = MSELoss()
        self.joint_grad = JointGrad()

        self.net_lpips = lpips.LPIPS(net="vgg").eval()
        for p in self.net_lpips.parameters():
            p.requires_grad_(False)

    def forward(self, x_tgt_pred, x_src11, x_src22, return_parts=False):


        x_tgt_preds = (x_tgt_pred - x_tgt_pred.min()) / (x_tgt_pred.max() - x_tgt_pred.min() + 1e-8)
        x_src11s = (x_src11 - x_src11.min()) / (x_src11.max() - x_src11.min() + 1e-8)
        x_src22s = (x_src22 - x_src22.min()) / (x_src22.max() - x_src22.min() + 1e-8)
        loss_ssim1 = 1 - kornia.metrics.ssim(x_tgt_preds.float(), x_src11s.float(), window_size=11).mean()
        loss_ssim2 = 1 - kornia.metrics.ssim(x_tgt_preds.float(), x_src22s.float(), window_size=11).mean()
        loss_ssim = (0.5 * loss_ssim1 + 0.5 * loss_ssim2) * self.lambda_ssim

        # L_grad
        loss_jgrad = self.joint_grad(x_tgt_pred, x_src11, x_src22) * self.lambda_jgrad


        loss_l2 = self.mse_loss(x_tgt_pred.float(), x_src11.float(), x_src22.float()) * self.lambda_l2

        # L_per
        loss_lpips = (self.net_lpips(x_tgt_pred.float(), x_src11.float()).mean()
                      + self.net_lpips(x_tgt_pred.float(), x_src22.float()).mean()) * 0.5 * self.lambda_lpips

        loss = loss_ssim + loss_l2 + loss_jgrad + loss_lpips
        if return_parts:
            return loss, {"l2": loss_l2, "ssim": loss_ssim, "jgrad": loss_jgrad, "lpips": loss_lpips}
        return loss
