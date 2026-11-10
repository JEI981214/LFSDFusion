import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, image_fused, vis, ir):
        assert (vis.size() == ir.size() == image_fused.size()), "Input image shapes must match"

        ir_mse = F.mse_loss(image_fused, ir, reduction=self.reduction)
        vis_mse = F.mse_loss(image_fused, vis, reduction=self.reduction)

        return ir_mse + vis_mse


class JointGrad(nn.Module):
    def __init__(self):
        super(JointGrad, self).__init__()

        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir, im_vi):

        ir_grad = torch.abs(self.laplacian(im_ir, 3))
        vi_grad = torch.abs(self.laplacian(im_vi, 3))
        fus_grad = torch.abs(self.laplacian(im_fus, 3))

        loss_JGrad = self.l1_loss(torch.max(ir_grad, vi_grad), fus_grad)

        return loss_JGrad
