from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class SSIMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = self.pool(F.pad(pred, (1, 1, 1, 1), mode="reflect"))
        mu_y = self.pool(F.pad(target, (1, 1, 1, 1), mode="reflect"))
        sigma_x = self.pool(F.pad(pred * pred, (1, 1, 1, 1), mode="reflect")) - mu_x * mu_x
        sigma_y = self.pool(F.pad(target * target, (1, 1, 1, 1), mode="reflect")) - mu_y * mu_y
        sigma_xy = self.pool(F.pad(pred * target, (1, 1, 1, 1), mode="reflect")) - mu_x * mu_y
        ssim_n = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        ssim_d = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim = ssim_n / (ssim_d + 1e-6)
        return torch.clamp((1.0 - ssim) * 0.5, 0.0, 1.0).mean()


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight: float, grad_weight: float, ssim_weight: float) -> None:
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.grad_weight = float(grad_weight)
        self.ssim_weight = float(ssim_weight)
        self.l1 = nn.L1Loss()
        self.grad = GradientLoss()
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.l1_weight * self.l1(pred, target)
        if self.grad_weight > 0.0:
            loss = loss + self.grad_weight * self.grad(pred, target)
        if self.ssim_weight > 0.0:
            loss = loss + self.ssim_weight * self.ssim(pred, target)
        return loss
