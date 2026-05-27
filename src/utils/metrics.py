from __future__ import annotations

import torch
import torch.nn.functional as F


def _gradient_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask_sum = float(mask.sum().item())
    if mask_sum < 1e-6:
        return 0.0
    return float((values * mask).sum().item() / mask_sum)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor, shadow_mask: torch.Tensor) -> dict[str, float]:
    abs_error = (pred - target).abs()
    sq_error = (pred - target) ** 2
    shadow = (shadow_mask > 0.5).float()
    non_shadow = 1.0 - shadow
    dilated_shadow = F.max_pool2d(shadow, kernel_size=7, stride=1, padding=3)
    eroded_shadow = -F.max_pool2d(-shadow, kernel_size=7, stride=1, padding=3)
    boundary = (dilated_shadow - eroded_shadow).clamp(0.0, 1.0)
    return {
        "mae": float(abs_error.mean().item()),
        "rmse": float(torch.sqrt(sq_error.mean()).item()),
        "grad_error": float(_gradient_error(pred, target).item()),
        "shadow_mae": _masked_mean(abs_error, shadow),
        "non_shadow_mae": _masked_mean(abs_error, non_shadow),
        "boundary_mae": _masked_mean(abs_error, boundary),
    }


def merge_metric_sums(target: dict[str, float], batch: dict[str, float], batch_size: int) -> None:
    for key, value in batch.items():
        target[key] = target.get(key, 0.0) + float(value) * batch_size
