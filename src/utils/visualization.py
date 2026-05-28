from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.io import ensure_dir


def save_training_preview(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_dir: str | Path,
    max_items: int,
) -> None:
    ensure_dir(output_dir)
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            preds = model(batch["input"].to(device)).cpu()
            for i in range(preds.shape[0]):
                sample_id = str(batch["sample_id"][i])
                image = batch["image"][i].permute(1, 2, 0).cpu().numpy()
                target = batch["height"][i, 0].cpu().numpy()
                pred = preds[i, 0].cpu().numpy()
                error = np.abs(pred - target)
                mae = float(error.mean())
                rmse = float(np.sqrt(np.mean((pred - target) ** 2)))

                fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                axes[0].imshow(np.clip(image, 0.0, 1.0))
                axes[0].set_title("RGB")
                axes[1].imshow(target, cmap="terrain", vmin=0.0, vmax=1.0)
                axes[1].set_title("Target")
                axes[2].imshow(pred, cmap="terrain", vmin=0.0, vmax=1.0)
                axes[2].set_title("Prediction")
                axes[3].imshow(error, cmap="magma")
                axes[3].set_title("Abs Error")
                fig.suptitle(
                    f"pred min/max: {pred.min():.3f}/{pred.max():.3f} | "
                    f"target min/max: {target.min():.3f}/{target.max():.3f} | "
                    f"MAE: {mae:.4f} | RMSE: {rmse:.4f}",
                    fontsize=9,
                )
                for ax in axes:
                    ax.axis("off")
                fig.tight_layout(rect=(0, 0, 1, 0.92))
                fig.savefig(Path(output_dir) / f"{sample_id}_overview.png", dpi=160)
                plt.close(fig)

                saved += 1
                if saved >= max_items:
                    return


def save_inference_outputs(output_dir: str | Path, image: torch.Tensor, prediction: np.ndarray) -> None:
    output_dir = ensure_dir(output_dir)
    np.save(output_dir / "pred_height.npy", prediction.astype(np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Input RGB")
    axes[1].imshow(prediction, cmap="terrain", vmin=0.0, vmax=1.0)
    axes[1].set_title("Predicted Height")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_overview.png", dpi=180)
    plt.close(fig)

    pred_u8 = np.clip(prediction * 255.0, 0.0, 255.0).astype(np.uint8)
    plt.imsave(output_dir / "pred_height.png", pred_u8, cmap="terrain")
