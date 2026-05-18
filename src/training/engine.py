from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TerrainDataset
from src.models import UNet
from src.training.losses import CombinedLoss
from src.utils.io import ensure_dir, save_checkpoint, resolve_repo_path
from src.utils.metrics import batch_metrics, merge_metric_sums
from src.utils.visualization import save_training_preview


def _device_from_config(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _make_dataloaders(config: dict, repo_root: Path) -> tuple[TerrainDataset, TerrainDataset, DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    train_ds = TerrainDataset(
        root=resolve_repo_path(dataset_cfg["root"], repo_root),
        manifest_path=resolve_repo_path(dataset_cfg["train_manifest"], repo_root),
        image_size=int(dataset_cfg["image_size"]),
        metadata_keys=list(dataset_cfg["metadata_keys"]),
        use_pair=bool(dataset_cfg["use_pair"]),
        use_metadata=bool(dataset_cfg["use_metadata"]),
        split="train",
        shadow_augmentation=bool(dataset_cfg.get("shadow_augmentation", False)),
    )
    val_ds = TerrainDataset(
        root=resolve_repo_path(dataset_cfg["root"], repo_root),
        manifest_path=resolve_repo_path(dataset_cfg["val_manifest"], repo_root),
        image_size=int(dataset_cfg["image_size"]),
        metadata_keys=list(dataset_cfg["metadata_keys"]),
        use_pair=bool(dataset_cfg["use_pair"]),
        use_metadata=bool(dataset_cfg["use_metadata"]),
        split="val",
        shadow_augmentation=False,
    )
    training_cfg = config["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return train_ds, val_ds, train_loader, val_loader


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    metric_sums: dict[str, float] = {}
    total_items = 0

    progress = tqdm(loader, leave=False)
    for batch in progress:
        inputs = batch["input"].to(device)
        targets = batch["height"].to(device)
        shadow_mask = batch["shadow_mask"].to(device)

        with torch.set_grad_enabled(is_train):
            preds = model(inputs)
            loss = criterion(preds, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = inputs.shape[0]
        total_items += batch_size
        batch_result = batch_metrics(preds.detach(), targets.detach(), shadow_mask.detach())
        batch_result["loss"] = float(loss.item())
        merge_metric_sums(metric_sums, batch_result, batch_size)
        progress.set_description(f"{'train' if is_train else 'val'} loss={loss.item():.4f}")

    return {key: value / max(total_items, 1) for key, value in metric_sums.items()}


def train_from_config(config: dict) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    device = _device_from_config(config["training"]["device"])
    train_ds, val_ds, train_loader, val_loader = _make_dataloaders(config, repo_root)

    dataset_cfg = config["dataset"]
    input_channels = 3 * (2 if dataset_cfg["use_pair"] else 1) + (train_ds.metadata_dim if dataset_cfg["use_metadata"] else 0)
    model = UNet(
        in_channels=input_channels,
        out_channels=1,
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)

    loss_cfg = config["loss"]
    criterion = CombinedLoss(
        l1_weight=float(loss_cfg["l1_weight"]),
        grad_weight=float(loss_cfg["grad_weight"]),
        ssim_weight=float(loss_cfg["ssim_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    output_dir = resolve_repo_path(config["training"]["output_dir"], repo_root)
    ckpt_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)

    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    epochs = int(config["training"]["epochs"])
    num_visualizations = int(config["training"].get("save_visualizations", 4))

    for epoch in range(epochs):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device, optimizer=None)

        row = {"epoch": epoch + 1}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_config": config,
            "best_val_loss": min(best_val, val_metrics["loss"]),
        }
        save_checkpoint(ckpt_dir / "last.pt", payload)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(ckpt_dir / "best.pt", payload)

        epoch_sample_dir = sample_dir / f"epoch_{epoch + 1:03d}"
        save_training_preview(model, val_loader, device, epoch_sample_dir, max_items=num_visualizations)

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Training finished. Best validation loss: {best_val:.4f}")
