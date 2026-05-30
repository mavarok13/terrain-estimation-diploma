from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TerrainDataset, resolve_input_mode
from src.dataset.input_modes import build_model_input_for_mode, input_channels_for_mode, resolve_input_mode as resolve_checkpoint_input_mode
from src.models import UNet
from src.training.losses import CombinedLoss
from src.utils.io import ensure_dir, load_checkpoint, save_checkpoint, resolve_repo_path
from src.utils.metrics import batch_metrics, merge_metric_sums
from src.utils.visualization import save_training_preview


def _device_from_config(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _make_dataloaders(config: dict, repo_root: Path) -> tuple[TerrainDataset, TerrainDataset, DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    input_mode = resolve_input_mode(config.get("training", {}), dataset_cfg)
    train_ds = TerrainDataset(
        root=resolve_repo_path(dataset_cfg["root"], repo_root),
        manifest_path=resolve_repo_path(dataset_cfg["train_manifest"], repo_root),
        image_size=int(dataset_cfg["image_size"]),
        metadata_keys=list(dataset_cfg["metadata_keys"]),
        use_pair=bool(dataset_cfg["use_pair"]),
        use_metadata=bool(dataset_cfg["use_metadata"]),
        split="train",
        shadow_augmentation=bool(dataset_cfg.get("shadow_augmentation", False)),
        input_mode=input_mode,
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
        input_mode=input_mode,
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


def _load_compatible_model_state(
    model: torch.nn.Module,
    checkpoint: dict,
    prefixes: tuple[str, ...] | None = None,
) -> list[str]:
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    compatible_state = {}
    skipped = []

    for name, value in source_state.items():
        if prefixes is not None and not name.startswith(prefixes):
            continue
        if name not in target_state:
            skipped.append(f"{name} (missing in model)")
            continue
        if tuple(value.shape) != tuple(target_state[name].shape):
            skipped.append(f"{name} checkpoint{tuple(value.shape)} != model{tuple(target_state[name].shape)}")
            continue
        compatible_state[name] = value

    missing = [name for name in target_state if name not in compatible_state]
    model.load_state_dict(compatible_state, strict=False)
    skipped.extend(name for name in missing if name not in source_state)
    return skipped


def _input_mode_from_checkpoint(checkpoint: dict) -> str:
    train_config = checkpoint["train_config"]
    return resolve_checkpoint_input_mode(train_config.get("training", {}), train_config.get("dataset", {}))


def _make_model_from_checkpoint(checkpoint: dict, device: torch.device) -> tuple[torch.nn.Module, str]:
    train_config = checkpoint["train_config"]
    dataset_cfg = train_config["dataset"]
    model_cfg = train_config["model"]
    input_mode = _input_mode_from_checkpoint(checkpoint)
    metadata_keys = list(dataset_cfg.get("metadata_keys", []))
    input_channels = int(model_cfg.get("input_channels", input_channels_for_mode(input_mode, metadata_keys)))
    model = UNet(
        in_channels=input_channels,
        out_channels=1,
        base_channels=int(model_cfg["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, input_mode


def _build_teacher_input(batch: dict[str, torch.Tensor | str], input_mode: str, device: torch.device) -> torch.Tensor:
    image = batch["image"].to(device)
    image_alt = batch.get("image_alt")
    metadata = batch.get("metadata")
    shadow_mask = batch.get("shadow_mask")
    shadow_mask_alt = batch.get("shadow_mask_alt")

    if isinstance(image_alt, torch.Tensor):
        image_alt = image_alt.to(device)
    if isinstance(metadata, torch.Tensor):
        metadata = metadata.to(device)
    if isinstance(shadow_mask, torch.Tensor):
        shadow_mask = shadow_mask.to(device)
    if isinstance(shadow_mask_alt, torch.Tensor):
        shadow_mask_alt = shadow_mask_alt.to(device)

    inputs = [
        build_model_input_for_mode(
            image[idx],
            image_alt[idx] if isinstance(image_alt, torch.Tensor) else None,
            metadata[idx] if isinstance(metadata, torch.Tensor) else None,
            input_mode,
            shadow_mask=shadow_mask[idx] if isinstance(shadow_mask, torch.Tensor) else None,
            shadow_mask_alt=shadow_mask_alt[idx] if isinstance(shadow_mask_alt, torch.Tensor) else None,
        )
        for idx in range(image.shape[0])
    ]
    return torch.stack(inputs, dim=0)


def _set_encoder_frozen(model: torch.nn.Module, frozen: bool) -> None:
    for module_name in ("enc1", "enc2", "enc3", "enc4", "bottleneck"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = not frozen


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    teacher_model: torch.nn.Module | None = None,
    teacher_input_mode: str | None = None,
    distill_weight: float = 0.0,
    gt_weight: float = 1.0,
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
        if "shadow_mask_alt" in batch:
            shadow_mask = torch.maximum(shadow_mask, batch["shadow_mask_alt"].to(device))

        with torch.set_grad_enabled(is_train):
            preds = model(inputs)
            gt_loss = criterion(preds, targets)
            loss = gt_weight * gt_loss
            distill_loss = None
            if is_train and teacher_model is not None and teacher_input_mode is not None and distill_weight > 0.0:
                teacher_inputs = _build_teacher_input(batch, teacher_input_mode, device)
                with torch.no_grad():
                    teacher_preds = teacher_model(teacher_inputs)
                distill_loss = F.l1_loss(preds, teacher_preds)
                loss = loss + distill_weight * distill_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = inputs.shape[0]
        total_items += batch_size
        batch_result = batch_metrics(preds.detach(), targets.detach(), shadow_mask.detach())
        batch_result["loss"] = float(loss.item())
        batch_result["gt_loss"] = float(gt_loss.item())
        if distill_loss is not None:
            batch_result["distill_loss"] = float(distill_loss.item())
        merge_metric_sums(metric_sums, batch_result, batch_size)
        progress.set_description(f"{'train' if is_train else 'val'} loss={loss.item():.4f}")

    return {key: value / max(total_items, 1) for key, value in metric_sums.items()}


def train_from_config(config: dict, resume_checkpoint: str | Path | None = None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    device = _device_from_config(config["training"]["device"])
    train_ds, val_ds, train_loader, val_loader = _make_dataloaders(config, repo_root)

    dataset_cfg = config["dataset"]
    input_mode = resolve_input_mode(config.get("training", {}), dataset_cfg)
    input_channels = train_ds.input_channels
    config.setdefault("training", {})["input_mode"] = input_mode
    config.setdefault("model", {})["input_channels"] = input_channels
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
    history_path = output_dir / "history.json"
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)

    print(
        f"Training input_mode={input_mode} input_channels={input_channels} "
        f"train_samples={len(train_ds)} val_samples={len(val_ds)} output_dir={output_dir}"
    )

    distill_cfg = config.get("distillation", {}) or {}
    teacher_model = None
    teacher_input_mode = None
    distill_weight = 0.0
    gt_weight = 1.0
    teacher_checkpoint_path = distill_cfg.get("teacher_checkpoint")
    if bool(distill_cfg.get("enabled", False)):
        if not teacher_checkpoint_path:
            raise ValueError("distillation.enabled requires distillation.teacher_checkpoint")
        teacher_ckpt = load_checkpoint(resolve_repo_path(teacher_checkpoint_path, repo_root), map_location=device)
        teacher_model, teacher_input_mode = _make_model_from_checkpoint(teacher_ckpt, device)
        distill_weight = float(distill_cfg.get("distill_weight", 0.3))
        gt_weight = float(distill_cfg.get("gt_weight", 1.0))
        print(
            f"Distillation enabled teacher_input_mode={teacher_input_mode} "
            f"gt_weight={gt_weight} distill_weight={distill_weight}"
        )

        if bool(distill_cfg.get("encoder_transfer", True)):
            prefixes = ("enc", "bottleneck") if bool(distill_cfg.get("encoder_only", True)) else None
            skipped = _load_compatible_model_state(model, teacher_ckpt, prefixes=prefixes)
            print(f"Initialized student from teacher checkpoint: {teacher_checkpoint_path}")
            if skipped:
                print("Skipped teacher layers:")
                for layer in skipped:
                    print(f"  - {layer}")

    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    epochs = int(config["training"]["epochs"])
    num_visualizations = int(config["training"].get("save_visualizations", 4))
    start_epoch = 0

    if resume_checkpoint is not None:
        ckpt = load_checkpoint(resolve_repo_path(resume_checkpoint, repo_root), map_location=device)
        skipped = _load_compatible_model_state(model, ckpt)
        if skipped:
            print("Skipped checkpoint layers:")
            for layer in skipped:
                print(f"  - {layer}")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val = float(ckpt.get("best_val_loss", best_val))

        if start_epoch > 0 and history_path.exists():
            with history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            history = history[:start_epoch]

        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch + 1}/{epochs}")
        else:
            print("Loaded compatible checkpoint weights; starting fine-tuning from epoch 1")

    pretrained_checkpoint = config["training"].get("pretrained_checkpoint")
    if resume_checkpoint is None and pretrained_checkpoint:
        ckpt = load_checkpoint(resolve_repo_path(pretrained_checkpoint, repo_root), map_location=device)
        skipped = _load_compatible_model_state(model, ckpt)
        print(f"Loaded compatible pretrained weights from {pretrained_checkpoint}")
        if skipped:
            print("Skipped checkpoint layers:")
            for layer in skipped:
                print(f"  - {layer}")

    if start_epoch >= epochs:
        print(f"Checkpoint already covers {start_epoch} epochs. Configured epochs: {epochs}. Nothing to do.")
        return

    for epoch in range(start_epoch, epochs):
        freeze_encoder_epochs = int(config["training"].get("freeze_encoder_epochs", 0))
        encoder_frozen = epoch < freeze_encoder_epochs
        _set_encoder_frozen(model, encoder_frozen)
        if epoch == start_epoch and encoder_frozen:
            print(f"Freezing encoder for first {freeze_encoder_epochs} epoch(s)")
        if epoch == freeze_encoder_epochs and freeze_encoder_epochs > 0:
            print("Unfreezing encoder")

        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            teacher_model=teacher_model,
            teacher_input_mode=teacher_input_mode,
            distill_weight=distill_weight,
            gt_weight=gt_weight,
        )
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

        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Training finished. Best validation loss: {best_val:.4f}")
