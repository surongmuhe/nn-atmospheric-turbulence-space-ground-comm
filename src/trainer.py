import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def build_loaders(x_train, y_train, x_val, y_val, batch_size=64, num_workers=0, shuffle_train=True):
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def run_inference(model, x, device, batch_size=256):
    model.eval()
    if len(x) == 0:
        return np.empty((0,), dtype=np.float32)

    preds = []
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
            del xb
    return np.concatenate(preds, axis=0)


def train_single_model(model_name, model, train_loader, val_loader, cfg, logger, device, wandb_run=None):
    tcfg = cfg["train"]
    criterion = nn.MSELoss()
    loss_cfg = tcfg.get("loss", {})
    lr_scale_map = tcfg.get("lr_scale_map", {})
    model_lr = float(tcfg["learning_rate"]) * float(lr_scale_map.get(model_name, 1.0))
    optimizer_name = str(tcfg.get("optimizer", "adam")).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_lr,
            weight_decay=tcfg["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_lr,
            weight_decay=tcfg["weight_decay"],
        )
    scheduler = None
    scheduler_cfg = tcfg.get("scheduler", {}) or {}
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 2)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
        )
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 1.0))
    model = model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    logger.info(
        "Start training model=%s (optimizer=%s, lr=%.6g, grad_clip_norm=%.3f, scheduler=%s)",
        model_name,
        optimizer_name,
        model_lr,
        grad_clip_norm,
        scheduler_type,
    )
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        train_losses = []
        bad_train_batches = 0
        show_progress = bool(tcfg.get("show_progress", True))
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch}/{tcfg['epochs']}", leave=False, disable=not show_progress)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if hasattr(model, "compute_loss"):
                pred, loss, loss_stats = model.compute_loss(xb, yb, loss_cfg)
            else:
                pred = model(xb)
                loss = criterion(pred, yb)
                loss_stats = {}
            if not torch.isfinite(loss):
                bad_train_batches += 1
                pbar.set_postfix(train_loss="nan")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            train_losses.append(loss.item())
            postfix = {"train_loss": f"{loss.item():.5f}"}
            if "risk_loss" in loss_stats:
                postfix["risk"] = f"{loss_stats['risk_loss']:.4f}"
            pbar.set_postfix(**postfix)

        model.eval()
        val_losses = []
        bad_val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v = criterion(model(xb), yb)
                if not torch.isfinite(v):
                    bad_val_batches += 1
                    continue
                val_losses.append(v.item())

        train_mean = float(np.mean(train_losses)) if train_losses else float("inf")
        val_mean = float(np.mean(val_losses)) if val_losses else float("inf")
        logger.info(
            "model=%s epoch=%d train_loss=%.6f val_loss=%.6f bad_train_batches=%d bad_val_batches=%d",
            model_name,
            epoch,
            train_mean,
            val_mean,
            bad_train_batches,
            bad_val_batches,
        )
        if scheduler is not None:
            scheduler.step(val_mean)

        if val_mean < best_val:
            best_val = val_mean
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["early_stop_patience"]:
                logger.info("Early stop triggered for model=%s at epoch=%d", model_name, epoch)
                break

    model.load_state_dict(best_state)
    return model
