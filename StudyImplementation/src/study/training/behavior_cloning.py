"""Behavior cloning trainer for the bracket sketch policy."""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from study.data.dataset import BracketSketchDataset, collate_fn
from study.models.policy import SketchPolicy
from study.utils.random import set_global_seed


def train_behavior_cloning(config: Dict[str, Any], resume_checkpoint: Optional[pathlib.Path] = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_cfg = config["experiment"]
    data_cfg = config["data"]
    set_global_seed(experiment_cfg["seed"])
    train_path = pathlib.Path(data_cfg["processed_dir"]) / "train.jsonl"
    val_path = pathlib.Path(data_cfg["processed_dir"]) / "val.jsonl"

    train_dataset = BracketSketchDataset(train_path)
    val_dataset = BracketSketchDataset(val_path)

    param_dim = train_dataset[0]["primitive_params"].shape[-1]
    primitive_vocab = config["model"]["primitive_vocab_size"]
    constraint_vocab = config["model"]["constraint_vocab_size"]
    hidden_dim = config["model"]["hidden_dim"]

    model_kwargs = {
        "hidden_dim": hidden_dim,
        "primitive_vocab": primitive_vocab,
        "constraint_vocab": constraint_vocab,
        "param_dim": param_dim,
    }

    model = SketchPolicy(**model_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"Resumed training from epoch {start_epoch}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
    )

    output_dir = pathlib.Path(experiment_cfg["output_dir"]) / "bc"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = config["training"]["max_epochs_bc"]
    log_every = config["training"]["log_every_steps"]
    global_step = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            targets = _extract_targets(batch)
            outputs = model(
                primitive_types=batch["primitive_types"].to(device),
                constraint_types=batch["constraint_types"].to(device),
                span_mm=batch["requirement_span"].to(device),
            )
            loss = _bc_loss(outputs, targets, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            if global_step % log_every == 0:
                print(
                    f"[Epoch {epoch} Step {global_step}] "
                    f"Loss={loss.item():.4f}"
                )

        epoch_time = time.time() - start_time
        val_loss = _evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch} completed in {epoch_time:.1f}s "
            f"(train loss={epoch_loss / len(train_loader):.4f}, val loss={val_loss:.4f})"
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        _save_checkpoint(
            output_dir / f"epoch_{epoch}.ckpt",
            model,
            optimizer,
            epoch,
            best_val_loss,
            model_kwargs,
        )
        if is_best:
            _save_checkpoint(
                output_dir / "best.ckpt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                model_kwargs,
            )


def _extract_targets(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    primitive_types = batch["primitive_types"]
    primitive_params = batch["primitive_params"]

    valid_lengths = (primitive_types >= 0).sum(dim=1) - 1
    batch_size = primitive_types.shape[0]
    idx = torch.arange(batch_size)
    target_types = primitive_types[idx, valid_lengths].clone()
    target_params = primitive_params[idx, valid_lengths].clone()

    return {"types": target_types, "params": target_params}


def _bc_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    logits = outputs["logits"]
    mu = outputs["mu"]
    log_sigma = outputs["log_sigma"]

    target_types = targets["types"].to(device)
    target_params = targets["params"].to(device)

    ce_loss = F.cross_entropy(logits, target_types)
    sigma = torch.exp(log_sigma)
    mse = torch.mean(((mu - target_params) / sigma) ** 2 + 2 * log_sigma)
    return ce_loss + 0.1 * mse


@torch.no_grad()
def _evaluate(model: SketchPolicy, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in dataloader:
        targets = _extract_targets(batch)
        outputs = model(
            primitive_types=batch["primitive_types"].to(device),
            constraint_types=batch["constraint_types"].to(device),
            span_mm=batch["requirement_span"].to(device),
        )
        loss = _bc_loss(outputs, targets, device)
        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def _save_checkpoint(
    path: pathlib.Path,
    model: SketchPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    model_kwargs: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_kwargs": model_kwargs,
        },
        path,
    )
