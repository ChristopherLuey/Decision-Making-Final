"""Conservative Q-learning style fine-tuning."""

from __future__ import annotations

import copy
import pathlib
import time
from typing import Any, Dict

import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from study.data.dataset import BracketSketchDataset
from study.models.policy import SketchPolicy
from study.utils.random import set_global_seed


def train_conservative_q(config: Dict[str, Any], bc_checkpoint: pathlib.Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    experiment_cfg = config["experiment"]
    set_global_seed(experiment_cfg["seed"])
    train_path = pathlib.Path(data_cfg["processed_dir"]) / "train.jsonl"
    val_path = pathlib.Path(data_cfg["processed_dir"]) / "val.jsonl"

    train_dataset = BracketSketchDataset(train_path)
    val_dataset = BracketSketchDataset(val_path)
    param_dim = train_dataset.param_dim

    model_kwargs = {
        "hidden_dim": config["model"]["hidden_dim"],
        "primitive_vocab": train_dataset.primitive_vocab_size,
        "constraint_vocab": train_dataset.constraint_vocab_size,
        "param_dim": param_dim,
    }

    model = SketchPolicy(**model_kwargs).to(device)
    checkpoint = torch.load(bc_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    target_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=val_dataset.collate_fn,
    )

    output_dir = pathlib.Path(experiment_cfg["output_dir"]) / "cql"
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha = config["cql"]["alpha"]
    target_interval = config["cql"]["target_update_interval"]
    discount = config["cql"]["discount"]

    global_step = 0
    best_val_loss = float("inf")
    history: list[Dict[str, float]] = []
    for epoch in range(config["training"]["max_epochs_cql"]):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = _cql_loss(
                model,
                target_model,
                batch,
                device,
                alpha,
                discount,
                padding_idx=train_dataset.primitive_pad_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            if global_step % target_interval == 0:
                target_model.load_state_dict(model.state_dict())

        val_loss = _evaluate(model, val_loader, device, alpha, discount)
        print(
            f"[CQL] Epoch {epoch} loss={epoch_loss / len(train_loader):.4f} "
            f"val={val_loss:.4f} time={time.time() - start_time:.1f}s"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(epoch_loss / len(train_loader)),
                "val_loss": float(val_loss),
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(output_dir / "best.ckpt", model, optimizer, epoch, best_val_loss, model_kwargs)
        _save_checkpoint(output_dir / f"epoch_{epoch}.ckpt", model, optimizer, epoch, best_val_loss, model_kwargs)
    _save_history(output_dir, history)


def _cql_loss(
    model: SketchPolicy,
    target_model: SketchPolicy,
    batch: Dict[str, Any],
    device: torch.device,
    alpha: float,
    discount: float,
    padding_idx: int,
) -> torch.Tensor:
    primitive_types = batch["primitive_types"].to(device)
    constraint_types = batch["constraint_types"].to(device)
    span_mm = batch["requirement_span"].to(device)

    targets = _extract_targets(batch, device, padding_idx)
    outputs = model(primitive_types, constraint_types, span_mm)
    logits = outputs["logits"]
    mu = outputs["mu"]
    log_sigma = outputs["log_sigma"]
    latent = outputs["latent_state"]

    ce_loss = F.cross_entropy(logits, targets["types"])

    sigma = torch.exp(log_sigma)
    gaussian_loss = torch.mean(((mu - targets["params"]) / sigma) ** 2 + 2 * log_sigma)

    # Conservative penalty.
    logsumexp = torch.logsumexp(logits, dim=-1).mean()
    data_term = logits.gather(1, targets["types"].unsqueeze(-1)).mean()
    cql_penalty = logsumexp - data_term

    # Simple bootstrapped value loss using target network.
    with torch.no_grad():
        target_outputs = target_model(primitive_types, constraint_types, span_mm)
        target_q = target_model.q_values(
            primitive_types, constraint_types, span_mm, targets["types"]
        )
    current_q = model.q_values(primitive_types, constraint_types, span_mm, targets["types"])
    value_loss = F.mse_loss(current_q, discount * target_q)

    return ce_loss + 0.1 * gaussian_loss + alpha * cql_penalty + value_loss


@torch.no_grad()
def _evaluate(
    model: SketchPolicy,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float,
    discount: float,
) -> float:
    losses: list[float] = []
    for batch in dataloader:
        padding_idx = getattr(dataloader.dataset, "primitive_pad_id", model.policy_head.type_head.out_features)
        loss = _cql_loss(model, model, batch, device, alpha, discount, padding_idx=padding_idx)
        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def _extract_targets(batch: Dict[str, Any], device: torch.device, padding_idx: int) -> Dict[str, torch.Tensor]:
    if "target_type" in batch:
        return {
            "types": batch["target_type"].to(device),
            "params": batch["target_params"].to(device),
        }
    primitive_types = batch["primitive_types"].to(device)
    primitive_params = batch["primitive_params"].to(device)
    valid_lengths = (primitive_types != padding_idx).sum(dim=1) - 1
    idx = torch.arange(primitive_types.shape[0], device=device)
    target_types = primitive_types[idx, valid_lengths].clone()
    target_params = primitive_params[idx, valid_lengths].clone()
    return {"types": target_types, "params": target_params}


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


def _save_history(output_dir: pathlib.Path, history: list[Dict[str, float]]) -> None:
    if not history:
        return
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    ax.plot(epochs, train, label="train")
    ax.plot(epochs, val, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close(fig)
