"""Evaluation utilities for trained sketch policies."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from study.data.dataset import BracketSketchDataset, collate_fn
from study.models.policy import SketchPolicy


def evaluate_policy_checkpoint(
    config: Dict[str, Any],
    checkpoint_path: pathlib.Path,
    output_dir: Optional[pathlib.Path] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    eval_path = pathlib.Path(data_cfg["processed_dir"]) / "test.jsonl"
    dataset = BracketSketchDataset(eval_path)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SketchPolicy(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    metrics = _compute_metrics(model, dataloader, device, config["evaluation"]["risk_lambda"])

    if output_dir is None:
        output_dir = pathlib.Path(config["experiment"]["output_dir"]) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved evaluation metrics to {output_dir / 'metrics.json'}")
    return metrics


@torch.no_grad()
def _compute_metrics(
    model: SketchPolicy,
    dataloader: DataLoader,
    device: torch.device,
    risk_lambda: float,
) -> Dict[str, float]:
    correct = 0
    total = 0
    risks: list[float] = []
    for batch in dataloader:
        primitive_types = batch["primitive_types"].to(device)
        constraint_types = batch["constraint_types"].to(device)
        span_mm = batch["requirement_span"].to(device)

        outputs = model(primitive_types, constraint_types, span_mm)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        targets = _extract_targets(batch, device)
        correct += (preds == targets["types"]).sum().item()
        total += preds.shape[0]

        logsumexp = torch.logsumexp(logits, dim=-1)
        data_term = logits.gather(1, targets["types"].unsqueeze(-1)).squeeze(-1)
        risk = (logsumexp - data_term - risk_lambda).clamp(min=0.0).mean().item()
        risks.append(risk)

    accuracy = correct / max(total, 1)
    avg_risk = sum(risks) / max(len(risks), 1)
    return {
        "accuracy": float(accuracy),
        "avg_risk_penalty": float(avg_risk),
        "samples": int(total),
    }


def _extract_targets(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    primitive_types = batch["primitive_types"].to(device)
    primitive_params = batch["primitive_params"].to(device)
    valid_lengths = (primitive_types >= 0).sum(dim=1) - 1
    idx = torch.arange(primitive_types.shape[0], device=device)
    target_types = primitive_types[idx, valid_lengths].clone()
    target_params = primitive_params[idx, valid_lengths].clone()
    return {"types": target_types, "params": target_params}
