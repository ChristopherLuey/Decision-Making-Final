"""Dataset and collation utilities for bracket sketch sequences."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class BracketSketchDataset(Dataset):
    """Lightweight dataset backed by JSONL files produced by preprocessing."""

    def __init__(self, data_path: pathlib.Path | str) -> None:
        self.data_path = pathlib.Path(data_path)
        self.records: List[Dict[str, Any]] = []
        with self.data_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                self.records.append(json.loads(line))

        meta_path = self.data_path.parent / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file missing at {meta_path}")
        with meta_path.open("r", encoding="utf-8") as fp:
            self.meta = json.load(fp)

        self.param_dim: int = int(self.meta["param_dim"])
        self.primitive_vocab_size: int = int(self.meta["primitive_vocab_size"])
        self.constraint_vocab_size: int = int(self.meta["constraint_vocab_size"])
        self.primitive_pad_id: int = int(self.meta["primitive_pad_id"])
        self.constraint_pad_id: int = int(self.meta["constraint_pad_id"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        return {
            "sketch_id": rec.get("sketch_id"),
            "primitive_types": torch.tensor(rec["primitive_types"], dtype=torch.long),
            "primitive_params": torch.tensor(rec["primitive_params"], dtype=torch.float32),
            "constraint_types": torch.tensor(rec["constraint_types"], dtype=torch.long),
            "requirement_span": torch.tensor(rec["requirement_span"], dtype=torch.float32),
            "target_type": torch.tensor(rec["target_type"], dtype=torch.long),
            "target_params": torch.tensor(rec["target_params"], dtype=torch.float32),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_prim_len = max(item["primitive_types"].shape[0] for item in batch)
        max_cons_len = max((item["constraint_types"].shape[0] for item in batch), default=1)
        max_cons_len = max(max_cons_len, 1)

        prim_types = torch.full(
            (batch_size, max_prim_len), self.primitive_pad_id, dtype=torch.long
        )
        prim_params = torch.zeros((batch_size, max_prim_len, self.param_dim), dtype=torch.float32)
        cons_types = torch.full(
            (batch_size, max_cons_len), self.constraint_pad_id, dtype=torch.long
        )

        for i, item in enumerate(batch):
            plen = item["primitive_types"].shape[0]
            clen = item["constraint_types"].shape[0]
            prim_types[i, :plen] = item["primitive_types"]
            prim_params[i, :plen, : item["primitive_params"].shape[1]] = item["primitive_params"]
            cons_types[i, :clen] = item["constraint_types"]

        return {
            "primitive_types": prim_types,
            "primitive_params": prim_params,
            "constraint_types": cons_types,
            "requirement_span": torch.stack([item["requirement_span"] for item in batch]),
            "target_type": torch.stack([item["target_type"] for item in batch]),
            "target_params": torch.stack([item["target_params"] for item in batch]),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Best-effort collation when a dataset instance is not available."""
    if not batch:
        return {}

    param_dim = batch[0]["primitive_params"].shape[-1]
    primitive_pad = int(max(item["primitive_types"].max().item() for item in batch)) + 1
    constraint_pad = int(max((item["constraint_types"].max().item() for item in batch), default=0)) + 1
    max_prim_len = max(item["primitive_types"].shape[0] for item in batch)
    max_cons_len = max((item["constraint_types"].shape[0] for item in batch), default=1)

    prim_types = torch.full((len(batch), max_prim_len), primitive_pad, dtype=torch.long)
    prim_params = torch.zeros((len(batch), max_prim_len, param_dim), dtype=torch.float32)
    cons_types = torch.full((len(batch), max_cons_len), constraint_pad, dtype=torch.long)

    for i, item in enumerate(batch):
        plen = item["primitive_types"].shape[0]
        clen = item["constraint_types"].shape[0]
        prim_types[i, :plen] = item["primitive_types"]
        prim_params[i, :plen, : item["primitive_params"].shape[1]] = item["primitive_params"]
        cons_types[i, :clen] = item["constraint_types"]

    return {
        "primitive_types": prim_types,
        "primitive_params": prim_params,
        "constraint_types": cons_types,
        "requirement_span": torch.stack([item["requirement_span"] for item in batch]),
        "target_type": torch.stack([item["target_type"] for item in batch]),
        "target_params": torch.stack([item["target_params"] for item in batch]),
    }


__all__ = ["BracketSketchDataset", "collate_fn"]
