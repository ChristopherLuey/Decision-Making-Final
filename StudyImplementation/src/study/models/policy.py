"""Neural architectures for bracket sketch policy and value estimation."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphStateEncoder(nn.Module):
    """Encodes primitive and constraint sequences into a latent state."""

    def __init__(self, hidden_dim: int, primitive_vocab: int, constraint_vocab: int) -> None:
        super().__init__()
        self.primitive_embedding = nn.Embedding(primitive_vocab + 1, hidden_dim, padding_idx=primitive_vocab)
        self.constraint_embedding = nn.Embedding(constraint_vocab + 1, hidden_dim, padding_idx=constraint_vocab)
        self.primitive_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.constraint_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        primitive_types: torch.Tensor,
        constraint_types: torch.Tensor,
    ) -> torch.Tensor:
        primitive_embed = self.primitive_embedding(primitive_types)
        constraint_embed = self.constraint_embedding(constraint_types)

        prim_packed, _ = self.primitive_gru(primitive_embed)
        cons_packed, _ = self.constraint_gru(constraint_embed)

        prim_context = prim_packed.mean(dim=1)
        cons_context = cons_packed.mean(dim=1)
        pooled = torch.cat([prim_context, cons_context], dim=-1)
        return self.proj(pooled)


class RequirementEncoder(nn.Module):
    """Encodes requirement tuples using simple feed-forward networks."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, span_mm: torch.Tensor) -> torch.Tensor:
        # [batch]
        features = torch.stack(
            [
                span_mm,
                torch.sqrt(torch.clamp(span_mm, min=1e-6)),
                torch.log1p(span_mm),
                torch.ones_like(span_mm),
            ],
            dim=-1,
        )
        return self.mlp(features)


class PolicyHead(nn.Module):
    """Outputs primitive type logits and parameter distributions."""

    def __init__(self, hidden_dim: int, primitive_vocab: int, param_dim: int) -> None:
        super().__init__()
        self.type_head = nn.Linear(hidden_dim, primitive_vocab)
        self.param_mu = nn.Linear(hidden_dim, param_dim)
        self.param_log_sigma = nn.Linear(hidden_dim, param_dim)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.type_head(hidden)
        mu = self.param_mu(hidden)
        log_sigma = self.param_log_sigma(hidden).clamp(-5.0, 2.0)
        return {"logits": logits, "mu": mu, "log_sigma": log_sigma}


class ValueHead(nn.Module):
    """Predicts action value for conservative Q-learning."""

    def __init__(self, hidden_dim: int, primitive_vocab: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(primitive_vocab, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: torch.Tensor, action_types: torch.Tensor) -> torch.Tensor:
        action_embed = self.embedding(action_types)
        x = torch.cat([hidden, action_embed], dim=-1)
        return self.net(x).squeeze(-1)


class SketchPolicy(nn.Module):
    """Top-level module that combines encoders with policy/value heads."""

    def __init__(
        self,
        hidden_dim: int,
        primitive_vocab: int,
        constraint_vocab: int,
        param_dim: int,
    ) -> None:
        super().__init__()
        self.graph_encoder = GraphStateEncoder(hidden_dim, primitive_vocab, constraint_vocab)
        self.req_encoder = RequirementEncoder(hidden_dim)
        self.policy_head = PolicyHead(hidden_dim, primitive_vocab, param_dim)
        self.value_head = ValueHead(hidden_dim, primitive_vocab)
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        primitive_types: torch.Tensor,
        constraint_types: torch.Tensor,
        span_mm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        latent_state = self.graph_encoder(primitive_types, constraint_types)
        req_state = self.req_encoder(span_mm)
        fused = self.dropout(latent_state + req_state)
        policy_outputs = self.policy_head(fused)
        policy_outputs["latent_state"] = fused
        return policy_outputs

    def q_values(
        self,
        primitive_types: torch.Tensor,
        constraint_types: torch.Tensor,
        span_mm: torch.Tensor,
        action_types: torch.Tensor,
    ) -> torch.Tensor:
        latent_state = self.graph_encoder(primitive_types, constraint_types)
        req_state = self.req_encoder(span_mm)
        fused = latent_state + req_state
        return self.value_head(fused, action_types)
