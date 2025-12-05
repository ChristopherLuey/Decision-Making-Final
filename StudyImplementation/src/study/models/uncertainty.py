"""Uncertainty quantification utilities for policy ensembles."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .policy import SketchPolicy


class PolicyEnsemble:
    """Maintains an ensemble of policies for bootstrap-based uncertainty."""

    def __init__(self, members: List[SketchPolicy]) -> None:
        self.members = members

    @classmethod
    def from_checkpoint(cls, checkpoint_paths: List[str], device: torch.device) -> "PolicyEnsemble":
        members: List[SketchPolicy] = []
        for path in checkpoint_paths:
            checkpoint = torch.load(path, map_location=device)
            model = SketchPolicy(**checkpoint["model_kwargs"])
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            model.eval()
            members.append(model)
        return cls(members)

    def forward_pass(
        self,
        primitive_types: torch.Tensor,
        constraint_types: torch.Tensor,
        span_mm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = []
        mus = []
        for member in self.members:
            outputs = member(primitive_types, constraint_types, span_mm)
            logits.append(outputs["logits"])
            mus.append(outputs["mu"])
        stacked_logits = torch.stack(logits, dim=0)
        stacked_mu = torch.stack(mus, dim=0)
        return stacked_logits, stacked_mu

    def compute_risk_adjusted_scores(
        self,
        primitive_types: torch.Tensor,
        constraint_types: torch.Tensor,
        span_mm: torch.Tensor,
        risk_lambda: float,
    ) -> Dict[str, torch.Tensor]:
        logits, _ = self.forward_pass(primitive_types, constraint_types, span_mm)
        mean_logits = logits.mean(dim=0)
        var_logits = logits.var(dim=0)
        adjusted = mean_logits - risk_lambda * torch.sqrt(var_logits + 1e-8)
        return {"mean_logits": mean_logits, "variance": var_logits, "adjusted_logits": adjusted}
