#!/usr/bin/env python
"""Evaluate a trained policy on held-out specification tuples."""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from study.evaluation.rollout import evaluate_policy_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=REPO_ROOT / "configs" / "bracket_study.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional directory for evaluation reports (defaults to artifacts/eval).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    evaluate_policy_checkpoint(cfg, checkpoint_path=args.checkpoint, output_dir=args.output)


if __name__ == "__main__":
    main()
