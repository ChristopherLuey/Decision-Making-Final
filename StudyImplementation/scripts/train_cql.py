#!/usr/bin/env python
"""Fine-tune the policy using conservative Q-learning."""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from study.training.conservative_q import train_conservative_q  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=REPO_ROOT / "configs" / "bracket_study.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to a behavior cloning checkpoint for initialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    train_conservative_q(cfg, bc_checkpoint=args.bc_checkpoint)


if __name__ == "__main__":
    main()
