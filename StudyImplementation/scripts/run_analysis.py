#!/usr/bin/env python
"""Run comprehensive analysis (calibration, risk sweep, confusion, qualitative) on a checkpoint."""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from study.analysis.runner import run_full_analysis  # noqa: E402


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
        help="Checkpoint to analyze (typically artifacts/cql/best.ckpt).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "analysis",
        help="Directory to write analysis reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    summary = run_full_analysis(cfg, checkpoint_path=args.checkpoint, output_dir=args.output)
    print("Analysis summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
