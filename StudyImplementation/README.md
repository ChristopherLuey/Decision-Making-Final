# Automating Parametric CAD Sketch Synthesis – Implementation Plan

This folder provides the code scaffold and documentation needed to reproduce the two-week study described in the proposal. The project automates early-stage CAD sketch construction for planar brackets using the SketchGraphs subset of the Fusion 360 Gallery dataset and a belief-aware sequential decision-making pipeline.

## Folder Layout

- `configs/`: YAML configuration files for dataset curation, model, and training hyperparameters.
- `data/`: Placeholder for raw and processed SketchGraphs data artifacts (ignored by Git via `.gitignore`).
- `notebooks/`: Optional exploratory analysis notebooks (empty scaffold).
- `scripts/`: Command-line entry points that wrap the Python modules in `src/`.
- `src/`: Python package (`study`) containing reusable modules for data processing, modeling, training, and evaluation.
- `tests/`: Lightweight unit and smoke tests to sanity-check preprocessing and model wiring.

## Quickstart

1. **Install**: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. **Download data**: `bash scripts/download_sketchgraphs.sh data/raw`
3. **Curate bracket subset**: `python scripts/run_preprocessing.py --config configs/bracket_study.yaml`
4. **Train behavior cloning baseline**: `python scripts/train_bc.py --config configs/bracket_study.yaml`
5. **Fine-tune with CQL**: `python scripts/train_cql.py --config configs/bracket_study.yaml --bc-checkpoint artifacts/bc/latest.ckpt`
6. **Evaluate**: `python scripts/evaluate_policy.py --config configs/bracket_study.yaml --checkpoint artifacts/cql/best.ckpt`

Each script supports `--help` to list options. Logs and artifacts are written under the `artifacts/` directory created on first run.

## Timeline Alignment

- **Days 1–4**: Focus on `scripts/download_sketchgraphs.sh`, `scripts/run_preprocessing.py`, and the `study.data` package. Deliverables include summary statistics saved in `artifacts/data_report/`.
- **Days 5–9**: Implement and validate the behavior cloning pipeline (`study.models`, `study.training.behavior_cloning`). Calibration plots are stored in `artifacts/analysis/`.
- **Days 10–14**: Enable conservative Q-learning and uncertainty-aware planning (`study.training.conservative_q`, `study.evaluation.rollout`). Reporting utilities produce CSV/JSON and optional LaTeX tables.

## Notes

- The SketchGraphs dataset is large (~10 GB compressed). Ensure sufficient disk space before running the download script.
- Configuration values in `configs/bracket_study.yaml` are tuned for a workstation-class GPU but can be reduced for CPU-only experiments.
- The code uses PyTorch Geometric; refer to the official installation guide if you encounter CUDA-related dependency issues.

## References

- Kochenderfer et al., *Algorithms for Decision Making*, 2022.
- Autodesk Research, *Fusion 360 Gallery Dataset*, 2021.
- Autodesk Research, *SketchGraphs: A Large-Scale Dataset for Modeling Relational Geometry in CAD*, 2020.
