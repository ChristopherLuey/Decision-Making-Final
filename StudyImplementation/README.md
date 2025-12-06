# Automating Parametric CAD Sketch Synthesis

This project implements a belief-aware sequential decision-making pipeline for automating early-stage CAD sketch construction, specifically focusing on planar brackets using the SketchGraphs dataset. It employs Behavior Cloning (BC) as a baseline and Conservative Q-Learning (CQL) for robust policy learning.

## Setup

1.  **Environment Setup**:
    Create and activate a virtual environment, then install dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

    *Note: The code relies on PyTorch Geometric. If you encounter CUDA/installation issues, refer to the [official PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).*

## Usage Pipeline

The project is structured as a pipeline of scripts located in `scripts/`. Run them in the following order to reproduce the results.

### 1. Data Preparation

Download the SketchGraphs dataset and process it to curate the bracket subset.

```bash
# Download raw data (~10GB compressed)
bash scripts/download_sketchgraphs.sh data/raw

# Process and curate data into training splits
python scripts/run_preprocessing.py --config configs/bracket_study.yaml
```
*Outputs: `data/processed/` containing train/val/test splits and metadata.*

### 2. Training

First, train the Behavior Cloning (BC) baseline, then fine-tune using Conservative Q-Learning (CQL).

```bash
# Train Behavior Cloning (BC) model
python scripts/train_bc.py --config configs/bracket_study.yaml
```
*Outputs: Checkpoints in `artifacts/bc/`.*

```bash
# Train CQL model (initialized from BC checkpoint)
# Replace 'artifacts/bc/best.ckpt' with your actual best checkpoint path if different
python scripts/train_cql.py --config configs/bracket_study.yaml --bc-checkpoint artifacts/bc/best.ckpt
```
*Outputs: Checkpoints in `artifacts/cql/`.*

### 3. Evaluation & Analysis

Evaluate the trained policies and generate figures for the report.

```bash
# Evaluate the best CQL policy
python scripts/evaluate_policy.py --config configs/bracket_study.yaml --checkpoint artifacts/cql/best.ckpt
```

```bash
# Generate analysis figures and tables
python scripts/run_analysis.py
```

```bash
# Create final paper figures
python scripts/make_paper_figures.py
```

*Outputs: Metrics and figures in `artifacts/eval/`, `artifacts/analysis/`, and `Report/figures/`.*

## Configuration

The main configuration file is `configs/bracket_study.yaml`. You can modify training hyperparameters (batch size, learning rate), model architecture details, and data paths here.

## Outputs

- **`artifacts/`**: Contains training logs, checkpoints, and analysis outputs.
- **`data/`**: Contains raw and processed datasets.

## References

- Autodesk Research, *Fusion 360 Gallery Dataset*, 2021.
- Autodesk Research, *SketchGraphs: A Large-Scale Dataset for Modeling Relational Geometry in CAD*, 2020.
