
import sys
import pathlib
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add src to path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from study.models.policy import SketchPolicy
from study.data.dataset import BracketSketchDataset

# Set style for IEEE papers
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "lines.linewidth": 1.5
})

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SketchPolicy(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model

def get_logits_and_targets(model, dataloader, device):
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            primitive_types = batch["primitive_types"].to(device)
            constraint_types = batch["constraint_types"].to(device)
            span_mm = batch["requirement_span"].to(device)
            target_types = batch["target_type"].to(device)
            
            outputs = model(primitive_types, constraint_types, span_mm)
            all_logits.append(outputs["logits"].cpu())
            all_targets.append(target_types.cpu())
            
    return torch.cat(all_logits), torch.cat(all_targets)

def compute_calibration(logits, targets, n_bins=10):
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(targets)
    
    bins = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy = accuracies[in_bin].float().mean()
            avg_confidence = confidences[in_bin].mean()
            
            bin_accs.append(accuracy.item())
            bin_confs.append(avg_confidence.item())
            bin_sizes.append(prop_in_bin.item())
            
            ece += torch.abs(avg_confidence - accuracy) * prop_in_bin
            
    return bin_accs, bin_confs, ece.item()

def plot_reliability_comparison(bc_logits, bc_targets, cql_logits, cql_targets, output_path):
    bc_accs, bc_confs, bc_ece = compute_calibration(bc_logits, bc_targets)
    cql_accs, cql_confs, cql_ece = compute_calibration(cql_logits, cql_targets)
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k:", label="Perfect")
    
    # BC Curve
    ax.plot(bc_confs, bc_accs, "s-", label=f"BC (ECE={bc_ece:.2f})", color="gray", alpha=0.7, markersize=4)
    
    # CQL Curve
    ax.plot(cql_confs, cql_accs, "o-", label=f"CQL (ECE={cql_ece:.2f})", color="#d62728", linewidth=2, markersize=5)
    
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved reliability comparison to {output_path}")

def plot_training_curves(bc_csv, cql_csv, output_path):
    bc_df = pd.read_csv(bc_csv)
    cql_df = pd.read_csv(cql_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # BC Plot
    ax1.plot(bc_df['epoch'], bc_df['train_loss'], label='Train', color='tab:blue')
    ax1.plot(bc_df['epoch'], bc_df['val_loss'], label='Val', color='tab:orange', linestyle='--')
    ax1.set_title("Behavior Cloning (Stage 1)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CQL Plot
    ax2.plot(cql_df['epoch'], cql_df['train_loss'], label='Train', color='tab:blue')
    ax2.plot(cql_df['epoch'], cql_df['val_loss'], label='Val', color='tab:orange', linestyle='--')
    ax2.set_title("CQL Fine-tuning (Stage 2)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("CQL Loss (Conservative)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved training curves to {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    config_path = REPO_ROOT / "configs" / "bracket_study.yaml"
    test_data_path = REPO_ROOT / "data" / "processed" / "test.jsonl"
    bc_ckpt = REPO_ROOT / "artifacts" / "bc" / "best.ckpt"
    cql_ckpt = REPO_ROOT / "artifacts" / "cql" / "best.ckpt"
    bc_metrics = REPO_ROOT / "artifacts" / "bc" / "metrics.csv"
    cql_metrics = REPO_ROOT / "artifacts" / "cql" / "metrics.csv"
    
    output_dir = REPO_ROOT.parent / "Report" / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # Load Config & Data
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    dataset = BracketSketchDataset(test_data_path)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)
    
    # Generate Training Curves
    plot_training_curves(bc_metrics, cql_metrics, output_dir / "training_curves_combined.png")
    
    # Generate Reliability Comparison
    print("Generating reliability comparison...")
    bc_model = load_model(bc_ckpt, device)
    bc_logits, bc_targets = get_logits_and_targets(bc_model, dataloader, device)
    
    cql_model = load_model(cql_ckpt, device)
    cql_logits, cql_targets = get_logits_and_targets(cql_model, dataloader, device)
    
    plot_reliability_comparison(bc_logits, bc_targets, cql_logits, cql_targets, output_dir / "reliability_comparison.png")

if __name__ == "__main__":
    main()

