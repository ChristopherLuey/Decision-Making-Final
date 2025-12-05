"""End-to-end analysis pipeline for trained sketch policies."""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from study.data.dataset import BracketSketchDataset
from study.data.preprocess import _entity_to_primitive, _import_sketchgraphs, _primitive_parameters
from study.models.policy import SketchPolicy


def run_full_analysis(
    config: Dict[str, Any],
    checkpoint_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> Dict[str, Any]:
    """Compute calibration, risk sweeps, confusion, and qualitative rollouts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    eval_path = pathlib.Path(data_cfg["processed_dir"]) / "test.jsonl"
    dataset = BracketSketchDataset(eval_path)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=dataset.collate_fn,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SketchPolicy(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    outputs = _collect_outputs(model, dataloader, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    calib = _plot_reliability(outputs, output_dir)
    risk_curve = _plot_risk_sweep(outputs, output_dir)
    confusion = _plot_confusion(outputs, dataset, output_dir)
    qualitative = _save_qualitative(outputs, dataset, output_dir)
    gallery = _render_gallery(config, dataset, output_dir)
    progress = _render_progressions(config, dataset, model, output_dir)

    summary = {
        "n_samples": len(outputs["targets"]),
        "accuracy": float(np.mean(outputs["preds"] == outputs["targets"])),
        "ece": float(calib["ece"]),
        "risk_curve": risk_curve["risk_curve"],
        "confusion_path": str(confusion["path"]),
        "reliability_path": str(calib["path"]),
        "risk_path": str(risk_curve["path"]),
        "qualitative_path": str(qualitative["path"]),
        "gallery_path": str(gallery["path"]),
        "progress_paths": progress["paths"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    return summary


@torch.no_grad()
def _collect_outputs(model: SketchPolicy, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    targets: List[int] = []
    preds: List[int] = []
    confidences: List[float] = []
    data_terms: List[float] = []
    logsumexp_terms: List[float] = []

    for batch in dataloader:
        primitive_types = batch["primitive_types"].to(device)
        constraint_types = batch["constraint_types"].to(device)
        span_mm = batch["requirement_span"].to(device)
        outputs = model(primitive_types, constraint_types, span_mm)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        target = batch["target_type"].to(device)

        targets.append(target.cpu().numpy())
        preds.append(pred.cpu().numpy())
        conf = probs.gather(1, target.unsqueeze(-1)).squeeze(-1)
        confidences.append(conf.cpu().numpy())
        logsumexp_terms.append(torch.logsumexp(logits, dim=-1).cpu().numpy())
        data_terms.append(logits.gather(1, target.unsqueeze(-1)).squeeze(-1).cpu().numpy())

    return {
        "targets": np.concatenate(targets),
        "preds": np.concatenate(preds),
        "confidences": np.concatenate(confidences),
        "logsumexp": np.concatenate(logsumexp_terms),
        "data_term": np.concatenate(data_terms),
    }


def _plot_reliability(outputs: Dict[str, np.ndarray], output_dir: pathlib.Path) -> Dict[str, Any]:
    probs = outputs["confidences"]
    preds = outputs["preds"]
    targets = outputs["targets"]
    correct = (preds == targets).astype(np.float32)

    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(probs, bins) - 1
    accs: List[float] = []
    confs: List[float] = []
    counts: List[int] = []
    ece = 0.0
    for i in range(len(bins) - 1):
        mask = bin_ids == i
        if not np.any(mask):
            accs.append(0.0)
            confs.append(0.0)
            counts.append(0)
            continue
        acc = correct[mask].mean()
        conf = probs[mask].mean()
        accs.append(acc)
        confs.append(conf)
        counts.append(int(mask.sum()))
        ece += np.abs(acc - conf) * (mask.sum() / len(probs))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bins[:-1], accs, width=0.1, align="edge", alpha=0.7, label="Accuracy")
    ax.plot(bins[:-1], confs, marker="o", color="red", label="Confidence")
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability (ECE={ece:.3f})")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "reliability.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"ece": ece, "path": path}


def _plot_risk_sweep(outputs: Dict[str, np.ndarray], output_dir: pathlib.Path) -> Dict[str, Any]:
    lses = outputs["logsumexp"]
    data_terms = outputs["data_term"]
    lambdas = np.linspace(0.0, 1.5, 8)
    penalties = []
    for lam in lambdas:
        penalty = np.maximum(lses - data_terms - lam, 0.0).mean()
        penalties.append(penalty)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambdas, penalties, marker="o")
    ax.set_xlabel("Risk lambda")
    ax.set_ylabel("Avg risk penalty")
    ax.set_title("Risk-sensitive sweep")
    fig.tight_layout()
    path = output_dir / "risk_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"risk_curve": list(zip(lambdas.tolist(), penalties)), "path": path}


def _plot_confusion(outputs: Dict[str, np.ndarray], dataset: BracketSketchDataset, output_dir: pathlib.Path) -> Dict[str, Any]:
    mat = confusion_matrix(outputs["targets"], outputs["preds"], labels=range(dataset.primitive_vocab_size))
    inv_vocab = _invert_vocab(dataset.meta["primitive_vocab"])
    labels = [inv_vocab.get(i, str(i)) for i in range(dataset.primitive_vocab_size)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"path": path}


def _save_qualitative(outputs: Dict[str, np.ndarray], dataset: BracketSketchDataset, output_dir: pathlib.Path) -> Dict[str, Any]:
    inv_vocab = _invert_vocab(dataset.meta["primitive_vocab"])
    probs = outputs["confidences"]
    preds = outputs["preds"]
    targets = outputs["targets"]
    idx_sorted = np.argsort(-probs)
    exemplar_idxs = list(idx_sorted[:5]) + list(idx_sorted[-5:])
    rows: List[Dict[str, Any]] = []
    for idx in exemplar_idxs:
        rows.append(
            {
                "sample_idx": int(idx),
                "confidence": float(probs[idx]),
                "pred": inv_vocab.get(int(preds[idx]), str(preds[idx])),
                "target": inv_vocab.get(int(targets[idx]), str(targets[idx])),
            }
        )
    path = output_dir / "qualitative.jsonl"
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")
    return {"path": path, "n_rows": len(rows)}


def _render_gallery(config: Dict[str, Any], dataset: BracketSketchDataset, output_dir: pathlib.Path, num_examples: int = 6) -> Dict[str, Any]:
    data_cfg = config["data"]
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    _import_sketchgraphs(repo_root)
    from sketchgraphs.data import flat_array
    from sketchgraphs.data import sequence as sg_sequence

    raw_path = pathlib.Path(data_cfg["raw_dir"]) / data_cfg["sketchgraphs_filename"]
    raw_data = flat_array.load_dictionary_flat(str(raw_path))
    sketch_ids = raw_data["sketch_ids"]
    sequences = raw_data["sequences"]
    def _normalize(raw):
        sid = raw
        if isinstance(sid, (tuple, list, np.ndarray)):
            sid = sid[0]
        if isinstance(sid, bytes):
            sid = sid.decode()
        return str(sid)

    id_to_idx = {_normalize(skid): i for i, skid in enumerate(sketch_ids[: data_cfg.get("max_sequences", len(sketch_ids))])}

    sample_ids = []
    for rec in dataset.records:
        if rec.get("sketch_id") is not None:
            sample_ids.append(str(rec["sketch_id"]))
        if len(sample_ids) >= num_examples:
            break
    sample_idxs = [id_to_idx[sid] for sid in sample_ids if sid in id_to_idx]

    cols = min(3, num_examples)
    rows = math.ceil(len(sample_idxs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for ax, idx in zip(axes, sample_idxs):
        try:
            sketch = sg_sequence.sketch_from_sequence(sequences[idx])
            primitives = [_entity_to_primitive(ent) for ent in sketch.entities.values()]
            _draw_primitives(ax, primitives)
            ax.set_title(f"Sketch {sketch_ids[idx]}")
        except Exception:
            ax.axis("off")
            continue
    for ax in axes[len(sample_idxs) :]:
        ax.axis("off")

    fig.tight_layout()
    path = output_dir / "bracket_gallery.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"path": path, "n": len(sample_idxs)}


def _draw_primitives(ax: plt.Axes, primitives: List[Dict[str, Any]]) -> None:
    for prim in primitives:
        if prim["type"] == "Line" and len(prim.get("points", [])) >= 2:
            p0, p1 = prim["points"][:2]
            ax.plot([p0["x"], p1["x"]], [p0["y"], p1["y"]], color="#4C72B0")
        elif prim["type"] in ("Circle",) and prim.get("center") and prim.get("radius"):
            circ = patches.Circle((prim["center"]["x"], prim["center"]["y"]), prim["radius"], fill=False, color="#C44E52")
            ax.add_patch(circ)
        elif prim["type"] == "Arc" and prim.get("center") and prim.get("points"):
            cx, cy = prim["center"]["x"], prim["center"]["y"]
            start = prim["points"][0]
            end = prim["points"][-1]
            r = prim.get("radius", 0.0)
            start_angle = math.degrees(math.atan2(start["y"] - cy, start["x"] - cx))
            end_angle = math.degrees(math.atan2(end["y"] - cy, end["x"] - cx))
            arc = patches.Arc((cx, cy), 2 * r, 2 * r, angle=0, theta1=start_angle, theta2=end_angle, color="#55A868")
            ax.add_patch(arc)
        elif prim["type"] == "Point" and prim.get("points"):
            pt = prim["points"][0]
            ax.plot(pt["x"], pt["y"], marker="o", color="#8172B3")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _primitive_from_params(type_name: str, params: np.ndarray) -> Dict[str, Any]:
    prim: Dict[str, Any] = {"type": type_name}
    if type_name == "Line":
        prim["points"] = [
            {"x": float(params[0]), "y": float(params[1])},
            {"x": float(params[2]), "y": float(params[3])},
        ]
    elif type_name in ("Circle", "Arc"):
        prim["center"] = {"x": float(params[0]), "y": float(params[1])}
        prim["radius"] = abs(float(params[4]))
        prim["points"] = [
            {"x": float(params[0] + prim["radius"]), "y": float(params[1])},
            {"x": float(params[0]), "y": float(params[1] + prim["radius"])},
        ]
    elif type_name == "Point":
        prim["points"] = [{"x": float(params[0]), "y": float(params[1])}]
    return prim


def _render_progressions(
    config: Dict[str, Any],
    dataset: BracketSketchDataset,
    model: SketchPolicy,
    output_dir: pathlib.Path,
    num_samples: int = 3,
    steps: int = 4,
) -> Dict[str, Any]:
    data_cfg = config["data"]
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    _import_sketchgraphs(repo_root)
    from sketchgraphs.data import flat_array
    from sketchgraphs.data import sequence as sg_sequence

    raw_path = pathlib.Path(data_cfg["raw_dir"]) / data_cfg["sketchgraphs_filename"]
    raw_data = flat_array.load_dictionary_flat(str(raw_path))
    sketch_ids = raw_data["sketch_ids"]
    sequences = raw_data["sequences"]

    inv_vocab = _invert_vocab(dataset.meta["primitive_vocab"])
    vocab = dataset.meta["primitive_vocab"]
    allowed = set(vocab.keys())
    pad_id = dataset.primitive_pad_id
    param_dim = dataset.param_dim

    def normalize_sid(raw):
        sid = raw
        if isinstance(sid, (tuple, list, np.ndarray)):
            sid = sid[0]
        if isinstance(sid, bytes):
            sid = sid.decode()
        return str(sid)

    id_to_idx = {normalize_sid(skid): i for i, skid in enumerate(sketch_ids[: data_cfg.get("max_sequences", len(sketch_ids))])}

    paths: List[str] = []
    device = next(model.parameters()).device
    chosen = dataset.records[:num_samples]
    for sample_idx, rec in enumerate(chosen):
        sid = str(rec.get("sketch_id"))
        if sid not in id_to_idx:
            continue
        sg_idx = id_to_idx[sid]
        sketch = sg_sequence.sketch_from_sequence(sequences[sg_idx])
        primitives = [_entity_to_primitive(ent) for ent in sketch.entities.values() if getattr(ent, "type", None) and ent.type.name in allowed]
        types = [vocab[p["type"]] for p in primitives]
        params = [_primitive_parameters(p) for p in primitives]
        if len(types) < 3:
            continue

        step_points = np.linspace(1, len(types) - 1, steps, dtype=int)
        fig, axes = plt.subplots(1, steps, figsize=(4 * steps, 4))
        if steps == 1:
            axes = [axes]
        for ax, t in zip(axes, step_points):
            context_types = torch.full((1, t), pad_id, dtype=torch.long, device=device)
            context_params = torch.zeros((1, t, param_dim), dtype=torch.float32, device=device)
            context_types[0, :t] = torch.tensor(types[:t], device=device)
            context_params[0, :t, :] = torch.tensor(params[:t], device=device)
            cons = torch.full((1, max(1, t - 1)), dataset.constraint_pad_id, dtype=torch.long, device=device)
            outputs = model(context_types.cpu(), cons.cpu(), torch.tensor([rec["requirement_span"]], dtype=torch.float32))
            pred_type_id = int(outputs["logits"].argmax(dim=-1).item())
            pred_type = inv_vocab.get(pred_type_id, str(pred_type_id))
            pred_params = outputs["mu"].detach().cpu().numpy()[0]
            pred_prim = _primitive_from_params(pred_type, pred_params)
            gt_type = inv_vocab.get(types[t], str(types[t])) if t < len(types) else "None"
            gt_prim = primitives[t] if t < len(primitives) else {}

            _draw_primitives(ax, primitives[:t])
            if gt_prim:
                _draw_primitives(ax, [gt_prim])
            if pred_prim:
                _draw_primitives(ax, [pred_prim])
            ax.set_title(f"t={t}: pred {pred_type} vs gt {gt_type}")

        fig.tight_layout()
        out_path = output_dir / f"progress_{sample_idx}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        paths.append(str(out_path))

    return {"paths": paths}


def _invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {int(v): k for k, v in vocab.items()}
