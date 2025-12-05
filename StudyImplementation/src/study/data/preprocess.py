"""Data curation utilities for the bracket sketch study."""

from __future__ import annotations

import json
import math
import pathlib
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

# These helpers are exercised by tests; keep them available at module scope.
PARAM_DIM = 6


def preprocess_dataset(config: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    """Orchestrate the full preprocessing pipeline.

    Steps:
    - Ensure the raw SketchGraphs file is present (download if missing).
    - Load the serialized sequences using the SketchGraphs flat-array loader.
    - Convert sequences to sketches, filter to bracket-like samples, and write JSONL splits.
    - Persist summary statistics and lightweight figures under ``stats_dir``.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    _import_sketchgraphs(repo_root)
    from sketchgraphs.data import flat_array
    from sketchgraphs.data import sequence as sg_sequence
    from sketchgraphs.data._entity import EntityType

    data_cfg = config["data"]
    exp_cfg = config.get("experiment", {})
    raw_dir = pathlib.Path(data_cfg["raw_dir"])
    processed_dir = pathlib.Path(data_cfg["processed_dir"])
    stats_dir = pathlib.Path(data_cfg.get("stats_dir", processed_dir / "stats"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / data_cfg["sketchgraphs_filename"]
    if not raw_path.exists() or overwrite:
        _download_raw(data_cfg["sketchgraphs_url"], raw_path)

    # Build vocabularies.
    allowed_primitives = _normalize_names(data_cfg["allowed_primitive_types"])
    allowed_constraints = _normalize_names(data_cfg["allowed_constraint_types"])
    primitive_vocab = {name: idx for idx, name in enumerate(allowed_primitives)}
    primitive_unknown_id = len(primitive_vocab)
    primitive_vocab["Other"] = primitive_unknown_id
    constraint_vocab = {name: idx for idx, name in enumerate(allowed_constraints)}

    records: list[Dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    length_hist: list[int] = []
    span_hist: list[float] = []
    loop_hist: list[int] = []
    primitive_hist: Counter[str] = Counter()
    constraint_hist: Counter[str] = Counter()

    data = flat_array.load_dictionary_flat(str(raw_path))
    sequences = data["sequences"]
    sketch_ids = data["sketch_ids"]
    max_sequences = data_cfg.get("max_sequences")
    num_sequences = len(sequences) if max_sequences is None else min(max_sequences, len(sequences))
    seed = exp_cfg.get("seed", 0)
    rng = random.Random(seed)

    for idx in tqdm(range(num_sequences), desc="Filtering sketches"):
        try:
            seq = sequences[idx]
        except Exception:
            skip_counts["decode_error"] += 1
            continue

        try:
            sketch = sg_sequence.sketch_from_sequence(seq)
        except Exception:
            skip_counts["invalid_sequence"] += 1
            continue

        primitives = []
        for ent in sketch.entities.values():
            if ent.type in (EntityType.External, EntityType.Stop):
                continue
            if ent.type.name not in allowed_primitives:
                continue
            primitives.append(_entity_to_primitive(ent))

        if len(primitives) < 2:
            skip_counts["too_few_primitives"] += 1
            continue
        if len(primitives) > data_cfg["max_primitives"]:
            skip_counts["too_many_primitives"] += 1
            continue

        bbox = _bounding_box(primitives)
        if bbox["width"] <= 1e-6 or bbox["height"] <= 1e-6:
            skip_counts["degenerate_bbox"] += 1
            continue

        loops = _count_loops(primitives)
        if loops > data_cfg["max_profile_loops"]:
            skip_counts["too_many_loops"] += 1
            continue

        constraint_types = [
            c.type.name for c in sketch.constraints.values() if c.type.name in allowed_constraints
        ]
        if not _looks_like_bracket(primitives, constraint_types, bbox):
            skip_counts["failed_bracket_heuristic"] += 1
            continue

        prim_params = [_primitive_parameters(p) for p in primitives]
        prim_ids = [primitive_vocab.get(p["type"], primitive_unknown_id) for p in primitives]
        constraint_ids = [constraint_vocab[c] for c in constraint_types[: len(primitives) - 1]]

        # Use prefix as context; last primitive is the prediction target.
        context_types = prim_ids[:-1]
        context_params = prim_params[:-1]
        target_type = prim_ids[-1]
        target_params = prim_params[-1]

        raw_skid = sketch_ids[idx]
        if isinstance(raw_skid, (tuple, list, np.ndarray)):
            raw_skid = raw_skid[0]
        if isinstance(raw_skid, bytes):
            raw_skid = raw_skid.decode()

        records.append(
            {
                "sketch_id": str(raw_skid),
                "primitive_types": context_types,
                "primitive_params": context_params,
                "constraint_types": constraint_ids,
                "target_type": target_type,
                "target_params": target_params,
                "requirement_span": max(bbox["width"], bbox["height"]),
                "loops": loops,
            }
        )

        length_hist.append(len(context_types))
        span_hist.append(max(bbox["width"], bbox["height"]))
        loop_hist.append(loops)
        for p in primitives:
            primitive_hist[p["type"]] += 1
        for c in constraint_types:
            constraint_hist[c] += 1

    if not records:
        raise RuntimeError("No sketches survived preprocessing; relax filters or verify raw data.")

    rng.shuffle(records)
    splits = _split_records(records, data_cfg, rng)

    for name, split_records in splits.items():
        path = processed_dir / f"{name}.jsonl"
        _write_jsonl(path, split_records)

    metadata = {
        "primitive_vocab": primitive_vocab,
        "constraint_vocab": constraint_vocab,
        "primitive_vocab_size": len(primitive_vocab),
        "constraint_vocab_size": len(constraint_vocab),
        "primitive_pad_id": len(primitive_vocab),
        "constraint_pad_id": len(constraint_vocab),
        "param_dim": PARAM_DIM,
        "max_primitives": data_cfg["max_primitives"],
        "config_snapshot": data_cfg,
    }
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    stats = {
        "total_sequences": int(num_sequences),
        "retained": {k: len(v) for k, v in splits.items()},
        "skipped": dict(skip_counts),
        "primitive_hist": primitive_hist,
        "constraint_hist": constraint_hist,
        "length_histogram": _hist_summary(length_hist),
        "span_summary": _summary(span_hist),
        "loop_summary": _hist_summary(loop_hist),
    }
    with (stats_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
    _plot_histograms(stats_dir, length_hist, span_hist, loop_hist, primitive_hist, constraint_hist)

    return stats


def _import_sketchgraphs(repo_root: pathlib.Path) -> None:
    """Lazy-import SketchGraphs, falling back to the vendored clone."""
    try:
        import sketchgraphs  # noqa: F401
        return
    except ImportError:
        sg_path = repo_root / "SketchGraphs"
        if sg_path.exists():
            sys.path.insert(0, str(sg_path))
            import sketchgraphs  # noqa: F401
            return
    raise ImportError("SketchGraphs is required; ensure the repository exists under SketchGraphs/.")


def _download_raw(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SketchGraphs sequences from {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _normalize_names(names: Iterable[str]) -> List[str]:
    return [str(n).strip() for n in names]


def _entity_to_primitive(ent) -> Dict[str, Any]:
    """Convert a SketchGraphs entity into a normalized primitive dictionary."""
    prim: Dict[str, Any] = {"type": ent.type.name}
    if ent.type.name == "Line":
        start = _line_endpoint(ent, ent.startParam)
        end = _line_endpoint(ent, ent.endParam)
        prim["points"] = [start, end]
        prim["length"] = math.dist((start["x"], start["y"]), (end["x"], end["y"]))
    elif ent.type.name == "Arc":
        prim["center"] = {"x": float(ent.xCenter), "y": float(ent.yCenter)}
        prim["points"] = [
            {"x": float(ent.start_point[0]), "y": float(ent.start_point[1])},
            {"x": float(ent.end_point[0]), "y": float(ent.end_point[1])},
        ]
        prim["radius"] = float(ent.radius)
    elif ent.type.name == "Circle":
        prim["center"] = {"x": float(ent.xCenter), "y": float(ent.yCenter)}
        prim["radius"] = float(ent.radius)
    elif ent.type.name == "Point":
        prim["points"] = [{"x": float(ent.x), "y": float(ent.y)}]
    else:
        prim["points"] = []
    return prim


def _line_endpoint(ent, offset: float) -> Dict[str, float]:
    return {
        "x": float(ent.pntX + ent.dirX * offset),
        "y": float(ent.pntY + ent.dirY * offset),
    }


def _bounding_box(primitives: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute axis-aligned bounding box for a collection of primitives."""
    xs: list[float] = []
    ys: list[float] = []
    for prim in primitives:
        if "points" in prim:
            for pt in prim["points"]:
                xs.append(float(pt["x"]))
                ys.append(float(pt["y"]))
        if prim.get("center") and prim.get("radius") is not None:
            cx, cy = prim["center"]["x"], prim["center"]["y"]
            r = prim.get("radius", 0.0)
            xs.extend([cx - r, cx + r])
            ys.extend([cy - r, cy + r])
    if not xs or not ys:
        return {"width": 0.0, "height": 0.0, "xmin": 0.0, "ymin": 0.0}

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return {"width": float(xmax - xmin), "height": float(ymax - ymin), "xmin": float(xmin), "ymin": float(ymin)}


def _primitive_parameters(primitive: Dict[str, Any]) -> List[float]:
    """Project a primitive dictionary to a fixed-length parameter vector."""
    params = [0.0] * PARAM_DIM
    points = primitive.get("points", [])
    if points:
        params[0] = float(points[0]["x"])
        params[1] = float(points[0]["y"])
        if len(points) > 1:
            params[2] = float(points[1]["x"])
            params[3] = float(points[1]["y"])
    if "center" in primitive:
        params[0] = float(primitive["center"]["x"])
        params[1] = float(primitive["center"]["y"])
        params[2] = float(primitive["center"]["x"])
        params[3] = float(primitive["center"]["y"])
    if "radius" in primitive:
        params[4] = float(primitive["radius"])
    params[5] = float(primitive.get("length", 0.0))
    return params


def _count_loops(primitives: List[Dict[str, Any]], tol: float = 1e-3) -> int:
    """Approximate the number of closed loops based on endpoint connectivity."""
    if not primitives:
        return 0

    graph = nx.Graph()
    circle_like = 0
    for prim in primitives:
        pts = prim.get("points", [])
        if len(pts) >= 2:
            start = (round(pts[0]["x"] / tol), round(pts[0]["y"] / tol))
            end = (round(pts[-1]["x"] / tol), round(pts[-1]["y"] / tol))
            graph.add_edge(start, end)
        elif prim.get("radius"):
            # Circles without explicit endpoints still contribute a closed profile.
            circle_like += 1

    loops = circle_like
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        # Count a loop if the component has more edges than nodes - 1 (contains a cycle).
        if subgraph.number_of_edges() >= subgraph.number_of_nodes():
            loops += 1
    return loops


def _looks_like_bracket(primitives: List[Dict[str, Any]], constraints: List[str], bbox: Dict[str, float]) -> bool:
    """Heuristic to flag L/U bracket sketches."""
    if len(primitives) < 3:
        return False
    has_circle = any(p["type"] == "Circle" for p in primitives)
    has_arc = any(p["type"] == "Arc" for p in primitives)
    num_lines = sum(1 for p in primitives if p["type"] == "Line")
    aspect = bbox["width"] / max(bbox["height"], 1e-6)
    aspect_ok = 0.3 <= aspect <= 3.5
    perpendicular_like = sum(c in ("Perpendicular", "Horizontal", "Vertical") for c in constraints)
    return aspect_ok and num_lines >= 3 and (has_circle or has_arc) and perpendicular_like >= 1


def _split_records(records: List[Dict[str, Any]], data_cfg: Dict[str, Any], rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    n_total = len(records)
    n_train = int(n_total * data_cfg["train_split"])
    n_val = int(n_total * data_cfg["val_split"])
    splits = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }
    return splits


def _write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def _hist_summary(values: List[int]) -> Dict[str, float]:
    return _summary(values)


def _summary(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _plot_histograms(
    output_dir: pathlib.Path,
    lengths: List[int],
    spans: List[float],
    loops: List[int],
    prim_hist: Counter[str],
    constraint_hist: Counter[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(lengths, bins=20, color="#4C72B0")
    axes[0, 0].set_title("Primitive counts")
    axes[0, 1].hist(spans, bins=20, color="#55A868")
    axes[0, 1].set_title("Span (mm)")
    axes[1, 0].bar(prim_hist.keys(), prim_hist.values(), color="#C44E52")
    axes[1, 0].set_title("Primitive frequency")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 1].bar(constraint_hist.keys(), constraint_hist.values(), color="#8172B3")
    axes[1, 1].set_title("Constraint frequency")
    axes[1, 1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(output_dir / "data_histograms.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(loops, bins=np.arange(-0.5, max(loops) + 1.5, 1.0), color="#64B5CD")
    ax2.set_title("Loop counts")
    fig2.tight_layout()
    fig2.savefig(output_dir / "loops.png", dpi=200)
    plt.close(fig2)


__all__ = [
    "preprocess_dataset",
    "_bounding_box",
    "_primitive_parameters",
]
