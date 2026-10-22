"""CPU heterogeneous GraphSAGE experiment with observable historical snapshots.

This module does not promote experimental GNN weights to the serving bundle.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv, RGCNConv, SAGEConv

from .features import BEHAVIOR_FEATURES
from .graph import build_snapshot
from .models import FPR_LIMIT, TARGETS, _check_frames, _latest_account, _numeric, choose_threshold, split_validation


class HeterogeneousGraphSAGE(nn.Module):
    """Two relation-specific mean-aggregation layers and separate binary heads."""

    def __init__(self, node_types, edge_types, input_dims: dict, behavior_dim: int, hidden: int = 24,
                 architecture: str = "graphsage"):
        super().__init__()
        if architecture not in ("graphsage", "gat", "rgcn", "relation_ablated"):
            raise ValueError("Unknown graph architecture")
        self.architecture, self.edge_types = architecture, list(edge_types)
        self.node_types = list(node_types)
        if architecture in ("rgcn", "relation_ablated"):
            relations = len(edge_types) if architecture == "rgcn" else 1
            if len(set(input_dims.values())) != 1:
                raise ValueError("RGCN requires a common numeric feature width")
            self.first = RGCNConv(next(iter(input_dims.values())), hidden, relations)
            self.second = RGCNConv(hidden, hidden, relations)
        else:
            conv = SAGEConv if architecture == "graphsage" else GATConv
            kwargs = {} if architecture == "graphsage" else {"add_self_loops": False}
            self.first = HeteroConv({edge: conv((input_dims[edge[0]], input_dims[edge[2]]), hidden, **kwargs)
                                    for edge in edge_types}, aggr="mean")
            self.second = HeteroConv({edge: conv((hidden, hidden), hidden, **kwargs) for edge in edge_types}, aggr="mean")
            if architecture == "gat":
                self.residual = nn.ModuleDict({node: nn.Linear(input_dims[node], hidden) for node in node_types})
        self.unknown = nn.ParameterDict({node: nn.Parameter(torch.zeros(hidden)) for node in ["account", "device", "ip"]})
        self.heads = nn.ModuleDict({target: nn.Sequential(nn.Linear(3 * hidden + behavior_dim, 32),
                                                        nn.ReLU(), nn.Linear(32, 1)) for target in TARGETS})

    def forward(self, graph, indices: dict[str, torch.Tensor], behavior: torch.Tensor) -> torch.Tensor:
        x = {key: torch.sign(value) * torch.log1p(value.abs()) for key, value in graph.x_dict.items()}
        if self.architecture in ("rgcn", "relation_ablated"):
            offsets, sizes, position = {}, {}, 0
            for node in self.node_types:
                offsets[node], sizes[node] = position, len(x[node])
                position += sizes[node]
            edges, relations = [], []
            for index, edge in enumerate(self.edge_types):
                values = graph[edge].edge_index
                edges.append(values + values.new_tensor([[offsets[edge[0]]], [offsets[edge[2]]]]))
                relations.append(values.new_full((values.shape[1],), index if self.architecture == "rgcn" else 0))
            edge_index, edge_type = torch.cat(edges, dim=1), torch.cat(relations)
            values = torch.cat([x[node] for node in self.node_types])
            values = torch.relu(self.first(values, edge_index, edge_type))
            values = torch.relu(self.second(values, edge_index, edge_type))
            x = {node: values[offsets[node]:offsets[node] + sizes[node]] for node in self.node_types}
        else:
            original = x
            x = {key: torch.relu(value) for key, value in self.first(x, graph.edge_index_dict).items()}
            x = {key: torch.relu(value) for key, value in self.second(x, graph.edge_index_dict).items()}
            if self.architecture == "gat":
                x = {key: value + torch.relu(self.residual[key](original[key])) for key, value in x.items()}
        pooled = []
        for node_type in ["account", "device", "ip"]:
            node_values = x[node_type]
            # Appending a learned row supports new entities without inventing future graph edges.
            table = torch.cat([node_values, self.unknown[node_type].unsqueeze(0)], dim=0)
            index = indices[node_type]
            safe = torch.where(index >= 0, index, len(node_values))
            pooled.append(table[safe])
        query = torch.cat([*pooled, behavior], dim=1)
        return torch.cat([self.heads[target](query) for target in TARGETS], dim=1)


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def train_gnn(accounts, attributes, account_attributes, events, train_frame, validation_frame,
              test_frames: dict[str, pd.DataFrame], output_dir: Path, epochs: int = 12,
              architecture: str = "graphsage", extra_populations: dict | None = None) -> dict:
    """Fit on train queries only, then calibrate and evaluate without test tuning.

    Snapshots are taken at 00:00 UTC, so graph structure can be up to 24 hours
    stale. Exact point-in-time behavioral features supplement stale graph context.
    Earlier validation/test observations may update subsequent snapshots, but
    their labels never update weights, calibration, or selected thresholds.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    _check_frames(train_frame, validation_frame)
    started = time.perf_counter()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = {"train": train_frame, "validation": validation_frame, **test_frames}
    extra_populations = extra_populations or {}
    all_frames.update({name: value[1] for name, value in extra_populations.items()})
    behavior_names = list(BEHAVIOR_FEATURES)
    scaler = StandardScaler().fit(_numeric(train_frame, behavior_names))
    main_observed = dict(accounts=accounts, attributes=attributes, account_attributes=account_attributes, events=events)
    snapshots = {}
    prepared = {}
    fallback_counts = {}
    for split, frame in all_frames.items():
        population = split if split in extra_populations else "main"
        observed = extra_populations[split][0] if split in extra_populations else main_observed
        event_lookup = observed["events"].set_index("event_id", verify_integrity=True)
        if frame.empty:
            prepared[split] = []
            fallback_counts[split] = {key: 0 for key in ["account", "device", "ip"]}
            continue
        missing = set(frame.event_id) - set(event_lookup.index)
        if missing:
            raise ValueError(f"GNN {split} has {len(missing)} event IDs absent from observed events")
        ordered = frame.reset_index(drop=True).copy()
        ordered["_position"] = np.arange(len(ordered))
        ordered["_cutoff"] = pd.to_datetime(ordered.event_time, utc=True).dt.floor("D")
        split_batches = []
        counts = {key: 0 for key in ["account", "device", "ip"]}
        identity_counts = ordered.account_id.value_counts()
        for cutoff, group in ordered.groupby("_cutoff", sort=True):
            cutoff_string = cutoff.isoformat().replace("+00:00", "Z")
            key = (population, cutoff_string)
            if key not in snapshots:
                snapshots[key] = build_snapshot(**observed, cutoff=cutoff_string)
            graph, mappings = snapshots[key]
            source = event_lookup.loc[group.event_id]
            indices = {}
            for node, column in [("account", "account_id"), ("device", "device_id"), ("ip", "ip_id")]:
                values = [mappings[node].get(str(value), -1) for value in source[column]]
                indices[node] = torch.tensor(values, dtype=torch.long)
                counts[node] += sum(value < 0 for value in values)
            values = scaler.transform(_numeric(group, behavior_names)).astype(np.float32)
            labels = group[list(TARGETS.values())].to_numpy(dtype=np.float32)
            # Keep frequent accounts from dominating the identity head.
            weights = group.account_id.map(lambda key: 1 / identity_counts[key]).to_numpy(dtype=np.float32)
            split_batches.append({"cutoff": cutoff_string, "graph": graph, "indices": indices,
                                  "behavior": torch.from_numpy(values), "labels": torch.from_numpy(labels),
                                  "identity_weights": torch.from_numpy(weights),
                                  "positions": group["_position"].to_numpy()})
        prepared[split] = split_batches
        fallback_counts[split] = counts
    prototype = next(iter(snapshots.values()))[0]
    metadata = prototype.metadata()
    input_dims = {node: int(prototype[node].x.shape[1]) for node in metadata[0]}
    model = HeterogeneousGraphSAGE(*metadata, input_dims, len(behavior_names), architecture=architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    losses = []
    train_labels = train_frame[list(TARGETS.values())].to_numpy(dtype=np.float32)
    positive_weights = np.clip((len(train_labels) - train_labels.sum(axis=0)) /
                               np.maximum(train_labels.sum(axis=0), 1), 1, 10)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(positive_weights))
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        # Time order is fixed; no validation or test batches can enter this loop.
        for batch in prepared["train"]:
            optimizer.zero_grad()
            logits = model(batch["graph"], batch["indices"], batch["behavior"])
            per_row = criterion(logits, batch["labels"])
            identity_weight = batch["identity_weights"]
            loss = per_row[:, 0].mean() + (per_row[:, 1] * identity_weight).sum() / identity_weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(logits)
            count += len(logits)
        losses.append({"epoch": epoch + 1, "training_loss": epoch_loss / max(count, 1)})
    raw_predictions = {}
    model.eval()
    with torch.no_grad():
        for split, frame in all_frames.items():
            scores = np.zeros((len(frame), 2), dtype=np.float64)
            for batch in prepared[split]:
                logits = model(batch["graph"], batch["indices"], batch["behavior"])
                scores[batch["positions"]] = torch.sigmoid(logits).numpy()
            raw_predictions[split] = scores
    calibration_frame, selection_frame = split_validation(validation_frame)
    calibration_ids = set(calibration_frame.event_id)
    selection_ids = set(selection_frame.event_id)
    calibrated_predictions = {key: value.copy() for key, value in raw_predictions.items()}
    calibration_info = {}
    thresholds = {}
    for target_index, (target, label) in enumerate(TARGETS.items()):
        calibration = _latest_account(calibration_frame) if target == "identity" else calibration_frame
        selection = _latest_account(selection_frame) if target == "identity" else selection_frame
        event_positions = {value: index for index, value in enumerate(validation_frame.event_id)}
        calibration_positions = [event_positions[value] for value in calibration.event_id]
        selection_positions = [event_positions[value] for value in selection.event_id]
        info = {"status": "skipped_insufficient_validation_classes", "examples": len(calibration)}
        if len(calibration) >= 12 and calibration[label].nunique() == 2:
            calibrator = LogisticRegression(C=1.0, max_iter=300, random_state=42)
            calibrator.fit(_logit(raw_predictions["validation"][calibration_positions, target_index]),
                           calibration[label].to_numpy(dtype=int))
            for split, scores in raw_predictions.items():
                if len(scores):
                    calibrated_predictions[split][:, target_index] = calibrator.predict_proba(
                        _logit(scores[:, target_index]))[:, 1]
            info.update(status="platt_earlier_validation", coefficient=float(calibrator.coef_[0, 0]),
                        intercept=float(calibrator.intercept_[0]))
        calibration_info[target] = info
        thresholds[target] = choose_threshold(selection[label].to_numpy(),
                                             calibrated_predictions["validation"][selection_positions, target_index])
    tables = []
    for split, frame in all_frames.items():
        result = frame[["event_id", "account_id", "event_time", *TARGETS.values()]].copy()
        result["split"] = split
        if split == "validation":
            result["validation_role"] = ["calibration" if value in calibration_ids else
                                          "selection" if value in selection_ids else "unused"
                                          for value in frame.event_id]
        for index, target in enumerate(TARGETS):
            result[f"{target}_raw_score"] = raw_predictions[split][:, index]
            result[f"{target}_score"] = calibrated_predictions[split][:, index]
            result[f"{target}_threshold"] = thresholds[target]["threshold"]
        tables.append(result)
    pd.concat(tables, ignore_index=True).to_csv(output_dir / "gnn_predictions.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "node_types": list(metadata[0]),
                "architecture": architecture,
                "edge_types": list(metadata[1]), "input_dims": input_dims, "behavior_names": behavior_names,
                "hidden": 24, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "calibration": calibration_info, "thresholds": thresholds}, output_dir / "gnn_weights.pt")
    summary = {"model": "heterogeneous_" + architecture, "model_version": "identitygraph-gnn-v2",
               "epochs": epochs, "training_history": losses, "runtime_seconds": time.perf_counter() - started,
               "snapshot_policy": "UTC day start; all graph events strictly before cutoff",
               "maximum_graph_staleness_hours": 24, "behavior_features": behavior_names,
               "online_history_policy": "Earlier observed validation/test events may enter later snapshots; labels never do",
               "cold_start_policy": "Learned per-type missing-node embedding plus observed query behavior features",
               "fallback_queries": fallback_counts, "calibration": calibration_info, "thresholds": thresholds,
               "fpr_limit": FPR_LIMIT, "calibration_population": "simulated validation population",
               "splits": {key: len(value) for key, value in all_frames.items()},
               "training_graph_cutoffs": [batch["cutoff"] for batch in prepared["train"]],
               "snapshot_count": len(snapshots), "node_types": list(metadata[0]),
               "edge_types": [list(edge) for edge in metadata[1]], "input_dims": input_dims,
               "served_by_api": False, "weights_path": str(output_dir / "gnn_weights.pt"),
               "predictions_path": str(output_dir / "gnn_predictions.csv"),
               "torch_version": torch.__version__,
               "limitations": ["Daily graph snapshots omit same-day graph updates.",
                               "Calibration and performance describe the simulator only.",
                               "No GNN architecture selection or unseen attack success guarantee.",
                               "Identity loss is inverse-account weighted within each daily batch."]}
    (output_dir / "gnn_metadata.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
