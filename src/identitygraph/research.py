"""Frozen graph representations and experiments; labels never construct a graph."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .features import BEHAVIOR_FEATURES, GRAPH_FEATURES
from .graph import build_snapshot
from .models import (CalibratedEstimator, TARGETS, _latest_account, _positive_probability,
                     choose_threshold, split_validation)


def adjacency_at(observed, cutoff):
    graph, mappings = build_snapshot(**observed, cutoff=cutoff)
    names = [f"{kind}:{key}" for kind in mappings for key in mappings[kind]]
    index = {key: i for i, key in enumerate(names)}
    adjacency = [set() for _ in names]
    reverse = {kind: {value: key for key, value in mapping.items()} for kind, mapping in mappings.items()}
    for (source, _, target), edges in graph.edge_index_dict.items():
        for left, right in edges.t().tolist():
            a = index[f"{source}:{reverse[source][left]}"]
            b = index[f"{target}:{reverse[target][right]}"]
            adjacency[a].add(b)
            adjacency[b].add(a)
    return names, [sorted(value) for value in adjacency]


def fit_node2vec(observed, cutoff, dimensions=12, walks=4, length=10, epochs=3):
    """Second-order p=1,q=2 walks with skip-gram negative sampling on CPU."""
    names, adjacency = adjacency_at(observed, cutoff)
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    pairs = []
    for start in range(len(names)):
        for _ in range(walks):
            walk = [start]
            for _ in range(length - 1):
                neighbors = adjacency[walk[-1]]
                if not neighbors:
                    break
                previous = walk[-2] if len(walk) > 1 else None
                weights = np.array([1.0 if previous is None or node == previous or node in adjacency[previous]
                                    else 0.5 for node in neighbors])
                walk.append(int(rng.choice(neighbors, p=weights / weights.sum())))
            for i, center in enumerate(walk):
                pairs.extend((center, walk[j]) for j in range(max(0, i - 2), min(len(walk), i + 3)) if i != j)
    embedding, context = nn.Embedding(max(1, len(names)), dimensions), nn.Embedding(max(1, len(names)), dimensions)
    nn.init.normal_(embedding.weight, std=0.1)
    nn.init.normal_(context.weight, std=0.1)
    optimizer = torch.optim.Adam([*embedding.parameters(), *context.parameters()], lr=0.025)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    losses = []
    for _ in range(epochs):
        loss_sum, count = 0.0, 0
        for batch in np.array_split(rng.permutation(len(pairs)), max(1, (len(pairs) + 2047) // 2048)):
            if not len(batch):
                continue
            values = torch.from_numpy(pairs[batch])
            center, positive = embedding(values[:, 0]), context(values[:, 1])
            negative = context(torch.from_numpy(rng.integers(0, max(1, len(names)), (len(batch), 4))))
            loss = -torch.nn.functional.logsigmoid((center * positive).sum(1)).mean()
            loss -= torch.nn.functional.logsigmoid(-(negative * center[:, None, :]).sum(2)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            count += len(batch)
        losses.append(loss_sum / max(1, count))
    return {"names": names, "vectors": embedding.weight.detach().numpy()[:len(names)],
            "cutoff": cutoff, "p": 1.0, "q": 2.0, "dimensions": dimensions,
            "walks_per_node": walks, "walk_length": length, "epochs": epochs,
            "training_loss": losses, "positive_pairs": len(pairs)}


def embedding_frame(frame, observed, embedding):
    lookup = {key: value for key, value in zip(embedding["names"], embedding["vectors"])}
    events = observed["events"].set_index("event_id")
    result = frame.copy()
    for kind, column in [("account", "account_id"), ("device", "device_id"), ("ip", "ip_id")]:
        keys = [f"{kind}:{value}" for value in events.loc[frame.event_id, column]]
        values = np.asarray([lookup.get(key, np.zeros(embedding["dimensions"])) for key in keys])
        for index in range(embedding["dimensions"]):
            result[f"n2v_{kind}_{index}"] = values[:, index] if len(values) else []
        result[f"n2v_{kind}_unknown"] = [float(key not in lookup) for key in keys]
    return result


def readout_predictions(train, validation, tests, names, model_name):
    calibration_frame, selection_frame = split_validation(validation)
    predictions, metadata = [], {}
    for target, label in TARGETS.items():
        fit = _latest_account(train) if target == "identity" else train
        calibration = _latest_account(calibration_frame) if target == "identity" else calibration_frame
        selection = _latest_account(selection_frame) if target == "identity" else selection_frame
        estimator = LGBMClassifier(n_estimators=100, num_leaves=12, max_depth=5, learning_rate=0.05,
                                   min_child_samples=15, reg_lambda=2, verbosity=-1, n_jobs=2,
                                   random_state=42, deterministic=True, force_col_wise=True)
        estimator.fit(fit[names].to_numpy(), fit[label].to_numpy())
        calibrator = None
        if len(calibration) >= 12 and calibration[label].nunique() == 2:
            raw = np.clip(_positive_probability(estimator, calibration[names].to_numpy()), 1e-6, 1 - 1e-6)
            calibrator = LogisticRegression(random_state=42).fit(np.log(raw / (1 - raw)).reshape(-1, 1), calibration[label])
        fitted = CalibratedEstimator(estimator, calibrator)
        threshold = choose_threshold(selection[label].to_numpy(), fitted.predict_score(selection[names].to_numpy()))
        metadata[target] = {"estimator": fitted, "feature_names": names, "threshold": threshold, "fit_examples": len(fit)}
        for split, frame in tests.items():
            result = frame[["event_id", "account_id", "event_time"]].copy()
            result["label"], result["target"], result["model"], result["split"] = frame[label], target, model_name, split
            result["score"] = fitted.predict_score(frame[names].to_numpy()) if len(frame) else []
            result["threshold"] = threshold["threshold"]
            predictions.append(result)
    return pd.concat(predictions, ignore_index=True), metadata


def train_representation_experiments(observed, train, validation, tests, populations, output: Path, warmup_cutoff):
    output.mkdir(parents=True, exist_ok=True)
    embedding = fit_node2vec(observed, warmup_cutoff)
    np.savez_compressed(output / "node2vec.npz", names=np.array(embedding["names"]), vectors=embedding["vectors"])
    eligible = train[train.event_time >= warmup_cutoff]
    augmented_train = embedding_frame(eligible, observed, embedding)
    augmented_val = embedding_frame(validation, observed, embedding)
    augmented_tests = {name: embedding_frame(frame, populations.get(name, observed), embedding) for name, frame in tests.items()}
    names = BEHAVIOR_FEATURES + GRAPH_FEATURES + [column for column in augmented_train if column.startswith("n2v_")]
    predictions, models = readout_predictions(augmented_train, augmented_val, augmented_tests, names, "node2vec_lightgbm")
    matched, matched_models = readout_predictions(eligible, validation, tests, BEHAVIOR_FEATURES + GRAPH_FEATURES,
                                                 "lightgbm_matched_warmup")
    joblib.dump({"node2vec": models, "matched_warmup": matched_models}, output / "readouts.joblib")
    summary = {key: value for key, value in embedding.items() if key not in ("names", "vectors")}
    summary.update(nodes=len(embedding["names"]), readout_training_rows=len(eligible),
                   inference="Frozen warmup embeddings; unseen IDs use zero vectors and explicit missing-node flags.",
                   comparison="Matched-warmup LightGBM uses identical training rows without embeddings.",
                   limitation="Transductive embeddings cannot transfer entity identity to independent populations.")
    (output / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pd.concat([predictions, matched], ignore_index=True), summary


def train_anomaly(train, validation, output: Path):
    names = BEHAVIOR_FEATURES + GRAPH_FEATURES
    model = make_pipeline(StandardScaler(), IsolationForest(n_estimators=100, random_state=42, n_jobs=2))
    model.fit(train[names].to_numpy())  # Deliberately unsupervised; training includes undiagnosed anomalies.
    reference = np.sort(-model.score_samples(validation[names].to_numpy()))
    bundle = {"estimator": model, "feature_names": names, "reference_scores": reference,
              "score_type": "validation_percentile_not_fraud_probability", "training_rows": len(train)}
    joblib.dump(bundle, output)
    return bundle


def anomaly_scores(bundle, frame):
    raw = -bundle["estimator"].score_samples(frame[bundle["feature_names"]].to_numpy())
    return np.searchsorted(bundle["reference_scores"], raw, side="right") / max(1, len(bundle["reference_scores"]))


def link_examples(observed, start, end, seed=42):
    """Future first-use account-device links; negatives mean unobserved in this window."""
    events = observed["events"]
    history = events[events.event_time < start]
    upcoming = events[(events.event_time >= start) & (events.event_time < end)]
    known_accounts = set(observed["accounts"].loc[observed["accounts"].created_at < start, "account_id"])
    known_devices = set(history.device_id)
    prior = set(zip(history.account_id, history.device_id))
    future = set(zip(upcoming.account_id, upcoming.device_id))
    excluded = prior | future
    # Retrieval universe is explicitly bounded to entities observed at the cutoff.
    positives = sorted((a, d) for a, d in future - prior if a in known_accounts and d in known_devices)
    candidates = sorted((a, d) for a in known_accounts for d in known_devices if (a, d) not in excluded)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(candidates), min(len(candidates), max(20, len(positives) * 5)), replace=False)
    pairs = positives + [candidates[i] for i in sampled]
    account_devices, device_accounts, account_ips = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in history.itertuples():
        account_devices[row.account_id].add(row.device_id)
        device_accounts[row.device_id].add(row.account_id)
        account_ips[row.account_id].add(row.ip_id)
    records = []
    for index, (account, device) in enumerate(pairs):
        peers = device_accounts[device]
        contacts = set().union(*(device_accounts[d] for d in account_devices[account])) if account_devices[account] else set()
        peer_ips = set().union(*(account_ips[a] for a in peers)) if peers else set()
        records.append({"account_id": account, "device_id": device, "label": int(index < len(positives)),
                        "account_degree": len(account_devices[account]), "device_degree": len(peers),
                        "shared_contacts": len(contacts & peers),
                        "network_jaccard": len(account_ips[account] & peer_ips) / max(1, len(account_ips[account] | peer_ips)),
                        "history_cutoff": start, "label_window_end": end})
    return pd.DataFrame(records), {"eligible_positive_links": len(positives), "all_new_links": len(future - prior),
                                   "candidate_universe": len(candidates) + len(positives), "sampled_negatives": len(sampled)}


def link_experiment(observed, boundaries, output):
    frames, coverage = {}, {}
    for name, (start, end) in boundaries.items():
        frames[name], coverage[name] = link_examples(observed, start, end)
    names = ["account_degree", "device_degree", "shared_contacts", "network_jaccard"]
    if any(frame.empty or frame.label.nunique() < 2 for frame in frames.values()):
        return {"status": "insufficient_classes", "coverage": coverage}
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    estimator.fit(frames["train"][names].to_numpy(), frames["train"].label)
    selection = choose_threshold(frames["validation"].label.to_numpy(), estimator.predict_proba(frames["validation"][names].to_numpy())[:, 1])
    test = frames["test"].copy()
    test["score"] = estimator.predict_proba(test[names].to_numpy())[:, 1]
    test.to_parquet(output / "link_predictions.parquet", index=False)
    joblib.dump({"estimator": estimator, "feature_names": names, "threshold": selection}, output / "link_decoder.joblib")
    y, alerts = test.label.to_numpy(), test.score.to_numpy() >= selection["threshold"]
    return {"status": "complete", "average_precision": float(average_precision_score(y, test.score)),
            "recall": float((alerts & (y == 1)).sum() / max(1, y.sum())), "selection": selection,
            "coverage": coverage, "windows": boundaries, "test_examples": len(test),
            "target": "First observed account-device link in the future window, not a fraud label",
            "limitations": ["Sampled-negative ranking; scores are not deployment-calibrated relationship probabilities.",
                            "Only nodes observed before each cutoff enter the candidate universe; cold nodes are excluded.",
                            "Unobserved links are window-specific negatives, not confirmed absent relationships."]}
