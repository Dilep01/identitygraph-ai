"""Model checks cover label leakage, threshold constraints and cold-start graphs."""

import numpy as np
import pandas as pd
import pytest
import torch

from identitygraph.features import BEHAVIOR_FEATURES, FEATURE_VERSION, GRAPH_FEATURES
from identitygraph.gnn import HeterogeneousGraphSAGE, train_gnn
from identitygraph.graph import build_snapshot
from identitygraph.models import (
    RuleEstimator,
    choose_threshold,
    load_bundle,
    predict_baselines,
    predict_bundle,
    split_validation,
    train_baselines,
)


def frame(start: str, count: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(321)
    data = pd.DataFrame(rng.random((count, len(BEHAVIOR_FEATURES + GRAPH_FEATURES))),
                        columns=BEHAVIOR_FEATURES + GRAPH_FEATURES)
    data["event_id"] = [f"{start}-{i}" for i in range(count)]
    data["account_id"] = [f"account-{i % 25}" for i in range(count)]
    data["event_time"] = pd.date_range(start, periods=count, freq="h", tz="UTC").astype(str)
    data["is_takeover"] = (data.new_device > 0.7).astype(int)
    data["is_synthetic"] = [i % 3 == 0 for i in range(count)]
    return data


def test_threshold_meets_budget_and_timestamp_split_keeps_ties_together():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9])
    chosen = choose_threshold(labels, scores, fpr_limit=0.0)
    assert chosen["fpr"] == 0
    assert chosen["recall"] == 0.5
    validation = frame("2026-02-01", 20)
    validation.loc[10, "event_time"] = validation.loc[9, "event_time"]
    calibration, selection = split_validation(validation)
    assert set(calibration.event_time).isdisjoint(selection.event_time)
    assert len(calibration) + len(selection) == len(validation)


def test_rules_use_declared_failure_and_graph_features():
    names = BEHAVIOR_FEATURES + GRAPH_FEATURES
    values = np.zeros((2, len(names)))
    values[1, names.index("failure_count_24h")] = 4
    values[1, names.index("device_other_account_count")] = 3
    values[1, names.index("shared_attribute_account_count")] = 3
    for target in ["takeover", "identity"]:
        scores = RuleEstimator(names, target).predict_proba(values)[:, 1]
        assert scores[1] > scores[0]


def test_artifact_round_trip_excludes_truth_and_predictions_do_not_refit(tmp_path):
    train, validation = frame("2026-01-01"), frame("2026-02-01")
    train["attack_group"] = "secret-oracle-value"
    summary = train_baselines(train, validation, tmp_path)
    bundle = load_bundle(tmp_path)
    assert bundle["feature_version"] == FEATURE_VERSION
    assert summary["calibration_event_count"] + summary["selection_event_count"] == len(validation)
    for target in ["takeover", "identity"]:
        assert set(bundle["candidate_models"][target]) == {
            "rules", "logistic", "lightgbm_behavior", "lightgbm_graph", "lightgbm_no_device", "lightgbm_no_assurance"}
        assert "attack_group" not in bundle["targets"][target]["feature_names"]
    test = frame("2026-03-01", 10)
    first = predict_baselines(bundle, test)
    test["is_takeover"] = 1 - test["is_takeover"]
    test["is_synthetic"] = 1 - test["is_synthetic"]
    second = predict_baselines(bundle, test)
    np.testing.assert_array_equal(first.score.to_numpy(), second.score.to_numpy())
    scores = predict_bundle(bundle, test.iloc[0].to_dict())
    assert set(scores) == {"takeover", "identity"}
    assert all(0 <= result["score"] <= 1 for result in scores.values())
    assert (tmp_path / "baseline_predictions.csv").exists()
    with pytest.raises(ValueError, match="overlap"):
        train_baselines(train, train, tmp_path)


def test_gnn_unknown_entities_and_empty_edges_have_finite_scores():
    accounts = pd.DataFrame([{"account_id": "A", "created_at": "2026-01-01T00:00:00Z"}])
    attributes = pd.DataFrame(columns=["attribute_id", "attribute_type", "value"])
    links = pd.DataFrame(columns=["account_id", "attribute_id", "observed_at"])
    events = pd.DataFrame(columns=["event_id", "account_id", "device_id", "ip_id", "event_time",
                                   "received_at", "region", "outcome", "auth_method", "browser", "device_quality"])
    graph, _ = build_snapshot(accounts, attributes, links, events, "2026-01-02T00:00:00Z")
    node_types, edge_types = graph.metadata()
    dimensions = {node: int(graph[node].x.shape[1]) for node in node_types}
    model = HeterogeneousGraphSAGE(node_types, edge_types, dimensions, len(BEHAVIOR_FEATURES))
    indices = {node: torch.tensor([-1, -1]) for node in ["account", "device", "ip"]}
    result = model(graph, indices, torch.zeros((2, len(BEHAVIOR_FEATURES))))
    assert result.shape == (2, 2)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert model.unknown["device"].grad is not None


def test_gnn_training_persists_predictions_and_uses_train_cutoffs(tmp_path):
    train, validation, test = frame("2026-01-01", 24), frame("2026-02-01", 24), frame("2026-03-01", 12)
    joined = pd.concat([train, validation, test], ignore_index=True)
    accounts = pd.DataFrame({"account_id": joined.account_id.unique(), "created_at": "2025-12-30T00:00:00Z"})
    attributes = pd.DataFrame(columns=["attribute_id", "attribute_type", "value"])
    links = pd.DataFrame(columns=["account_id", "attribute_id", "observed_at"])
    events = joined[["event_id", "account_id", "event_time"]].copy()
    events["received_at"] = events.event_time
    events["device_id"] = [f"device-{i % 5}" for i in range(len(events))]
    events["ip_id"] = "shared-ip"
    events["region"], events["outcome"], events["auth_method"] = "region-a", "success", "password"
    events["browser"], events["device_quality"] = "browser-a", "stable"
    result = train_gnn(accounts, attributes, links, events, train, validation,
                       {"future_time": test}, tmp_path, epochs=1)
    predictions = pd.read_csv(tmp_path / "gnn_predictions.csv")
    assert set(predictions.split) == {"train", "validation", "future_time"}
    assert result["served_by_api"] is False
    assert (tmp_path / "gnn_weights.pt").exists()
    assert (tmp_path / "gnn_metadata.json").exists()
    assert all(pd.Timestamp(cutoff) < pd.Timestamp(validation.event_time.min())
               for cutoff in result["training_graph_cutoffs"])
    assert predictions[["takeover_score", "identity_score"]].map(lambda value: 0 <= value <= 1).all().all()
