"""Completion checks focus on false-positive and information-boundary failures."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from identitygraph.api import create_app
from identitygraph.features import HistoryState, observable_event
from identitygraph.gnn import HeterogeneousGraphSAGE
from identitygraph.graph import build_snapshot, candidate_clusters, canonical_attribute_mapping
from identitygraph.ieee import run_ieee, temporal_relationship_features
from identitygraph.monitoring import feedback_metrics, population_shift
from identitygraph.research import fit_node2vec, link_examples
from identitygraph.simulator import SimulationConfig, generate, load_observed
from test_api import local_project  # noqa: F401
from test_features_graph import T0, T1, T2, T3, event, event_frame, metadata


def shared_tables():
    return metadata(attrs=[("p", "phone", "phone-token-shared"), ("ad", "address", "fictional lane 947801"),
                           ("mail", "email", "household@example.test")],
                    links=[(a, key, T0) for a in ("a", "b") for key in ("p", "ad", "mail")])


def stable_history(same_home=False):
    rows = []
    for account in ("a", "b"):
        for number in range(4):
            rows.append(event(f"{account}{number}", account, f"personal-{account}",
                              "home" if same_home else f"home-{account}", at=T1))
        rows.append(event(f"{account}-shared", account, "shared-device", f"home-{account}", at=T1))
    return event_frame(rows)


def test_household_sharing_is_not_enough_and_independent_corroboration_is_retained():
    assert candidate_clusters(*shared_tables(), stable_history(True), T2) == []
    clusters = candidate_clusters(*shared_tables(), stable_history(), T2)
    assert len(clusters) == 1 and clusters[0]["account_ids"] == ["a", "b"]
    assert any(e["type"] == "different_primary_networks" for e in clusters[0]["evidence"])
    assert candidate_clusters(*metadata(), stable_history(), T2) == []
    assert candidate_clusters(*shared_tables(), event_frame(), T2) == []


def test_corroborated_candidates_ignore_future_history_and_hidden_labels():
    original = candidate_clusters(*shared_tables(), stable_history(), T2)
    tables = [table.assign(is_synthetic=1, attack_group="oracle") for table in shared_tables()]
    history = pd.concat([stable_history(), event_frame([event("future", "c", "shared-device", at=T3)])])
    assert candidate_clusters(*tables, history.assign(is_takeover=1), T2) == original


def test_bridge_does_not_join_unrelated_accounts():
    attrs = [(f"{kind}{pair}", kind, f"{kind}-fictional-{pair}@example.test") for pair in ("ab", "bc") for kind in ("phone", "email")]
    links = [(a, f"{kind}{pair}", T0) for pair in ("ab", "bc") for a in pair for kind in ("phone", "email")]
    rows = [event(f"{a}{i}", a, f"personal-{a}", f"home-{a}", at=T1) for a in "abc" for i in range(6)]
    rows += [event(f"{a}-{pair}", a, f"device-{pair}", f"home-{a}", at=T1) for pair in ("ab", "bc") for a in pair]
    clusters = candidate_clusters(*metadata(attrs=attrs, links=links), event_frame(rows), T2)
    assert {tuple(c["account_ids"]) for c in clusters} == {("a", "b"), ("b", "c")}


def test_attribute_canonicalization_never_merges_accounts():
    attrs = pd.DataFrame([("a", "address", "Fictional LN 123"), ("b", "address", "fictional lane 123"),
                          ("blank", "address", "")], columns=["attribute_id", "attribute_type", "value"])
    aliases = canonical_attribute_mapping(attrs)
    assert aliases[0]["canonical_attribute_id"] == aliases[1]["canonical_attribute_id"]
    assert aliases[2]["canonical_attribute_id"] is None


def test_session_assurance_is_observable_optional_and_not_an_oracle():
    state = HistoryState(*metadata())
    baseline = state.features(event())
    assert baseline["binding_unknown"] == 1
    enhanced = event(session_binding="mismatch", challenge_failures=2, recovery_used=True)
    score_features = state.features(enhanced)
    assert score_features["binding_mismatch"] == 1 and score_features["challenge_failures"] == 2
    assert state.features({**enhanced, "is_takeover": 0}) == state.features({**enhanced, "is_takeover": 1})
    for invalid in ({"challenge_failures": -1}, {"recovery_used": "false"}, {"session_binding": "fraud"}):
        with pytest.raises(ValueError):
            observable_event(event(**invalid))


def test_simulator_overlaps_group_sizes_and_keeps_new_families_held_out(tmp_path):
    manifest = generate(tmp_path / "sim", SimulationConfig(seed=211, accounts=160, takeover_rate=.15))
    sizes = manifest["group_sizes"]
    assert len(set(sizes["fabricated"])) >= 2
    assert len(set(sizes["fabricated"]) & set(sizes["legitimate_households"])) >= 2
    observed = load_observed(tmp_path / "sim")
    truth = pd.read_parquet(tmp_path / "sim/truth/events.parquet")
    joined = observed["events"].merge(truth, on="event_id")
    held = joined[joined.scenario.isin(manifest["heldout_attack_families"])]
    assert set(held.scenario) == {"familiar_session", "session_replay"}
    assert (held.event_time >= manifest["replay_cutoff"]).all()
    assert ((joined.is_takeover == 0) & (joined.session_binding == "mismatch")).any()
    assert ((joined.is_takeover == 1) & (joined.session_binding == "verified")).any()


@pytest.mark.parametrize("architecture", ["graphsage", "rgcn", "gat", "relation_ablated"])
def test_graph_architectures_support_missing_nodes_and_roundtrip(architecture, tmp_path):
    graph, _ = build_snapshot(*metadata(), event_frame(), T1)
    dimensions = {node: 4 for node in graph.node_types}
    model = HeterogeneousGraphSAGE(*graph.metadata(), dimensions, 3, architecture=architecture)
    indices = {kind: torch.tensor([-1, 0 if kind == "account" else -1]) for kind in ("account", "device", "ip")}
    behavior = torch.zeros(2, 3)
    first = model(graph, indices, behavior)
    assert first.shape == (2, 2) and torch.isfinite(first).all()
    first.sum().backward()
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    restored = HeterogeneousGraphSAGE(*graph.metadata(), dimensions, 3, architecture=architecture)
    restored.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True))
    torch.testing.assert_close(model(graph, indices, behavior), restored(graph, indices, behavior), rtol=0, atol=0)


def test_node2vec_cannot_observe_future_edges_or_labels():
    tables = metadata()
    obs = dict(zip(("accounts", "attributes", "account_attributes"), tables))
    obs["events"] = event_frame([event(), event("b", "b")])
    first = fit_node2vec(obs, T2, dimensions=4, walks=1, epochs=1)
    changed = {key: value.assign(is_synthetic=1) for key, value in obs.items()}
    changed["events"] = pd.concat([changed["events"], event_frame([event("future", "c", "future-device", at=T3)])])
    second = fit_node2vec(changed, T2, dimensions=4, walks=1, epochs=1)
    assert first["names"] == second["names"]
    np.testing.assert_array_equal(first["vectors"], second["vectors"])


def test_temporal_link_examples_exclude_previous_links_and_future_metadata():
    obs = dict(zip(("accounts", "attributes", "account_attributes"), metadata()))
    obs["events"] = event_frame([event(), event("other", "b", "db"), event("link", "a", "db", at=T2)])
    frame, coverage = link_examples(obs, T2, T3)
    positive = frame[frame.label == 1]
    assert list(zip(positive.account_id, positive.device_id)) == [("a", "db")]
    assert ("a", "d1") not in set(zip(frame.account_id, frame.device_id))
    assert coverage["eligible_positive_links"] == 1


def test_shift_summary_is_descriptive_and_does_not_invent_performance():
    report = population_shift({"new_device": [0] * 100}, {"new_device": [1] * 100}, ["new_device"])
    assert report["features"]["new_device"]["status"] == "shift_observed"
    assert "accuracy" not in report
    assert population_shift({}, {}, ["missing"])["features"]["missing"]["status"] == "insufficient_data"


def test_delayed_feedback_never_changes_prediction_and_survives_restart(local_project):  # noqa: F811
    client = TestClient(create_app(local_project))
    first = client.post("/api/demo/replay", json={"events": 1}).json()["assessments"][0]
    payload = {"event_id": first["event_id"], "target": "takeover", "label": 1,
               "observed_at": "2026-01-03T03:00:00Z", "source": "test investigator"}
    assert client.post("/api/feedback", json=payload).status_code == 200
    assert client.get("/api/monitoring/performance").json()["targets"]["takeover"]["n"] == 0
    assert client.post("/api/feedback", json={**payload, "label": 0}).status_code == 409
    client.post("/api/demo/replay", json={"events": 100})
    restarted = TestClient(create_app(local_project))
    assert restarted.get("/api/monitoring/performance").json()["targets"]["takeover"]["n"] == 1
    assert restarted.get("/api/sessions/" + first["event_id"]).json() == first
    export = restarted.get("/api/sessions/" + first["event_id"] + "/export")
    assert export.status_code == 200 and export.json()["assessment"] == first
    assert "attachment" in export.headers["content-disposition"]
    assert restarted.post("/api/identity/assess", json={"account_id": first["account_id"]}).json() == restarted.get("/api/identity/" + first["account_id"]).json()
    assert restarted.get("/api/monitoring/performance?as_of=2030-01-01T00:00:00Z").status_code == 422
    assert restarted.post("/api/demo/reset").status_code == 200
    assert restarted.get("/api/monitoring/performance").json()["targets"]["takeover"]["n"] == 0


def test_identity_feedback_uses_latest_account_and_available_labels():
    result = {"status": "assessed", "account_id": "a", "event_time": T1, "identity": {"score": .8, "threshold": .5}}
    results = {"first": result, "second": {**result, "event_time": T2}}
    feedback = [{"event_id": key, "target": "identity", "label": 1, "observed_at": T2} for key in results]
    assert feedback_metrics(results, feedback, T1)["targets"]["identity"]["n"] == 0
    assert feedback_metrics(results, feedback, T3)["targets"]["identity"]["n"] == 1


def test_ieee_fixture_audit_and_time_safe_counts(tmp_path):
    transactions = pd.DataFrame({"TransactionID": range(100), "TransactionDT": np.arange(100) * 100,
                                 "TransactionAmt": np.arange(100) + 10, "isFraud": np.arange(100) % 5 == 0,
                                 "card1": np.arange(100) % 7, "ProductCD": "W"})
    transactions.isFraud = transactions.isFraud.astype(int)
    transactions.to_csv(tmp_path / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": range(0, 100, 2), "DeviceInfo": "fixture-device"}).to_csv(tmp_path / "train_identity.csv", index=False)
    report = run_ieee(tmp_path, tmp_path / "benchmark")
    assert report["status"] == "benchmark_complete" and report["join_coverage"] == .5
    assert report["split_counts"] == {"train": 60, "validation": 20, "test": 20}
    assert report["takeover_labels_available"] is False
    same_time = pd.DataFrame({"TransactionDT": [0, 0, 1], "card1": [1, 1, 1]})
    assert temporal_relationship_features(same_time).card1_prior_count.tolist() == [0, 0, 2]
    assert json.loads((tmp_path / "benchmark/audit.json").read_text())["files"]["train_identity.csv"]["rows"] == 50
    with pytest.raises(FileNotFoundError):
        run_ieee(tmp_path / "missing", tmp_path / "no-result")
    assert not (tmp_path / "no-result").exists()
