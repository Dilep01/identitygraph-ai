"""Information-boundary, graph-shape, and linkage-evidence checks."""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from identitygraph.features import (
    BEHAVIOR_FEATURES, EVENT_COLUMNS, GRAPH_FEATURES, HistoryState,
    build_feature_frame, normalize_attribute,
)
from identitygraph.graph import (
    EDGE_TYPES, MAX_GRAPH_EDGES, MAX_GRAPH_NODES, NODE_TYPES,
    account_graph, build_snapshot, broad_candidate_clusters as candidate_clusters,
)

T0 = "2026-01-01T00:00:00Z"
T1 = "2026-01-02T00:00:00Z"
T2 = "2026-01-03T00:00:00Z"
T3 = "2026-01-04T00:00:00Z"


def event(identifier="e1", account="a", device="d1", ip="ip1", at=T1,
          outcome="success", **extra):
    return {"event_id": identifier, "account_id": account, "device_id": device,
            "ip_id": ip, "event_time": at, "received_at": at, "region": "r1",
            "outcome": outcome, "auth_method": "password", "browser": "browser1",
            "device_quality": "stable", **extra}


def metadata(account_ids=("a", "b", "c"), attrs=(), links=()):
    accounts = pd.DataFrame([{"account_id": key, "created_at": T0} for key in account_ids])
    attributes = pd.DataFrame(attrs, columns=["attribute_id", "attribute_type", "value"])
    account_attributes = pd.DataFrame(links, columns=["account_id", "attribute_id", "observed_at"])
    return accounts, attributes, account_attributes


def event_frame(rows=()):
    return pd.DataFrame(rows, columns=EVENT_COLUMNS + sorted(set().union(*(row.keys() for row in rows)) - set(EVENT_COLUMNS)))


def test_feature_allowlist_and_initial_history():
    state = HistoryState(*metadata())
    values = state.features(event(is_takeover=1, owner_group="hidden"))
    assert list(values) == BEHAVIOR_FEATURES + GRAPH_FEATURES
    assert all(isinstance(value, float) for value in values.values())
    assert values["prior_login_count"] == 0
    assert values["new_device"] == 1
    assert values["missing_attribute_type_count"] == 3
    assert set(values).isdisjoint({"is_takeover", "owner_group", "event_id"})


def test_equal_time_and_future_history_are_excluded_even_after_updates():
    state = HistoryState(*metadata())
    query = event("query", at=T2)
    state.update(event("past", at=T1, outcome="failure"))
    expected = state.features(query)
    state.update(event("same_time", account="b", at=T2))
    state.update(event("future", account="a", device="future-device", at=T3))
    assert state.features(query) == expected
    assert expected["prior_failure_count"] == 1
    assert expected["device_other_account_count"] == 0


def test_batch_equal_time_bucket_and_failed_attempt_context():
    rows = [event("b-success", "b"), event("a-failure", outcome="failure"),
            event("a-success", at=T2)]
    frame = build_feature_frame(*metadata(), event_frame(rows))
    assert set(frame.event_id) == {"b-success", "a-success"}
    initial = frame.set_index("event_id").loc["b-success"]
    later = frame.set_index("event_id").loc["a-success"]
    assert initial["device_other_account_count"] == 0
    assert later["prior_failure_count"] == 1
    assert later["device_other_account_count"] == 1
    pd.testing.assert_frame_equal(frame, build_feature_frame(*metadata(), event_frame(rows[::-1])))


def test_observed_metadata_uses_availability_not_complete_dataset():
    tables = metadata(attrs=[("p", "phone", "phone-token-ab123cd45")],
                      links=[("a", "p", T0), ("b", "p", T3)])
    state = HistoryState(*tables)
    assert state.features(event(at=T2))["shared_attribute_account_count"] == 0
    assert state.features(event(at=T3))["shared_attribute_account_count"] == 1
    # Querying an earlier time again must not reuse future metadata.
    assert state.features(event(at=T2))["shared_attribute_account_count"] == 0


def test_account_creation_is_an_information_boundary():
    accounts, attributes, links = metadata()
    accounts.loc[accounts.account_id == "b", "created_at"] = T3
    state = HistoryState(accounts, attributes, links)
    with pytest.raises(ValueError, match="does not exist"):
        state.features(event(account="b", at=T2))
    with pytest.raises(ValueError, match="Unknown account"):
        state.features(event(account="missing"))


def test_future_rows_and_all_label_columns_cannot_change_prior_features():
    tables = metadata(attrs=[("p", "phone", "phone-token-abc123abc")],
                      links=[("a", "p", T0), ("b", "p", T0)])
    original = event_frame([event(), event("e2", "b", at=T2)])
    expected = build_feature_frame(*tables, original)
    polluted_tables = tuple(table.assign(is_synthetic=1, attack_group="secret", label=99) for table in tables)
    altered = pd.concat([original.assign(is_takeover=1, scenario="answer"),
                         event_frame([event("future", "b", "future-device", at=T3)])], ignore_index=True)
    actual = build_feature_frame(*polluted_tables, altered).iloc[:len(expected)].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual)


def test_update_idempotency_and_changed_duplicate_rejection():
    state = HistoryState(*metadata())
    state.update(event())
    state.update(event(is_takeover=1))
    assert state.features(event("next", at=T2))["prior_login_count"] == 1
    with pytest.raises(ValueError, match="changed contents"):
        state.update(event(device="different"))
    with pytest.raises(ValueError, match="Duplicate event IDs"):
        build_feature_frame(*metadata(), event_frame([event(), event()]))


@pytest.mark.parametrize("change", [
    {"received_at": T2}, {"outcome": "unknown"}, {"auth_method": "unknown"},
    {"device_quality": "unknown"}, {"event_time": "2026-01-02T00:00:00"},
])
def test_invalid_event_rejected(change):
    with pytest.raises(ValueError):
        HistoryState(*metadata()).features(event(**change))


def test_phone_tokens_do_not_collapse_to_digits():
    assert normalize_attribute("phone", "PHONE-TOKEN-ABC123DEF") == "phone-token-abc123def"
    assert normalize_attribute("phone", "phone-token-xyz123uvw") != normalize_attribute("phone", "phone-token-abc123def")
    assert normalize_attribute("phone", "+99 (123) 456") == "99123456"


def test_snapshot_node_shapes_reverse_edges_and_strict_cutoff():
    tables = metadata(attrs=[("p", "phone", "phone-token-abc123def")], links=[("a", "p", T0)])
    data, mappings = build_snapshot(*tables, event_frame([event(), event("same", "b", "same-device", at=T2)]), T2)
    assert set(data.node_types) == set(NODE_TYPES)
    assert "same-device" not in mappings["device"]
    for kind in NODE_TYPES:
        assert data[kind].x.shape == (len(mappings[kind]), 4)
        assert torch.isfinite(data[kind].x).all()
    for source, relation, target in EDGE_TYPES:
        forward = data[(source, relation, target)].edge_index
        reverse = data[(target, "rev_" + relation, source)].edge_index
        assert torch.equal(reverse, forward.flip(0))
    assert data.validate()


def test_empty_snapshot_keeps_complete_relation_schema():
    data, mappings = build_snapshot(*metadata(), event_frame(), T1)
    assert len(data.edge_types) == 8
    assert data["device"].x.shape == (0, 4)
    assert not mappings["device"]
    for relation in data.edge_types:
        assert data[relation].edge_index.shape == (2, 0)
    assert data.validate()


def test_snapshot_invariance_to_future_metadata_events_and_labels():
    tables = metadata(attrs=[("p", "phone", "phone-token-abc123def")], links=[("a", "p", T0)])
    initial, mapping = build_snapshot(*tables, event_frame([event()]), T2)
    accounts, attrs, links = [table.assign(is_synthetic=1) for table in tables]
    accounts = pd.concat([accounts, pd.DataFrame([{"account_id": "future-account", "created_at": T3}])], ignore_index=True)
    attrs = pd.concat([attrs, pd.DataFrame([{"attribute_id": "future-attr", "attribute_type": "address", "value": "fictional avenue"}])], ignore_index=True)
    links = pd.concat([links, pd.DataFrame([{"account_id": "a", "attribute_id": "future-attr", "observed_at": T3}])], ignore_index=True)
    changed, changed_mapping = build_snapshot(accounts, attrs, links,
        event_frame([event(is_takeover=1), event("future", "future-account", "future-device", at=T3)]), T2)
    assert mapping == changed_mapping
    for kind in NODE_TYPES:
        assert torch.equal(initial[kind].x, changed[kind].x)
    for relation in initial.edge_types:
        assert torch.equal(initial[relation].edge_index, changed[relation].edge_index)


def test_shared_ip_or_single_device_or_single_attribute_does_not_create_candidate():
    tables = metadata(attrs=[("p", "phone", "phone-token-abc123def")], links=[("a", "p", T0), ("b", "p", T0)])
    assert candidate_clusters(*metadata(), event_frame([event(device="da"), event("b1", "b", "db")]), T2) == []
    assert candidate_clusters(*metadata(), event_frame([event(), event("b1", "b")]), T2) == []
    assert candidate_clusters(*tables, event_frame(), T2) == []


def test_exact_attribute_candidates_corroborated_and_never_merge_accounts():
    tables = metadata(attrs=[("pa", "phone", "phone-token-abc123def"),
                             ("pb", "phone", "PHONE-TOKEN-ABC123DEF"),
                             ("ad", "address", "123 fictional avenue")],
                      links=[("a", "pa", T0), ("b", "pb", T0), ("a", "ad", T0), ("b", "ad", T0)])
    clusters = candidate_clusters(*tables, event_frame(), T2)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["account_ids"] == ["a", "b"]
    assert cluster["score"] == 50
    assert cluster["score_type"] == "heuristic_0_100"
    assert {e["attribute_type"] for e in cluster["evidence"]} == {"phone", "address"}
    assert {e["type"] for e in cluster["evidence"]} == {"exact_attribute"}
    _, mappings = build_snapshot(*tables, event_frame(), T2)
    assert len(mappings["account"]) == 3


def test_fuzzy_evidence_separate_from_exact_and_requires_corroboration():
    tables = metadata(attrs=[("aa", "address", "123 fictional demonstration avenue"),
                             ("ab", "address", "123 fictional demonstraton avenue")],
                      links=[("a", "aa", T0), ("b", "ab", T0)])
    assert candidate_clusters(*tables, event_frame(), T2) == []
    clusters = candidate_clusters(*tables, event_frame([event(), event("b1", "b")]), T2)
    assert len(clusters) == 1
    assert {item["type"] for item in clusters[0]["evidence"]} == {"similar_attribute", "shared_device"}
    similar = next(item for item in clusters[0]["evidence"] if item["type"] == "similar_attribute")
    assert 0.92 <= similar["similarity"] < 1
    assert clusters[0]["score"] == 35


def test_candidates_ignore_future_evidence_and_truth_and_are_deterministic():
    tables = metadata(attrs=[("p", "phone", "phone-token-abc123def")], links=[("a", "p", T0), ("b", "p", T0)])
    history = event_frame([event(), event("b1", "b")])
    expected = candidate_clusters(*tables, history, T2)
    assert expected
    altered_tables = tuple(table.assign(attack_group="hidden", scenario="secret") for table in tables)
    altered_history = pd.concat([history.assign(is_takeover=1), event_frame([event("future", "c", at=T3)])], ignore_index=True)
    assert candidate_clusters(*altered_tables, altered_history, T2) == expected
    assert candidate_clusters(*tables, history.iloc[::-1], T2) == expected


def test_bounded_explorer_has_valid_endpoints_and_displays_weak_ip_edges():
    accounts = tuple(f"a{i:03d}" for i in range(200))
    tables = metadata(account_ids=accounts)
    history = event_frame([event(f"e{i}", account, f"d{i}") for i, account in enumerate(accounts)])
    graph = account_graph(accounts[0], *tables, history, T2)
    assert len(graph["nodes"]) <= MAX_GRAPH_NODES
    assert len(graph["edges"]) <= MAX_GRAPH_EDGES
    ids = {node["id"] for node in graph["nodes"]}
    assert f"account:{accounts[0]}" in ids
    assert all(edge["source"] in ids and edge["target"] in ids for edge in graph["edges"])
    assert any(edge["type"] == "used_ip" for edge in graph["edges"])
    assert candidate_clusters(*tables, history, T2) == []
    with pytest.raises(ValueError, match="Unknown account"):
        account_graph("missing", *tables, history, T2)
