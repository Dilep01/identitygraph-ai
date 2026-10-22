import pandas as pd

from identitygraph.evaluation import cluster_metrics, metrics


def test_candidate_coverage_counts_wholly_missed_groups():
    truth = pd.DataFrame({"attack_group": ["one", "one", "two", "two"], "account_id": ["a", "b", "c", "d"]})
    result = cluster_metrics([{"account_ids": ["a", "b", "e"]}], truth, set("abcde"))
    assert result["candidate_group_coverage"] == 0.5
    assert result["missed_groups"] == ["two"]
    assert result["pair_precision"] == 1 / 3
    assert result["benign_accounts_flagged"] == 1


def test_identity_evaluation_counts_accounts_not_repeated_sessions():
    predictions = pd.DataFrame({"event_id": ["1", "2", "3"], "account_id": ["a", "a", "b"],
                                "event_time": ["2026-01-01", "2026-01-02", "2026-01-03"],
                                "label": [1, 1, 0], "score": [0.9, 0.8, 0.1],
                                "threshold": [0.5] * 3, "target": ["identity"] * 3})
    result = metrics(predictions, bootstrap=10)
    assert result["n"] == 2
    assert result["recall"] == 1
    assert result["false_alerts_per_1000"] == 0
