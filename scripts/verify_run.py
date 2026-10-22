"""Verify the current research artifacts and exact offline/online scoring parity.

Uses only project-local generated data. Run before handing the demo to an investigator:
this check resets the current demo replay ledger at the beginning and end.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from identitygraph.api import create_app
from identitygraph.features import BEHAVIOR_FEATURES, GRAPH_FEATURES
from identitygraph.models import load_bundle, selected_bundle_predictions
from identitygraph.simulator import digest


def main():
    root = Path(__file__).resolve().parents[1]
    if "--reset-demo" not in sys.argv:
        raise SystemExit("Pass --reset-demo to verify and reset the generated demonstration replay ledger.")
    pointer = json.loads((root / "artifacts/current.json").read_text())
    data = root / pointer["data_dir"]
    artifacts = root / pointer["artifact_dir"]
    manifest = json.loads((data / "manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        assert digest(data / filename) == expected["sha256"], filename
    checksums = json.loads((artifacts / "checksums.json").read_text())
    for filename, expected in checksums.items():
        assert digest(artifacts / filename) == expected, filename
    frame = pd.read_parquet(artifacts / "features/labeled_features.parquet")
    frame = frame[frame.event_time >= manifest["replay_cutoff"]]
    bundle = load_bundle(artifacts / "models")
    offline = selected_bundle_predictions(bundle, frame).set_index("event_id")
    features = frame.set_index("event_id")
    feature_names = BEHAVIOR_FEATURES + GRAPH_FEATURES
    client = TestClient(create_app(root))
    assert client.get("/api/health").json()["status"] == "ready"
    assert client.post("/api/demo/reset").status_code == 200
    total, scored, max_error, first = 0, 0, 0.0, None
    while client.get("/api/demo/replay").json()["remaining_events"]:
        response = client.post("/api/demo/replay", json={"events": 100})
        assert response.status_code == 200, response.text
        for result in response.json()["assessments"]:
            total += 1
            if first is None:
                first = result
            if result["status"] == "context_only":
                assert "takeover" not in result
                continue
            scored += 1
            row = offline.loc[result["event_id"]]
            feature_row = features.loc[result["event_id"]]
            np.testing.assert_allclose([result["evidence"][key] for key in feature_names],
                                       feature_row[feature_names].to_numpy(dtype=float), rtol=0, atol=0)
            for target in ("takeover", "identity"):
                error = abs(row[f"{target}_score"] - result[target]["score"])
                max_error = max(max_error, float(error))
                assert error < 1e-12, (result["event_id"], target, error)
            assert result["model_version"] == bundle["model_version"]
            assert result["feature_version"] == bundle["feature_version"]
    assert scored == len(offline)
    restarted = TestClient(create_app(root))
    assert restarted.get("/api/sessions/" + first["event_id"]).json()["event_id"] == first["event_id"]
    summary = restarted.get("/api/summary").json()
    assert summary["assessments"] == scored
    # A real stored event remains idempotent after restarting the application.
    events = pd.read_parquet(data / "observed/events.parquet")
    first_payload = events[events.event_id == first["event_id"]].iloc[0].to_dict()
    assert restarted.post("/api/events", json=first_payload).json()["idempotent_replay"]
    graph = restarted.get("/api/accounts/" + first["account_id"] + "/graph")
    assert graph.status_code == 200
    assert restarted.post("/api/demo/reset").status_code == 200
    result = {"run": artifacts.name, "data_checksums_verified": len(manifest["files"]),
              "artifact_checksums_verified": len(checksums), "events_replayed": total,
              "successful_session_scores_verified": scored, "max_score_difference": max_error,
              "all_feature_values_exact": True, "restart_persistence_verified": True,
              "duplicate_idempotency_verified": True, "replay_reset_after_check": True}
    output = root / "artifacts/verification" / artifacts.name / "run_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
