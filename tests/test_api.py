"""Serving contract tests use real historical features and deterministic stub scores."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from identitygraph.api import create_app


def event(identifier, moment, account="acct-a", device="device-a", outcome="success"):
    return {"event_id": identifier, "account_id": account, "device_id": device, "ip_id": "ip-a",
            "event_time": moment, "received_at": moment, "region": "fictional-region",
            "outcome": outcome, "auth_method": "password", "browser": "fictional-browser",
            "device_quality": "stable"}


@pytest.fixture
def local_project(tmp_path, monkeypatch):
    from identitygraph import models
    observed = tmp_path / "data/demo/observed"
    observed.mkdir(parents=True)
    artifact = tmp_path / "artifacts/demo/models"
    artifact.mkdir(parents=True)
    (artifact / "bundle.joblib").write_bytes(b"fixture: loader patched")
    (tmp_path / "artifacts/current.json").write_text(json.dumps(
        {"data_dir": "data/demo", "artifact_dir": "artifacts/demo"}), encoding="utf-8")
    (tmp_path / "data/demo/manifest.json").write_text(json.dumps(
        {"replay_cutoff": "2026-01-03T00:00:00Z"}), encoding="utf-8")
    frames = {
        "accounts": pd.DataFrame([
            {"account_id": "acct-a", "created_at": "2026-01-01T00:00:00Z"},
            {"account_id": "acct-future", "created_at": "2026-01-05T00:00:00Z"}]),
        "attributes": pd.DataFrame([
            {"attribute_id": "attr-a", "attribute_type": "email", "value": "fiction@example.test"},
            {"attribute_id": "attr-future", "attribute_type": "phone", "value": "fake-phone-future"}]),
        "account_attributes": pd.DataFrame([
            {"account_id": "acct-a", "attribute_id": "attr-a", "observed_at": "2026-01-01T00:00:00Z"},
            {"account_id": "acct-a", "attribute_id": "attr-future", "observed_at": "2026-01-05T00:00:00Z"}]),
        "events": pd.DataFrame([
            event("e-seed", "2026-01-02T00:00:00Z"),
            event("e-first", "2026-01-03T01:00:00Z", device="device-new"),
            event("e-equal", "2026-01-03T01:00:00Z", device="device-new"),
            event("e-failure", "2026-01-03T02:00:00Z", outcome="failure"),
            event("e-later", "2026-01-04T00:00:00Z")]),
    }
    for name, frame in frames.items():
        frame.to_parquet(observed / f"{name}.parquet", index=False)
    monkeypatch.setattr(models, "load_bundle", lambda _: {
        "model_version": "test-bundle-v1", "feature_version": "test-features-v1"})

    def predict(_, features):
        score = .8 if features["new_device"] else .1
        return {target: {"score": score, "threshold": .5,
                         "decision": "manual_review" if score >= .5 else "low_risk"}
                for target in ["takeover", "identity"]}

    monkeypatch.setattr(models, "predict_bundle", predict)
    return tmp_path


def test_missing_artifacts_explain_unavailable(tmp_path):
    client = TestClient(create_app(tmp_path))
    health = client.get("/api/health").json()
    assert health["status"] == "unavailable"
    assert health["model_ready"] is False
    assert client.get("/api/summary").status_code == 503
    assert client.get("/").status_code == 200
    assert "SYNTHETIC DATA" in client.get("/").text


def test_replay_history_idempotency_and_restart(local_project):
    client = TestClient(create_app(local_project))
    before = client.get("/api/summary").json()
    assert before["accounts"] == 1 and before["historical_events"] == 1
    assert client.get("/api/accounts/acct-future").status_code == 404
    assert len(client.get("/api/accounts/acct-a").json()["attributes"]) == 1
    assert client.get("/api/identity/acct-a").status_code == 404
    replay = client.post("/api/demo/replay", json={"events": 1})
    assert replay.status_code == 200, replay.text
    result = replay.json()
    assert result["processed"] == 2  # complete equal-time bucket
    first, same_time = result["assessments"]
    assert first["evidence"]["prior_login_count"] == 1
    assert same_time["evidence"]["prior_login_count"] == 1
    assert first["evidence"]["new_device"] == same_time["evidence"]["new_device"] == 1
    assert first["evidence"]["attribute_count"] == 1
    assert first["takeover"]["decision"] == "manual_review"
    assert first["history_boundary"] == "strictly_before"
    payload = event("e-first", "2026-01-03T01:00:00Z", device="device-new")
    duplicate = client.post("/api/events", json=payload)
    assert duplicate.status_code == 200 and duplicate.json()["idempotent_replay"]
    changed = client.post("/api/events", json={**payload, "region": "changed"})
    assert changed.status_code == 409
    assert client.get("/api/summary").json()["assessments"] == 2
    restarted = TestClient(create_app(local_project))
    assert restarted.get("/api/sessions/e-first").json()["evidence"] == first["evidence"]
    assert restarted.get("/api/summary").json()["historical_events"] == 3
    assert restarted.get("/api/accounts/acct-a/graph").status_code == 200
    assert restarted.get("/api/clusters").json()["confirmed_fraud"] is False


def test_failure_context_unknown_entities_late_event_and_reset(local_project):
    client = TestClient(create_app(local_project))
    result = client.post("/api/demo/replay", json={"events": 3}).json()
    assert result["assessments"][-1]["status"] == "context_only"
    assert "takeover" not in result["assessments"][-1]
    late = event("late", "2026-01-03T00:30:00Z")
    assert client.post("/api/login/score", json=late).status_code == 409
    unknown = event("unknown", "2026-01-04T01:00:00Z", account="not-registered")
    assert client.post("/api/events", json=unknown).status_code == 404
    assert client.post("/api/accounts", json={"account_id": "not-registered",
                                              "created_at": "2026-01-04T00:00:00Z"}).status_code == 201
    assert client.post("/api/events", json=unknown).status_code == 200
    monitoring = client.get("/api/monitoring").json()
    assert monitoring["failed_authentications"] == 1
    assert monitoring["production_drift_detector"] is False
    assert client.post("/api/demo/reset").status_code == 200
    assert client.get("/api/summary").json()["assessments"] == 0
    assert client.get("/api/demo/replay").json()["remaining_events"] == 4


def test_missing_model_never_returns_low_risk(local_project):
    (local_project / "artifacts/demo/models/bundle.joblib").unlink()
    client = TestClient(create_app(local_project))
    assert client.get("/api/health").json()["data_ready"]
    assert client.get("/api/health").json()["model_ready"] is False
    response = client.post("/api/demo/replay", json={"events": 1})
    assert response.status_code == 503
    assert client.get("/api/summary").json()["assessments"] == 0
    assert client.get("/api/demo/replay").json()["replayed_events"] == 0


def test_validation_and_no_truth_access(local_project, monkeypatch):
    original = pd.read_parquet
    opened = []

    def read(path, *args, **kwargs):
        opened.append(str(path))
        assert "truth" not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    client = TestClient(create_app(local_project))
    assert client.get("/api/summary").status_code == 200
    assert len(opened) == 4
    bad = event("bad", "2026-01-04T00:00:00")
    assert client.post("/api/events", json=bad).status_code == 422
    good = event("new", "2026-01-04T00:00:00Z", device="previously-unseen")
    assert client.post("/api/events", json={**good, "is_takeover": 1}).status_code == 422
    assert client.post("/api/events", json={**good, "event_id": "  "}).status_code == 422
    assert client.post("/api/events", json=good).status_code == 200


def test_current_pointer_cannot_escape_project(tmp_path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/current.json").write_text(
        json.dumps({"data_dir": "../outside", "artifact_dir": "artifacts/demo"}), encoding="utf-8")
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/summary").status_code == 503
    assert "inside the project" in client.get("/api/health").json()["detail"]
