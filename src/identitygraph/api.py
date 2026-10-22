"""Local investigator API. Serving never opens simulator truth tables."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

UTC = timezone.utc


def normalized_time(value: str) -> str:
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            raise ValueError("Timezone required")
        return moment.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "Timestamps must be valid ISO 8601 values with a timezone.") from exc


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class LoginEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    event_id: str = Field(min_length=1, max_length=120)
    account_id: str = Field(min_length=1, max_length=120)
    device_id: str = Field(min_length=1, max_length=120)
    ip_id: str = Field(min_length=1, max_length=120)
    event_time: str
    received_at: str | None = None
    region: str = Field(min_length=1, max_length=80)
    outcome: Literal["success", "failure"] = "success"
    auth_method: Literal["password", "mfa"] = "password"
    browser: str = Field(default="unknown", max_length=80)
    device_quality: Literal["stable", "reset"] = "stable"
    session_binding: Literal["verified", "unverified", "mismatch", "unknown"] = "unknown"
    challenge_failures: int = Field(default=0, ge=0, le=10, strict=True)
    recovery_used: bool = Field(default=False, strict=True)


class Registration(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    account_id: str = Field(min_length=1, max_length=120)
    created_at: str


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: int = Field(default=1, ge=1, le=100)


class OutcomeFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    event_id: str = Field(min_length=1, max_length=120)
    target: Literal["takeover", "identity"]
    label: int = Field(ge=0, le=1, strict=True)
    observed_at: str
    source: str = Field(min_length=1, max_length=200)


class IdentityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    account_id: str = Field(min_length=1, max_length=120)


class LocalService:
    """One-process ordered replay service, with SQLite as the durable source of writes."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.lock = threading.RLock()
        self.loaded = False
        self.error = "Run the project build command to create local demo artifacts."
        self.bundle = None
        self.graph_cache = {}

    def inside(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Artifact paths must stay inside the project folder")
        return path

    def ensure_loaded(self):
        with self.lock:
            if self.loaded:
                return
            pointer = self.root / "artifacts" / "current.json"
            try:
                current = json.loads(pointer.read_text(encoding="utf-8"))
                self.data_dir = self.inside(current["data_dir"])
                self.artifact_dir = self.inside(current["artifact_dir"])
                manifest_text = (self.data_dir / "manifest.json").read_text(encoding="utf-8")
                self.manifest = json.loads(manifest_text)
                observed = self.data_dir / "observed"
                self.accounts = pd.read_parquet(observed / "accounts.parquet")
                self.attributes = pd.read_parquet(observed / "attributes.parquet")
                self.links = pd.read_parquet(observed / "account_attributes.parquet")
                self.source_events = pd.read_parquet(observed / "events.parquet")
                for frame, columns in [(self.accounts, ["created_at"]),
                                       (self.links, ["observed_at"]),
                                       (self.source_events, ["event_time", "received_at"])]:
                    for column in columns:
                        frame[column] = frame[column].map(normalized_time)
                self.source_events = self.source_events.sort_values(
                    ["received_at", "event_time", "event_id"]).reset_index(drop=True)
                cutoff = self.manifest.get("replay_cutoff") or self.manifest.get(
                    "split_cutoffs", {}).get("validation_cutoff")
                if not cutoff:
                    raise ValueError("Manifest must declare replay_cutoff; held-out replay cannot be guessed")
                self.replay_cutoff = normalized_time(cutoff)
                self.seed_events = self.source_events[
                    self.source_events.event_time < self.replay_cutoff].copy()
                self.queue = self.source_events[
                    self.source_events.event_time >= self.replay_cutoff].to_dict("records")
                self.seed_ids = set(self.seed_events.event_id)
                # Data checksum prevents a regenerated run from reusing an incompatible ledger.
                signature = hashlib.sha256((str(self.data_dir) + manifest_text).encode())
                signature.update(pd.util.hash_pandas_object(self.source_events, index=True).values.tobytes())
                runtime = self.root / "runtime"
                runtime.mkdir(parents=True, exist_ok=True)
                self.db_path = runtime / f"investigations-{signature.hexdigest()[:16]}.sqlite3"
                with self.connect() as db:
                    db.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, "
                               "fingerprint TEXT NOT NULL, payload TEXT NOT NULL, result TEXT NOT NULL)")
                    db.execute("CREATE TABLE IF NOT EXISTS registrations (account_id TEXT PRIMARY KEY, "
                               "created_at TEXT NOT NULL)")
                    db.execute("CREATE TABLE IF NOT EXISTS feedback (event_id TEXT NOT NULL, target TEXT NOT NULL, "
                               "payload TEXT NOT NULL, PRIMARY KEY(event_id,target))")
                    rows = db.execute("SELECT account_id, created_at FROM registrations").fetchall()
                    if rows:
                        registered = pd.DataFrame(rows, columns=["account_id", "created_at"])
                        self.accounts = pd.concat([self.accounts, registered], ignore_index=True)
                self.bundle_path = self.artifact_dir / "models" / "bundle.joblib"
                self._reload_model()
                self.anomaly = None
                anomaly_path = self.artifact_dir / "research/anomaly.joblib"
                if anomaly_path.exists():
                    import joblib
                    self.anomaly = joblib.load(anomaly_path)
                self._restore_history()
                self.loaded = True
                self.error = None
            except (OSError, ValueError, KeyError, ImportError, sqlite3.Error) as exc:
                self.error = f"Local demo artifacts unavailable: {exc}"
                raise HTTPException(503, self.error) from exc

    def connect(self):
        return sqlite3.connect(self.db_path)

    def _reload_model(self):
        from .models import load_bundle
        try:
            self.bundle = load_bundle(self.bundle_path) if self.bundle_path.is_file() else None
            self.model_hash = hashlib.sha256(self.bundle_path.read_bytes()).hexdigest() if self.bundle else None
        except (OSError, ValueError, EOFError, ImportError, KeyError) as exc:
            self.bundle = None
            self.error = "Local model artifact cannot be loaded; no risk decision was made."
            raise HTTPException(503, self.error) from exc

    def _restore_history(self):
        from .features import HistoryState
        self.state = HistoryState(self.accounts, self.attributes, self.links)
        self.live_events = []
        self.results = {}
        for event in self.seed_events.to_dict("records"):
            self.state.update(event)
        with self.connect() as db:
            rows = db.execute("SELECT payload, result FROM events ORDER BY rowid").fetchall()
        for payload, result in rows:
            event = json.loads(payload)
            self.state.update(event)
            self.live_events.append(event)
            self.results[event["event_id"]] = json.loads(result)
        self.watermark = max([self.replay_cutoff] + [e["event_time"] for e in self.live_events])
        self.revision = len(rows)
        self.graph_cache.clear()

    def cutoff(self):
        if not self.live_events:
            return self.replay_cutoff
        return normalized_time((datetime.fromisoformat(self.watermark.replace("Z", "+00:00"))
                                + timedelta(microseconds=1)).isoformat())

    def observed_events(self):
        if self.live_events:
            return pd.concat([self.seed_events, pd.DataFrame(self.live_events)], ignore_index=True)
        return self.seed_events

    def require_account(self, account_id, cutoff=None):
        cutoff = cutoff or self.cutoff()
        row = self.accounts[(self.accounts.account_id == account_id) & (self.accounts.created_at <= cutoff)]
        if row.empty:
            raise HTTPException(404, "Account is unknown at this time; register it before scoring.")
        return row.iloc[0].to_dict()

    def read_json(self, name):
        path = self.artifact_dir / name
        return json_safe(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None

    def ingest(self, incoming: LoginEvent):
        self.ensure_loaded()
        event = incoming.model_dump()
        event["event_time"] = normalized_time(event["event_time"])
        event["received_at"] = normalized_time(event["received_at"] or event["event_time"])
        if event["received_at"] != event["event_time"]:
            raise HTTPException(422, "This ordered-replay version requires received_at equal to event_time.")
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        with self.lock:
            with self.connect() as db:
                prior = db.execute("SELECT fingerprint, result FROM events WHERE event_id=?",
                                   (event["event_id"],)).fetchone()
            if prior:
                if prior[0] != fingerprint:
                    raise HTTPException(409, "This event ID already exists with different contents.")
                return {**json.loads(prior[1]), "idempotent_replay": True}
            if event["event_id"] in self.seed_ids:
                raise HTTPException(409, "This ID belongs to seeded historical context; no stored assessment exists.")
            if event["event_time"] < self.watermark:
                raise HTTPException(409, "Late events are rejected; event_time precedes the history watermark.")
            self.require_account(event["account_id"], event["event_time"])
            if event["outcome"] == "success":
                if self.bundle is None:
                    self._reload_model()
                if self.bundle is None:
                    raise HTTPException(503, "Scoring model unavailable. No risk decision was made.")
                from .models import predict_bundle
                feature_values = self.state.features(event)
                scores = json_safe(predict_bundle(self.bundle, feature_values))
                result = {
                    "event_id": event["event_id"], "account_id": event["account_id"],
                    "event_time": event["event_time"], "status": "assessed", **scores,
                    "model_version": self.bundle.get("model_version", "local-baseline-v1"),
                    "model_artifact_sha256": self.model_hash,
                    "feature_version": self.bundle.get("feature_version", "observable-v1"),
                    "history_cutoff": event["event_time"], "history_boundary": "strictly_before",
                    "evidence": json_safe(feature_values), "idempotent_replay": False,
                    "calibration_population": "simulated validation population",
                    "synthetic_data": True,
                }
                if self.anomaly is not None:
                    from .research import anomaly_scores
                    result["anomaly"] = {"percentile": float(anomaly_scores(self.anomaly, pd.DataFrame([feature_values]))[0]),
                                         "score_type": "validation_percentile_not_fraud_probability"}
            else:
                result = {"event_id": event["event_id"], "account_id": event["account_id"],
                          "event_time": event["event_time"], "status": "context_only",
                          "reason": "Failed authentication is historical context, not a successful-session score.",
                          "idempotent_replay": False, "synthetic_data": True}
            with self.connect() as db:
                db.execute("INSERT INTO events VALUES (?, ?, ?, ?)",
                           (event["event_id"], fingerprint, payload, json.dumps(result)))
            self.state.update(event)
            self.live_events.append(event)
            self.results[event["event_id"]] = result
            self.watermark = max(self.watermark, event["event_time"])
            self.revision += 1
            self.graph_cache.clear()
            return result


def create_app(project_root: Path | None = None) -> FastAPI:
    root = (Path(project_root) if project_root else Path.cwd()).resolve()
    app = FastAPI(title="IdentityGraph AI", version="1.0.0",
                  description="Local synthetic identity and authentication research. No automatic enforcement.")
    service = LocalService(root)
    app.state.service = service
    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    def health():
        try:
            service.ensure_loaded()
        except HTTPException:
            pass
        return {"status": "ready" if service.loaded and service.bundle is not None else "unavailable",
                "data_ready": service.loaded, "model_ready": service.bundle is not None,
                "detail": service.error, "synthetic_data": True}

    @app.get("/api/summary")
    def summary():
        service.ensure_loaded()
        cutoff = service.cutoff()
        accounts = service.accounts[service.accounts.created_at <= cutoff]
        events = service.observed_events()
        assessed = [r for r in service.results.values() if r["status"] == "assessed"]
        reviews = [r for r in assessed if any(r[t]["decision"] == "manual_review"
                                           for t in ["takeover", "identity"])]
        return {"synthetic_data": True, "accounts": len(accounts), "historical_events": len(events),
                "devices": int(events.device_id.nunique()), "assessments": len(assessed),
                "manual_reviews": len(reviews), "history_cutoff": cutoff,
                "model_ready": service.bundle is not None,
                "model_version": service.bundle.get("model_version") if service.bundle else None,
                "replay_cutoff": service.replay_cutoff,
                "evaluation": service.read_json("evaluation.json"),
                "model_metadata": service.read_json("models/model_metadata.json"),
                "research": service.read_json("research/summary.json"),
                "recent_assessments": list(reversed(assessed[-30:])),
                "limitations": ["Fictional observations and simulated labels only.",
                                "Scores are calibrated on simulated validation data.",
                                "Review decisions are advisory; this application never blocks accounts."]}

    @app.get("/api/accounts")
    def accounts(q: str = "", limit: int = Query(default=30, ge=1, le=100)):
        service.ensure_loaded()
        cutoff = service.cutoff()
        frame = service.accounts[(service.accounts.created_at <= cutoff)
                                 & service.accounts.account_id.str.contains(q, case=False, regex=False)]
        items = []
        for row in frame.head(limit).to_dict("records"):
            latest = next((r for r in reversed(list(service.results.values()))
                           if r["account_id"] == row["account_id"] and r["status"] == "assessed"), None)
            items.append({**row, "latest_assessment": latest})
        return {"items": items, "total": len(frame), "history_cutoff": cutoff}

    @app.post("/api/accounts", status_code=201)
    def register(account: Registration):
        service.ensure_loaded()
        created_at = normalized_time(account.created_at)
        with service.lock:
            if account.account_id in set(service.accounts.account_id):
                raise HTTPException(409, "Account ID already exists.")
            if created_at < service.watermark:
                raise HTTPException(409, "Registration cannot introduce metadata before the history watermark.")
            with service.connect() as db:
                db.execute("INSERT INTO registrations VALUES (?, ?)", (account.account_id, created_at))
            service.accounts = pd.concat([service.accounts, pd.DataFrame([
                {"account_id": account.account_id, "created_at": created_at}])], ignore_index=True)
            service._restore_history()
        return {"account_id": account.account_id, "created_at": created_at, "synthetic_data": True}

    @app.get("/api/accounts/{account_id}")
    def account_detail(account_id: str):
        service.ensure_loaded()
        row = service.require_account(account_id)
        cutoff = service.cutoff()
        events = service.observed_events()
        history = events[(events.account_id == account_id) & (events.event_time < cutoff)]
        links = service.links[(service.links.account_id == account_id) & (service.links.observed_at <= cutoff)]
        attributes = links.merge(service.attributes, on="attribute_id", how="left").to_dict("records")
        cases = [r for r in service.results.values() if r["account_id"] == account_id and r["status"] == "assessed"]
        return {**row, "history_cutoff": cutoff, "attributes": attributes,
                "events": history.tail(50).iloc[::-1].to_dict("records"),
                "device_history": history.groupby("device_id", sort=False).agg(
                    first_seen=("event_time", "min"), last_seen=("event_time", "max"),
                    sessions=("event_id", "count")).reset_index().to_dict("records"),
                "latest_assessment": cases[-1] if cases else None}

    @app.get("/api/accounts/{account_id}/graph")
    @app.get("/api/accounts/{account_id}/identity-graph")
    def graph(account_id: str):
        service.ensure_loaded()
        service.require_account(account_id)
        key = ("account", account_id, service.revision)
        if key not in service.graph_cache:
            from .graph import account_graph
            service.graph_cache[key] = account_graph(account_id, service.accounts, service.attributes,
                                                     service.links, service.observed_events(), service.cutoff())
        return json_safe(service.graph_cache[key])

    @app.get("/api/clusters")
    @app.get("/api/identity-clusters")
    def clusters():
        service.ensure_loaded()
        key = ("clusters", service.revision)
        if key not in service.graph_cache:
            from .graph import candidate_clusters
            service.graph_cache[key] = candidate_clusters(service.accounts, service.attributes, service.links,
                                                           service.observed_events(), service.cutoff())
        return {"items": json_safe(service.graph_cache[key]), "score_type": "heuristic_0_to_100",
                "history_cutoff": service.cutoff(), "confirmed_fraud": False}

    @app.post("/api/events")
    @app.post("/api/login/score")
    def ingest(event: LoginEvent):
        return service.ingest(event)

    @app.get("/api/sessions/{event_id}")
    @app.get("/api/sessions/{event_id}/risk")
    def session(event_id: str):
        service.ensure_loaded()
        result = service.results.get(event_id)
        if result is None:
            raise HTTPException(404, "No stored assessment or context record for this event.")
        return result

    @app.get("/api/sessions/{event_id}/export")
    def export_session(event_id: str):
        result = session(event_id)
        safe_name = hashlib.sha256(event_id.encode()).hexdigest()[:12]
        return JSONResponse(json_safe({"schema": "identitygraph-case-1", "assessment": result,
                                      "limitations": "Fictional data; relationships are not proof of ownership. Below threshold does not mean verified legitimate."}),
                            headers={"Content-Disposition": f'attachment; filename="case-{safe_name}.json"'})

    @app.post("/api/feedback")
    def feedback(value: OutcomeFeedback):
        service.ensure_loaded()
        with service.lock:
            result = session(value.event_id)
            if result["status"] != "assessed":
                raise HTTPException(422, "Feedback requires a scored successful session.")
            payload = value.model_dump()
            payload["observed_at"] = normalized_time(value.observed_at)
            if payload["observed_at"] < result["event_time"]:
                raise HTTPException(422, "A label cannot be available before its session.")
            with service.connect() as db:
                previous = db.execute("SELECT payload FROM feedback WHERE event_id=? AND target=?",
                                      (value.event_id, value.target)).fetchone()
                if previous:
                    if json.loads(previous[0]) != payload:
                        raise HTTPException(409, "Existing feedback differs; silent label replacement is not allowed.")
                    return {"stored": True, "idempotent": True}
                db.execute("INSERT INTO feedback VALUES(?,?,?)", (value.event_id, value.target, json.dumps(payload)))
            return {"stored": True, "idempotent": False, "used_for_scoring": False}

    @app.get("/api/monitoring/performance")
    def performance(as_of: str | None = None):
        service.ensure_loaded()
        from .monitoring import feedback_metrics
        cutoff = normalized_time(as_of) if as_of else service.watermark
        if cutoff > service.watermark:
            raise HTTPException(422, "Performance cutoff cannot exceed the event watermark.")
        with service.connect() as db:
            labels = [json.loads(row[0]) for row in db.execute("SELECT payload FROM feedback")]
        return feedback_metrics(service.results, labels, cutoff)

    @app.get("/api/identity/{account_id}")
    def identity(account_id: str):
        service.ensure_loaded()
        service.require_account(account_id)
        latest = next((r for r in reversed(list(service.results.values()))
                       if r["account_id"] == account_id and r["status"] == "assessed"), None)
        if latest is None:
            raise HTTPException(404, "No stored identity assessment. Replay or score a successful session first.")
        return {"account_id": account_id, "identity": latest["identity"], "event_id": latest["event_id"],
                "history_cutoff": latest["history_cutoff"], "model_version": latest["model_version"],
                "feature_version": latest["feature_version"], "evidence": latest["evidence"],
                "synthetic_data": True}

    @app.post("/api/identity/assess")
    def assess_identity(request: IdentityRequest):
        # Identity scores are tied to a real stored successful-session cutoff.
        return identity(request.account_id)

    @app.get("/api/monitoring")
    @app.get("/api/monitoring/identity")
    def monitoring():
        service.ensure_loaded()
        recent = service.live_events[-500:]
        assessments = [service.results[e["event_id"]] for e in recent
                       if service.results[e["event_id"]]["status"] == "assessed"]
        from .monitoring import population_shift
        reference = service.read_json("research/monitoring_reference.json") or {}
        values = pd.DataFrame([r["evidence"] for r in assessments]).to_dict("list")
        values.update({target + "_score": [r[target]["score"] for r in assessments] for target in ("takeover", "identity")})
        shift = population_shift(reference, values, list(reference))
        return {"window": "last 500 ingested events", "events": len(recent), "assessments": len(assessments),
                "failed_authentications": sum(e["outcome"] == "failure" for e in recent),
                "fingerprint_reset_events": sum(e["device_quality"] == "reset" for e in recent),
                "review_rate": sum(any(r[t]["decision"] == "manual_review" for t in ["takeover", "identity"])
                                   for r in assessments) / len(assessments) if assessments else None,
                "watermark": service.watermark, "model_ready": service.bundle is not None,
                "production_drift_detector": False,
                "shift": shift, "performance": performance(),
                "note": "Observable shifts are descriptive; accuracy is measured separately using submitted delayed feedback."}

    @app.get("/api/demo/replay")
    def replay_status():
        service.ensure_loaded()
        remaining = [e for e in service.queue if e["event_id"] not in service.results
                     and e["event_time"] >= service.watermark]
        skipped = sum(e["event_id"] not in service.results and e["event_time"] < service.watermark
                      for e in service.queue)
        return {"total_events": len(service.queue), "remaining_events": len(remaining),
                "replayed_events": sum(e["event_id"] in service.results for e in service.queue),
                "skipped_late_events": skipped, "next_event_time": remaining[0]["event_time"] if remaining else None,
                "history_cutoff": service.cutoff(), "seed_events": len(service.seed_events),
                "source": "held-out test events", "synthetic_data": True}

    @app.post("/api/demo/replay")
    def replay(request: ReplayRequest):
        service.ensure_loaded()
        with service.lock:
            remaining = [e for e in service.queue if e["event_id"] not in service.results
                         and e["event_time"] >= service.watermark]
            selected = remaining[:request.events]
            # Finish the selected equal-time bucket so subsequent UI snapshots are coherent.
            if selected:
                end_time = selected[-1]["event_time"]
                selected.extend(e for e in remaining[request.events:] if e["event_time"] == end_time)
            results = [service.ingest(LoginEvent.model_validate(e)) for e in selected]
            return {"processed": len(results), "assessments": results, "replay": replay_status()}

    @app.post("/api/demo/reset")
    def reset():
        service.ensure_loaded()
        with service.lock:
            with service.connect() as db:
                db.execute("DELETE FROM feedback")
                db.execute("DELETE FROM events")
            service._restore_history()
        return {"reset": True, "replay": replay_status()}

    return app
