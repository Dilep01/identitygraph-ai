# Local investigation API

The FastAPI application serves the investigator dashboard at `/` and interactive endpoint documentation at `/docs`. Start it through the project CLI. Bind it to `127.0.0.1` and use one worker: this research application has no authentication or multi-process state coordination.

`create_app(project_root=None)` uses the current working folder unless an explicit project root is supplied. The CLI accepts `--project-root` before the subcommand. It resolves `artifacts/current.json` under that folder. The pointer names `data_dir` and `artifact_dir`; both must resolve inside it. Models load from `<artifact_dir>/models/bundle.joblib`. Only locally generated, trusted model files should be used because joblib deserialization executes Python objects.

Serving reads the four observed Parquet files and the run manifest. It never opens the simulator truth tables. Offline evaluation results are displayed separately from replay assessments.

## Replay and temporal boundaries

The manifest must declare `replay_cutoff` (or `split_cutoffs.validation_cutoff`). Events strictly before that time seed observable history. Events at or after that time form the held-out demo queue. The read-only queue overview exposes counts and the next timestamp; it does not disclose future graph evidence or labels.

Each successful event is assessed before it updates history. Historical features exclude all events with the same timestamp, including equal-time events already ingested. Replay steps complete the selected equal-time bucket, so requesting one event can process more than one event. Failed authentications update context without receiving a successful-session risk score. Receipt time must equal event time in this implementation.

After replay, account details, graph neighborhoods, and candidates include ingested observations through the last event. Their exclusive history cutoff is one microsecond after the latest ingested timestamp. Stored assessment evidence retains the original event-time cutoff and is not recomputed using later history. Metadata is filtered by its observed/creation time.

An event earlier than the watermark receives `409`. An identical duplicate receives its stored result with `idempotent_replay: true`; changing the contents of an existing event ID receives `409`. A seed-history event cannot be rescored through the ingestion endpoint. Unknown devices and IPs are accepted. Unknown or not-yet-created accounts receive `404` until registered. Registration adds an account without fabricated attributes and requires a creation time at or after the watermark.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Readiness and model availability; returns a readable unavailable state before artifacts exist. |
| `GET /api/summary` | Available entity counts, recent stored assessments, model metadata, and offline evaluation. |
| `GET /api/accounts?q=&limit=30` | Search available account IDs; at most 100 results per request. |
| `POST /api/accounts` | Register an observed account with `account_id` and timezone-aware `created_at`. |
| `GET /api/accounts/{account_id}` | Available attributes, last 50 observed events, device history, latest stored assessment. |
| `GET /api/accounts/{account_id}/graph` | Bounded graph neighborhood; graph output is cached until history changes. |
| `GET /api/accounts/{account_id}/identity-graph` | Alias for the graph endpoint. |
| `GET /api/clusters` | Observable candidate groups; scores are explicitly heuristics from 0 to 100. |
| `GET /api/identity-clusters` | Alias for candidate groups. |
| `POST /api/events` | Ingest an observed authentication event and score if successful. |
| `POST /api/login/score` | Alias for event ingestion. |
| `GET /api/sessions/{event_id}` | Retrieve the original stored assessment/context record. |
| `GET /api/sessions/{event_id}/risk` | Alias for the saved session assessment. |
| `GET /api/sessions/{event_id}/export` | Download the saved assessment and evidence as JSON. |
| `GET /api/identity/{account_id}` | Latest identity score and its original evidence/cutoff; `404` when no score exists. |
| `POST /api/identity/assess` | Body `{"account_id":"..."}`; retrieves that latest stored identity assessment. |
| `GET /api/monitoring` | Recent counts, review rate, frozen-reference PSI and available feedback metrics. |
| `GET /api/monitoring/identity` | Alias for monitoring. |
| `POST /api/feedback` | Record an immutable delayed outcome for an assessed event and target. |
| `GET /api/monitoring/performance?as_of=` | Performance from submitted outcomes available by the cutoff, no later than the replay watermark. |
| `GET /api/demo/replay` | Read-only held-out replay progress. |
| `POST /api/demo/replay` | Advance the ordered queue; body `{"events":25}`, allowed range 1–100 plus equal-time bucket completion. |
| `POST /api/demo/reset` | Clear replay events, assessments and feedback for this run; keep source data, models, and registrations. |

## Event request and result

```json
{
  "event_id": "local-event-001",
  "account_id": "an-existing-observed-account-id",
  "device_id": "fictional-device",
  "ip_id": "fictional-network",
  "event_time": "2026-02-07T12:00:00Z",
  "received_at": "2026-02-07T12:00:00Z",
  "region": "fictional-region",
  "outcome": "success",
  "auth_method": "password",
  "browser": "fictional-browser",
  "device_quality": "stable",
  "session_binding": "unknown",
  "challenge_failures": 0,
  "recovery_used": false
}
```

Omitted `received_at` defaults to `event_time`. Unknown fields, including labels, are rejected. A successful assessment returns separate `takeover` and `identity` objects containing `score`, `threshold`, and `decision` (`manual_review` or `low_risk`). The UI describes `low_risk` as “below threshold”; it is not a guarantee of legitimacy. Each result stores `model_version`, `model_artifact_sha256`, `feature_version`, `history_cutoff`, the strictly-before boundary, and numeric observable `evidence`. The artifact hash identifies the fitted model file; the model version identifies its schema. Evidence values are model inputs, not causal explanations or SHAP values. A missing/unloadable model returns `503`; it never becomes a low-risk decision.

SQLite ledgers live under `runtime/`, keyed by run source and event content. Assessment and event rows are committed together. Restarting restores feature history from the observable seed and durable event ledger. Replay state is local and has no outbound integrations or account-blocking actions. If custom ingestion moves beyond queued demo events, those earlier unprocessed queue events are reported as skipped/late rather than silently scored out of order.

The optional session telemetry defaults to `unknown`, 0 and false. Binding values are `verified`, `unverified`, `mismatch` or `unknown`; challenge failures must be an integer from 0 to 10. These observations must exist before the decision. Model outputs remain conditional on simulated validation data.

An anomaly result, when the artifact is present, is a validation percentile from an unsupervised model. It is separate from the two decision scores. Case exports preserve the saved evidence and original cutoff rather than rebuilding them from later history.

## Delayed outcomes

Submit `event_id`, `target` (`takeover` or `identity`), `label` (0 or 1), timezone-aware `observed_at` and a nonempty `source` to `/api/feedback`. Here `observed_at` means when that outcome became available. The event must have an assessment. An identical repeat is idempotent; changed feedback for the same event/target receives `409`. Availability before the event receives `422`. Future-available feedback can be stored, but is excluded from performance until the replay watermark reaches it. The separate table survives restart and never updates features, predictions or models.

The dashboard includes automatic replay with pause, account search, relationship graphs, device history, stored evidence/export, candidate groups, offline comparisons, research results, PSI summaries and delayed-outcome counts. Graphs show current history; stored assessments retain their original cutoff. PSI is descriptive, and feedback may be selectively reported. Neither is a production validation claim.
