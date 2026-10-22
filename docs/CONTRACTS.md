# IdentityGraph AI v1.0 contracts

All data is fictional. This is a local research prototype, not an identity verification service.

## Files and interfaces

Simulator output: `data/<run>/observed/{accounts,attributes,account_attributes,events}.parquet`
and separate `truth/{accounts,events,groups}.parquet`, plus `manifest.json`.

Observed accounts: account_id (string), created_at (UTC ISO string).
Observed attributes: attribute_id, attribute_type (email, phone, address), value (fictional normalized string).
Observed account_attributes: account_id, attribute_id, observed_at (UTC ISO string).
Observed events: event_id, account_id, device_id, ip_id, event_time, received_at (UTC ISO strings),
region, outcome (success/failure), auth_method (password/mfa), browser, device_quality (stable/reset).
Optional pre-decision telemetry: session_binding (verified/unverified/mismatch/unknown),
challenge_failures (integer 0-10), recovery_used (boolean). Defaults are unknown, 0, false.
Missing legacy columns and missing dataframe cells use the same defaults.
All events have unique IDs and are ordered by received_at, event_time, event_id. Receipt equals event time
in this ordered-replay implementation. UTC strings are normalized before sorting.
Event and account IDs never encode labels. No real passwords or personal data are generated.

Truth accounts: account_id, is_synthetic (0/1), owner_group, attack_group (empty if benign),
scenario, cohort (train/validation/test/cold_start), label_available_at.
Truth events: event_id, is_takeover (0/1), scenario, attack_group, label_available_at.
Truth groups: attack_group, account_id (malicious identity membership only).
Truth is never imported by feature construction, the graph builder, or serving.

## Time and targets

Takeover target: unauthorized actor in a successful session. Failed attempts remain context and are
not scored by the successful-session model. Identity target: fabricated claimed account identity,
assessed at a given successful session time. A per-account latest assessment is used for identity metrics.
Use only observed account metadata and attributes available by the query time.
The event itself is query context. Historical counts use earlier events exclusively.
Equal-time events are scored as one bucket before any event in that bucket updates history.
Late events are rejected in serving. Duplicate identical events return the stored result;
duplicate event IDs with changed contents fail. Unknown accounts require registration before scoring;
unknown devices and IPs are supported. Models and thresholds are fitted on training/validation only.

## Feature interface (features.py)

`HistoryState(accounts: pandas.DataFrame, attributes: pandas.DataFrame,
account_attributes: pandas.DataFrame)` stores observable history only.
`features(event: dict) -> dict[str, float]` excludes current event history;
`update(event: dict) -> None` adds observable history.
`build_feature_frame(accounts, attributes, account_attributes, events) -> pandas.DataFrame`
returns event_id, account_id, event_time plus numeric feature columns for successful sessions.
Export BEHAVIOR_FEATURES and GRAPH_FEATURES lists. Numeric model features must be allowlisted.

## Graph interface (graph.py)

`build_snapshot(accounts, attributes, account_attributes, events, cutoff: str)` builds a
PyG HeteroData snapshot using events strictly before cutoff and metadata observed by cutoff.
Returns `(data, mappings)` with account/device/ip/attribute node ID mappings.
`candidate_clusters(accounts, attributes, account_attributes, events, cutoff: str)` returns
list of dicts containing cluster_id, account_ids, evidence, and score. No truth inputs.
`account_graph(account_id, accounts, attributes, account_attributes, events, cutoff: str)` returns
bounded JSON `{nodes: [...], edges: [...]}` for the investigator interface.
Shared IP edges alone never create identity candidates. No irreversible identity merges.
Fuzzy matching uses only fictional observable strings. Record exact/similarity evidence separately.

## Model interface (models.py)

`train_baselines(train: DataFrame, validation: DataFrame, output_dir: Path)`:
frames already contain feature allowlist and labels is_takeover/is_synthetic; return summary dict.
Train separate targets with rules, logistic, behavioral LightGBM and graph-feature LightGBM.
Device-history and session-assurance ablations are evaluated but cannot win serving selection.
Persist a selected calibrated estimator/threshold per target in `bundle.joblib`.
`load_bundle(path)` loads trusted local artifacts. `predict_bundle(bundle, features: dict)` returns
`{takeover: {score, threshold, decision}, identity: {...}}`, with score conditional on simulation.
Use validation only for calibration/threshold selection; report model versions and feature lists.

## GNN interface (gnn.py)

`train_gnn(accounts, attributes, account_attributes, events, train_frame, validation_frame,
test_frames: dict[str, DataFrame], output_dir: Path, epochs: int = 12) -> dict`.
Use observable historical snapshots only. For queries, pool account/device/ip embeddings and
current behavioral features with separate supervised heads. Train snapshots must not contain
validation/test future edges. Report exact cutoff treatment and cold-start fallback.
Keep graph experiment separate from the selected API baseline unless its evidence justifies promotion.
Persist weights and prediction tables including event_id, both target scores, and split.

## API and dashboard (api.py, static/)

`create_app(project_root: Path | None = None)` discovers run under `artifacts/current.json`:
`{"data_dir":"data/demo", "artifact_dir":"artifacts/demo"}`. Resolve paths under project root.
An omitted project root means the current working folder, including for installed wheels.
Expose health, dashboard summary, accounts, clusters, account graph, event ingest/login score,
identity assess, session assessment, monitoring, and demo replay endpoints. Serve dashboard at `/`.
SQLite assessment/event ledger is under runtime/. No automatic blocking or outbound actions.
Use explicit manual-review/low-risk/unavailable states and disclose simulated data in the UI.
Every score records its model version, feature version, history cutoff, and evidence.
Anomaly percentiles are separately labeled. Feedback is stored separately, immutable per
event/target, and included in reported performance only after its availability timestamp.
Feedback never changes a stored prediction or enters feature history.

## Evaluation

Chronological train/validation/test plus cold-start accounts, new devices, unseen attack family,
and separately generated shifted/reserved populations. Calibrate only on validation. Do not tune on test results.
Report AP, Brier, fixed validation threshold recall/FPR/precision, false alerts per 1,000 benign
sessions, recall at a validation-selected FPR limit, precision at a fixed review budget,
and per-account identity metrics. Candidate coverage includes all malicious groups, including misses.
No completion requirement that a GNN wins. Save limitations and exact runtime/dependency evidence.

Node2Vec uses only the six-day warmup graph and trains its downstream readout on subsequent
training observations. Its matched baseline uses the same rows. Link prediction targets a
first future account-device observation among known nodes; it is not a fraud target. Isolation
Forest is unsupervised and converted to a percentile against validation scores. PSI uses frozen
validation distributions, and cannot be interpreted as accuracy without delayed labels.

IEEE-CIS is a separate, optional transaction-fraud experiment. The command must validate locally
supplied CSVs, preserve chronological boundaries and report missing data without fabricating results.
