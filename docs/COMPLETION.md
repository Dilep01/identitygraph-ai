# IdentityGraph AI 1.0 completion record

The local research implementation is complete. The final synthetic experiment is `v1`,
seed 907, with 240 fictional accounts and 6,494 events. Its models are frozen; later
verification and documentation do not select new thresholds from test results.

This completion means that the implemented research workflow runs, saves evidence and
has been checked. It does not mean all attacks are detected. Both held-out takeover
families had zero recall at the selected threshold. See [RESULTS.md](RESULTS.md) before
using benchmark numbers in a portfolio or interview.

## Proposal coverage

| Phase | Delivered implementation | Evidence |
|---|---|---|
| 1. Data contracts and external-data audit | Separate observable/truth schemas; local IEEE-CIS audit and chronological benchmark importer | `CONTRACTS.md`, `IEEE_CIS.md`, `src/identitygraph/ieee.py`; actual external CSV run unavailable |
| 2. Synthetic identity and authentication simulation | Variable household and fabricated-group sizes, device/network changes, legitimate sharing, delayed labels and two held-out takeover families | `data/v1/manifest.json`, `src/identitygraph/simulator.py` |
| 3. Cases and point-in-time features | Two targets, 32 allowlisted inputs, strict earlier-event history, equal-time exclusion, optional session telemetry | `src/identitygraph/features.py`, historical invariance tests |
| 4. Entity resolution | Exact/fuzzy attribute evidence, canonical attribute aliases, corroborated candidates requiring direct support for every pair | `canonical_attributes.json`, `clusters.json`, `GRAPH_DESIGN.md` |
| 5. Historical heterogeneous graph | Four observable node types and eight directed relation types in PyG; bounded investigator graph | `src/identitygraph/graph.py`, snapshot and unknown-node tests |
| 6. Representations and relationship prediction | Second-order Node2Vec/skip-gram, matched-row LightGBM comparison, temporal account-device link decoder | `artifacts/v1/research/` |
| 7. Models and evaluation | Rules, logistic regression, LightGBM, GraphSAGE, RGCN, GAT, ablations and Isolation Forest | `artifacts/v1/evaluation.json`, `models/`, `gnn*/`, `research/` |
| 8. Generalization and sensitivity | Chronological/cold/new-device/held-out-family tests, shifted population, reserved seeds 1301/1709, attribute-removal sensitivity | `artifacts/v1/REPORT.md`, `research/summary.json` |
| 9. API and investigator application | Stored scores, registration, replay/play/pause, graph exploration, candidates, case export and offline research views | `API.md`, `src/identitygraph/api.py`, `static/` |
| 10. Monitoring and delivery | Frozen-reference PSI, separate delayed-feedback ledger, reproducible package, test suite, replay verification, browser checks and Windows CI configuration | [BUILD_STATUS.md](BUILD_STATUS.md), `artifacts/verification/v1/` |

Paths without an initial folder in the evidence column refer to their surrounding docs
or the `artifacts/v1` run. Phase manifests record executed offline steps; the verification
folder records application and packaging checks separately.

## E1-E6 experiments

| Experiment | Implemented comparison | Interpretation limit |
|---|---|---|
| E1: graph information | Behavior/relationship LightGBM and GraphSAGE/RGCN/GAT | Daily GNN snapshots differ from event-time baseline history; shared multitask heads and sampling also differ |
| E2: device history | Full relationship model vs no-device LightGBM | Other signals can proxy device history; no guarantee removal reduces performance |
| E3: edge types | RGCN vs relation-ablated RGCN | Collapsing relations also reduces parameter count |
| E4: legitimate sharing | Shared-device activity and candidate false-positive counts | One simulator; relatives living separately can still qualify |
| E5: cold start | New accounts/devices and reserved populations | Account holdouts may share infrastructure; independent populations still use the same simulator |
| E6: entity resolution | Exact/fuzzy candidates, address removal, recomputed features with frozen models | These perturbations do not cover every possible matching error |

## Design choices and remaining boundaries

An account is the observable claimed identity. Hidden owners are evaluation truth, not
graph nodes. Login sessions are durable queries/events, not permanent GNN nodes. The
prototype uses pandas/Parquet for offline data and SQLite for the replay ledger; it does
not require a graph database or a message broker for this dataset size.

The API serves validation-selected conventional models. GNNs, Node2Vec and link prediction
are saved offline experiments; Isolation Forest contributes a separately labeled anomaly
percentile. Evidence lists inputs and relationships, not causal explanations or SHAP values.

The IEEE-CIS command is implemented and tested with fictional fixtures. Without the actual
source CSVs, its external evaluation remains unrun. It is a separate transaction-fraud
benchmark and cannot establish takeover or synthetic-identity accuracy.

The application is local, single-worker and unauthenticated, with no identity-provider
integration, automatic blocking or public deployment. Late events are rejected and in-memory
history is bounded by the small experiment rather than a production retention policy.
The recorded verification describes the local build. The public repository's workflow
reports [GitHub Actions results](https://github.com/Dilep01/identitygraph-ai/actions/workflows/checks.yml)
separately; a local passing test run does not establish a remote CI result.
