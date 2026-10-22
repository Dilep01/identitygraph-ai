> Archived v0.1 documentation. For the current build see [BUILD_STATUS.md](BUILD_STATUS.md).

# Local build status

Checked on 21 October 2026. The first local research version is ready to run from the original project workspace.

## Open the existing build

The dashboard is served at [http://127.0.0.1:8765](http://127.0.0.1:8765). If its server has stopped, run this from the project folder in PowerShell:

```powershell
.\scripts\start.ps1
```

The environment, data and trained models already exist. There is no need to run setup or rebuild to view this version. Use the setup steps in [README.md](../README.md) for a fresh checkout.

The dashboard was left with 100 of 2,011 held-out events replayed, producing 86 successful-session assessments and 15 review candidates. These counts change when you replay more events. Failed attempts contribute history but do not receive successful-session scores. Reset replay returns the local demo ledger to its initial state.

## Implemented

- Fictional identity and authentication data with separate observable records and hidden evaluation labels.
- Historical features, relationship matching and account/device/network/attribute graphs.
- Rules, Logistic Regression, behavioral LightGBM, relationship-feature LightGBM and a heterogeneous GraphSAGE comparison.
- Separate takeover and fabricated-identity scores; validation-based calibration, model selection and thresholds.
- A local API, persistent event and assessment ledger, ordered replay, duplicate handling and account registration.
- An investigation dashboard with account search, relationship exploration, saved evidence, candidate groups, evaluation filters and replay monitoring.
- Chronological, new-account, new-device, unseen-attack and shifted-simulator evaluations, with saved predictions and checksums.

The API serves relationship-feature LightGBM for both targets. GraphSAGE is an offline comparison. The current experiment is `release`, with 240 fictional accounts and 6,502 events.

## Verification

| Check | Saved result | Evidence |
|---|---|---|
| Automated suite, 20 October | 38 passed; zero failures, errors or skips; two dependency deprecation warnings | `artifacts/verification/tests.xml`, `tests.log` |
| Static checks | Passed | `artifacts/verification/lint.log` |
| Complete held-out replay | 2,011 events; all 1,835 successful-session feature vectors matched offline values exactly; maximum score difference 0.0 | `artifacts/verification/run_checks.json` |
| Persistence and duplicate handling | Verified across application restart | `artifacts/verification/run_checks.json` |
| Dataset and model output integrity | Seven data files and 20 experiment artifacts verified | `artifacts/verification/run_checks.json` |
| Browser, 21 October | Account search and selection, stored evidence, graph rendering, candidate navigation and evaluation filters checked; no captured console errors | `artifacts/verification/browser_checks.json` |
| Narrow layout | At the observed 615-pixel viewport, document width was 609 pixels after the overflow correction | `artifacts/verification/browser_checks.json`, `dashboard.png` |

Only `static/index.html` and `static/styles.css` changed after the release training run, to fix narrow layouts and refresh the cached stylesheet. Training and scoring sources match the saved release source hashes. The original experiment manifest remains unchanged; `artifacts/verification/final_manifest.json` records the delivered source and verification files separately.

The browser check covers the observed in-app browser size, not a full browser or device compatibility matrix. The Windows CI configuration exists, but GitHub Actions has not been run.

## Research limitations and remaining work

This is a synthetic-data prototype. It does not verify real identities, block accounts or establish performance on real fraud.

The frozen models missed all 27 held-out familiar-session attacks. Broad relationship candidates included 197 of 200 legitimate accounts and are unsuitable as fraud alerts. Known-account identity results also exploit regularities in the simulator's group sizes. These findings are reported in the dashboard and [RESULTS.md](RESULTS.md).

Further research should use newly reserved data with more varied household and fabricated-identity patterns, improve candidate specificity, and investigate observable signals for familiar-session attacks. IEEE-CIS, additional GNN architectures, link prediction, GNN evaluation on the separate shifted population and production integrations remain extensions; they are not part of this build.
