# Local build status: version 1.0

The local research project is complete. Verification was performed on 21-22 October 2026
in the original project workspace. The current experiment is `v1`: seed 907,
240 fictional accounts and 6,494 events. The [completion matrix](COMPLETION.md) maps the
original scope to the delivered code, experiments and evidence.

## Open the project

The dashboard is at [http://127.0.0.1:8765](http://127.0.0.1:8765). If its local server
has stopped, run this from the project folder in PowerShell:

```powershell
.\scripts\start.ps1
```

The original development workspace already contains the environment, data and trained models.
A fresh clone must run the setup and build commands in [README.md](../README.md).
The recorded browser check left 106 of
2,133 held-out events replayed, with 96 successful-session assessments and 16 review
candidates. Playback is paused. These counts change when more events are replayed.
Failed attempts supply context but do not receive successful-session scores.

## Verification results

| Check | Verified result | Evidence under `artifacts/verification/v1/` |
|---|---|---|
| Source test suite, 21 October | 54 passed; zero failures, errors or skips | `tests.xml`, `tests.log` |
| Installed wheel suite, 21 October | Same 54 tests passed from the installed 1.0.0 wheel | `wheel_tests.xml`, `wheel_tests.log` |
| Package behavior | Imports from the wheel; real model readiness; packaged dashboard files; current-folder and explicit CLI project root | `package_checks.json`, `package_checks.log` |
| Static checks | Python lint and JavaScript syntax passed | `lint.log`, `final_manifest.json` |
| Complete held-out replay | All 2,133 events processed; exact feature agreement for 1,939 successful sessions; maximum score difference 0.0 | `run_checks.json`, `replay_checks.log` |
| Persistence and duplicates | Stored scores survive restart; identical duplicates return the saved assessment | `run_checks.json` |
| Artifact integrity | Seven primary data files and 44 experiment artifacts verified by replay check; final audit also checks reserved/shifted data | `run_checks.json`, `final_manifest.json` |
| Browser | Account search, account selection, candidate navigation, evidence/anomaly display, replay/play/pause, evaluation filters and research/monitoring views | `browser_checks.json`, `dashboard.png` |
| Browser errors/layout | No captured console errors; 609-pixel document within the observed 615-pixel viewport | `browser_checks.json` |

The suite includes historical/label invariance, unknown graph nodes, four graph encoders,
model persistence, candidate specificity, delayed-feedback availability and immutability,
case export, and the IEEE-CIS importer with explicitly fictional fixtures. Tests emit
upstream Starlette/httpx and Torch deprecation warnings. They are not test failures.
The browser check covers the observed in-app browser size, not all devices or browsers.

## Delivery and provenance

The wheel is `artifacts/package/identitygraph_ai-1.0.0-py3-none-any.whl`. The launcher keeps
serving on the loopback interface with one worker. The final trained model sources remain
unchanged. Three source files changed after training for delivery: `api.py` and `cli.py`
now resolve an installed application's project root correctly, and `static/app.js` stops
playback on errors/reset and handles empty candidates. These changes do not modify
features, fitted weights, calibration, thresholds or model selection.

The original experiment hashes remain intact. The local `final_manifest.json` records the delivered
source, documentation, wheel, verification evidence and the post-training changes. The
old `release` artifacts and [v0.1 status](BUILD_STATUS_V0.md) are preserved separately.
The public repository includes [results and selected verification evidence](reports/v1/README.md).
Model binaries, datasets and full local logs are excluded. Publication documentation changes
do not change the frozen experiment; the evidence snapshot records its source hashes.
Current remote CI results are reported by [GitHub Actions](https://github.com/Dilep01/identitygraph-ai/actions/workflows/checks.yml).

## What completion does not establish

The served takeover model missed all 15 familiar-session and all 14 session-replay attacks
in the held-out scenarios. Stricter candidates reached 8 of 11 true groups and included
6 of 198 legitimate accounts; specificity and coverage varied across reserved populations.
Read [RESULTS.md](RESULTS.md) for denominators, independent-population results and other limits.

The actual IEEE-CIS CSV files were not supplied. Its importer and benchmark code are
implemented, but that external evaluation remains unrun. No real identity verification,
authentication-provider integration, automated account blocking or production deployment
is claimed. This is a completed local research implementation with documented detection gaps.
