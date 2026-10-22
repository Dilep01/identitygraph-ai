# Published v1 evidence

These files are a compact snapshot of the completed local synthetic experiment,
not a benchmark rerun performed during publication.

| File | Contents |
|---|---|
| [REPORT.md](REPORT.md) | Readable comparison of all model families and diagnostic populations |
| [evaluation.json](evaluation.json) | Full metrics, calibration, thresholds and experiment metadata |
| [research_summary.json](research_summary.json) | Node2Vec, sampled link prediction, anomaly scoring and candidate sensitivity |
| [data_manifest.json](data_manifest.json) | Synthetic generator settings, split boundaries, file counts and hashes |
| [training_source_checksums.json](training_source_checksums.json) | Source hashes captured for the original training run |
| [verification.json](verification.json) | Recorded local tests, replay parity, artifact integrity and browser checks |
| [provenance.json](provenance.json) | Original evidence hashes and published snapshot hashes |

The [dashboard screenshot](../../images/dashboard.png) contains fictional accounts and
local replay counts. Model weights, data files, environments, caches and replay databases
are excluded from the repository. Paths mentioned inside the generated report refer to
artifacts created when you run the project locally. Machine-specific paths and hostnames
are omitted from the public verification summary.

The original training sources are unchanged for model fitting and historical features.
The verification summary records the three later application fixes: package-root
resolution and replay/empty-candidate handling. Publication adds documentation, these
evidence copies, ignore patterns and CI configuration; it does not retrain models or
choose thresholds using the test set.

All accuracy claims remain conditional on this simulator. The served model missed the
15 familiar-session and 14 session-replay attacks in the held-out scenarios. Candidate
precision varies across independent simulated populations. IEEE-CIS source CSVs were
not supplied, so no external benchmark result is included.

Follow the [repository setup instructions](../../../README.md) to generate a local run.
The [GitHub Actions workflow](https://github.com/Dilep01/identitygraph-ai/actions/workflows/checks.yml)
reports remote checks separately from this saved local evidence.
