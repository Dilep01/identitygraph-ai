# Optional IEEE-CIS transaction benchmark

The importer and benchmark are implemented. The actual IEEE-CIS CSVs were not supplied,
so no external benchmark has been run. IdentityGraph's generated data is sufficient to
run the main project. Fixture tests use fictional CSV rows only to verify this importer.

If authorized source files become available, place `train_transaction.csv` and
`train_identity.csv` together under `data/external/ieee/` inside this project. The command
does not download data, access a Kaggle account, or accept dataset terms for you.

Run from the project folder in PowerShell:

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -m identitygraph.cli ieee-benchmark --data-dir data/external/ieee --run ieee-audit --audit-only
.\.venv\Scripts\python.exe -m identitygraph.cli ieee-benchmark --data-dir data/external/ieee --run ieee-benchmark
```

Use different run names: an existing output folder is never overwritten. Missing files
produce an explicit error without creating a benchmark result.

The audit covers every CSV header, row count, missing-value fraction, duplicate join ID,
identity join coverage, identity orphan count and source SHA-256. The training command
loads a declared subset of transaction/device/email fields and validates a one-to-one
TransactionID join. Missing required fields, labels or timestamps fail explicitly.

Distinct TransactionDT values define chronological 60/20/20 train/validation/test blocks.
Numerical imputation and categorical frequency encoding fit on training only. A fixed
LightGBM baseline is compared with the same model plus prior card/device/email counts.
Counts use earlier timestamps only; equal-time transactions cannot affect each other.
Earlier observed test transactions may update counts without their labels.

Validation chooses a threshold under an empirical 2% false-positive-rate constraint.
Held-out predictions and models are saved with the audit and evaluation. Scores are
uncalibrated. This is a limited, reproducible baseline using selected columns, not a
full competition solution. Its target is transaction fraud; it cannot establish
synthetic-identity or account-takeover accuracy.
