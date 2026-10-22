# Final local research results

Run `v1`, seed 907, generator 1.0.0: 240 fictional accounts, 6,494 authentication events,
42 fabricated accounts and 36 simulated days. The earlier [v0.1 report](RESULTS_V0.md)
and its `release` artifacts are preserved. The generator, observability and group-size
rules changed between versions, so their metrics are not a controlled before/after comparison.

The final experiment was reserved in [COMPLETION_PLAN.md](COMPLETION_PLAN.md). Development
used seed 211. Final training used seed 907; frozen-model checks used seeds 1301, 1709 and
a shifted population. Models, calibration and thresholds were not retuned on final test results.

## Selected models and generalization

Relationship-feature LightGBM was selected for both targets using later validation.
A model with better test average precision was not substituted after seeing the test set.
The full 12-model comparison is in [the generated report](reports/v1/REPORT.md).

| Population | Takeover AP | Takeover recall | Identity AP | Identity recall |
|---|---:|---:|---:|---:|
| Future time, returning accounts | 0.7276 | 64.6% | 0.9342 | 85.3% |
| Cold accounts | 0.6099 | 45.5% | 0.6613 | 50.0% |
| Shifted simulator | 0.7524 | 64.4% | 0.5412 | 38.1% |
| Reserved seed 1301 | 0.6891 | 57.4% | 0.6680 | 42.9% |
| Reserved seed 1709 | 0.6756 | 58.0% | 0.6207 | 71.4% |

The main future-time takeover set contains 1,525 successful sessions, including 65
unauthorized sessions. At the frozen threshold, the selected model recalled 42 of 65,
with one false alert among 1,460 legitimate sessions (0.685 per 1,000). Identity metrics
use one latest assessment for each of 192 returning accounts, including 34 fabricated
identities. Four of 158 legitimate identities were flagged (2.53%). The validation 2%
false-positive constraint did not guarantee a 2% test false-positive rate.

Cold-account identity AP and independent-population identity results are substantially
lower than returning-account results. The simulator remains a shared source of structure
across every population. These are not measurements of real-world fraud detection.

## Held-out failures

The selected model missed all 15 familiar-session attacks and all 14 session-replay
attacks at its frozen threshold. The familiar scenario deliberately resembles legitimate
activity. The replay scenario contains noisy binding/challenge observations, but adding
those observations did not produce threshold-level generalization to this family.
Takeover AP was 0.0075 and 0.0902 respectively. The additional signals do not establish
that this failure is solved.

There was only one positive in the globally-new-device subset, and it was missed.
That subset is too small to support a broad new-device detection claim. Other diagnostic
subsets overlap; do not add their counts or treat them as independent replications.

## Relationship candidates

The following comparison evaluates both candidate policies on the same final dataset:

| Policy | True groups reached | Pair precision | Legitimate accounts included |
|---|---:|---:|---:|
| Broad legacy policy | 11 / 11 | 0.6% | 192 / 198 |
| Corroborated v1 policy | 8 / 11 | 90.6% | 6 / 198 |

The stricter policy trades coverage for specificity. It requires a shared email, another
attribute type, a shared device, different established primary networks, and qualifying
evidence for every pair within a displayed group. It still flags genuine relatives and
misses weakly connected fabricated groups. The displayed score of 80 is a fixed heuristic,
not a probability. Candidates can overlap.

Across the shifted and reserved populations, group coverage ranged from 41.7% to 83.3%,
pair precision from 60.8% to 88.5%, and legitimate accounts included from 6 to 22 out of 198.
The main-run 90.6% precision is therefore not a stable performance guarantee.

## Additional research

GraphSAGE, RGCN and GAT are evaluated on all populations with frozen weights. Node2Vec
uses a six-day historical graph, with a matched-warmup LightGBM comparison. These models
remain offline experiments. The no-device ablation scored better on the main takeover
ranking metric, so the results do not establish that device history always helps.
Collapsing RGCN edge types also changes parameter count; the comparison has that confound.

The temporal account-device link decoder achieved AP 0.9537 and recall 90.2% on 246
sampled pairs. Only 41 of 332 new test links had both endpoints in its known-node candidate
universe; 205 negatives were sampled. This is a limited link-ranking result, not general
link coverage, verified ownership, or fraud accuracy. Anomaly scores are validation
percentiles from Isolation Forest, not probabilities of compromise.

Address-removal sensitivity and exact/fuzzy comparisons are saved in
`artifacts/v1/research/summary.json` and `resolution_model_sensitivity.json`. Frozen baseline
models are evaluated on recomputed features without refitting. The E1-E6 coverage table
is in [COMPLETION.md](COMPLETION.md).

The IEEE-CIS importer/benchmark code is implemented and fixture-tested. Its real CSVs
were not supplied, so no external benchmark has been run or substituted with generated data.

## Evidence and reproduction

- [Full model and population report](reports/v1/REPORT.md)
- `artifacts/v1/evaluation.json`: thresholds, calibration, metrics and account-bootstrap intervals.
- `artifacts/v1/test_predictions.parquet`: saved predictions for recomputing metrics.
- `artifacts/v1/source_checksums.json`, `phase_manifest.json`, `checksums.json`: run provenance.
- [Application and package verification](BUILD_STATUS.md).

Run `./scripts/build_demo.ps1 -Run reproduction01 -Seed 907` from the project folder to
create a separate experiment. Existing run folders are never overwritten. Keep the recorded
source and locked dependencies. Account-bootstrap intervals do not remove dependence
between accounts sharing household infrastructure. Build wall time includes elapsed time
while the process was active and is not used as a throughput benchmark.
