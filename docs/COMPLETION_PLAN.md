# Completion experiment contract

Written before the new held-out runs on 21 October 2026. Existing `release` artifacts remain unchanged.

Development uses seed 211. The final experiment uses seed 907, with frozen-model replication on seeds 1301 and 1709 and a shifted population. Development findings may change implementation; final test results may only change the report, not model selection or thresholds. A failed implementation check can require a corrected run with a new name, with the reason recorded.

## Deliverables

1. Vary legitimate household and fabricated-group sizes using overlapping distributions. Keep complete fabricated groups in either the known or cold-start cohort.
2. Keep familiar-session attacks as an explicit observational blind spot. Add optional, observable session-binding and challenge signals, plus a separate held-out session-replay scenario. These change the simulator and do not retrospectively fix the old benchmark.
3. Compare the original broad linkage candidates with corroborated candidates that require independent evidence and prevent transitive bridges. Preserve exact and approximate attribute evidence and evaluate legitimate shared-device cases.
4. Add Node2Vec representations, a temporal link-prediction experiment, anomaly scoring, relational GCN and heterogeneous attention comparisons. No requirement that a graph model wins.
5. Run device-history and relationship-type ablations and attribute-resolution sensitivity checks. Evaluate frozen models on reserved populations, including GNNs on the shifted population.
6. Add observable drift summaries, a separate delayed-label feedback ledger, case export and dashboard research results. Scoring must never read hidden simulator labels.
7. Provide a local IEEE-CIS audit and chronological transaction-fraud benchmark command. No source CSVs are available, so external benchmark results must remain explicitly unrun.
8. Verify temporal boundaries, unseen nodes, model persistence, API behavior, end-to-end score agreement, the browser and the packaged application. Publish a completion matrix with measured outcomes and remaining operating limits.

Completion means the documented local research functions work and their experiments are reproducible. It does not mean every simulated attack is detectable or that the system is suitable for production decisions. Scores and candidate relationships must retain their simulation and evidence boundaries.

## Development finding, before the final run

The seed-211, 120-account development run still included 44 of 99 legitimate accounts in corroborated candidates. Requiring different established networks did not resolve relatives living separately. The final candidate rule therefore requires an exact shared email, another contact type, a shared device and different established primary networks. The revised simulator includes shared email in both legitimate households and fabricated groups, with overlapping group sizes. This is a new simulation contract, not a retrospective repair of the original benchmark. Complete pair support is required within every displayed group.
