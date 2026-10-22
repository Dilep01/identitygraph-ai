# IdentityGraph AI: v1

All results below use fictional observations and simulated labels. The earlier v0.1 benchmark remains unchanged.

## Main chronological comparison

| Model | Takeover AP | Takeover recall | False alerts / 1,000 | Identity AP |
|---|---:|---:|---:|---:|
| lightgbm_behavior | 0.7394 | 69.2% | 3.4247 | 0.1884 |
| lightgbm_graph | 0.7276 | 64.6% | 0.6849 | 0.9342 |
| lightgbm_no_assurance | 0.6694 | 64.6% | 0.6849 | 0.9342 |
| lightgbm_no_device | 0.7517 | 69.2% | 4.1096 | 0.9391 |
| logistic | 0.7209 | 72.3% | 39.7260 | 0.2493 |
| rules | 0.7169 | 66.2% | 3.4247 | 0.5639 |
| heterogeneous_graphsage | 0.6971 | 64.6% | 0.6849 | 0.6834 |
| heterogeneous_rgcn | 0.6747 | 61.5% | 0.0000 | 0.8044 |
| heterogeneous_gat | 0.6983 | 64.6% | 0.0000 | 0.8100 |
| heterogeneous_relation_ablated | 0.6734 | 64.6% | 0.0000 | 0.6335 |
| lightgbm_matched_warmup | 0.6835 | 61.5% | 1.3699 | 0.9342 |
| node2vec_lightgbm | 0.6937 | 64.6% | 0.0000 | 0.9962 |

Thresholds are frozen using later validation. These single-run differences are not a statistical superiority claim. The JSON report contains account-block bootstrap intervals.

## Selected model on diagnostic populations

| Population | Successful sessions | Takeover positives | AP | Recall | False alerts / 1,000 |
|---|---:|---:|---:|---:|---:|
| future_time | 1525 | 65 | 0.7276 | 64.6% | 0.6849 |
| cold_start | 414 | 11 | 0.6099 | 45.5% | 0.0000 |
| new_device | 290 | 20 | 0.9976 | 95.0% | 0.0000 |
| globally_new_device | 236 | 1 | 0.5000 | 0.0% | 0.0000 |
| unseen_attack | 1892 | 29 | 0.0586 | 0.0% | 0.5368 |
| familiar_session | 1878 | 15 | 0.0075 | 0.0% | 0.5368 |
| session_replay | 1877 | 14 | 0.0902 | 0.0% | 0.5368 |
| shared_device_activity | 326 | 0 | n/a | n/a | 3.0675 |
| simulator_shift | 1970 | 174 | 0.7524 | 64.4% | 4.4543 |
| reserved_seed_1301 | 1794 | 101 | 0.6891 | 57.4% | 0.0000 |
| reserved_seed_1709 | 1875 | 88 | 0.6756 | 58.0% | 0.0000 |

Selected takeover model: `lightgbm_graph`. Selected identity model: `lightgbm_graph`.

Familiar-session attacks intentionally lack distinguishing observable signals. The separate session-replay family has noisy session-binding and challenge observations. Improved results on that family are not a repair of the old familiar-session experiment.

## Candidate specificity

| Candidate experiment | Group coverage | Pair precision | Legitimate accounts included |
|---|---:|---:|---:|
| corroborated | 72.7% | 90.6% | 6 / 198 |
| broad_legacy | 100.0% | 0.6% | 192 / 198 |
| exact_only | 72.7% | 90.6% | 6 / 198 |
| simulator_shift | 44.4% | 80.6% | 12 / 198 |
| reserved_seed_1301 | 41.7% | 88.5% | 6 / 198 |
| reserved_seed_1709 | 83.3% | 60.8% | 22 / 198 |

The stricter candidates require shared email, another contact type, a device and different established primary networks. Every pair in a displayed group must qualify. Genuine relatives can still qualify, and sparse fabricated groups can be missed.

## Research coverage

- E1: behavior vs relationship LightGBM vs GraphSAGE/RGCN/GAT.
- E2: lightgbm_graph vs lightgbm_no_device.
- E3: heterogeneous_rgcn vs heterogeneous_relation_ablated.
- E4: shared_device_activity and candidate false positives.
- E5: cold_start and reserved populations.
- E6: exact/fuzzy and missing-address candidate sensitivity.

Future account-device link decoder: complete; AP 0.9537. Its candidate set uses known nodes and sampled window-specific negatives. It predicts a future observed link, not fraud.

Node2Vec is frozen after a six-day warmup; unseen entities use zero vectors with missing-node flags. Its matched-warmup LightGBM comparison uses identical training rows. GraphSAGE, RGCN and GAT use daily graph snapshots. The relation-ablated RGCN collapses edge types and has fewer relation parameters, so that experiment also changes capacity.

Isolation Forest produces a validation percentile, not a fraud probability. The address-removal experiment recomputes historical features and evaluates frozen baseline models; detailed results are in `research/resolution_model_sensitivity.json`.

The IEEE-CIS importer and transaction benchmark are implemented, but source CSVs were not supplied. No external benchmark was run.

## Reproduction

Run `identitygraph build --run <new-name> --seed 907 --accounts 240`. Keep the recorded source and locked environment for exact reproduction. Existing run folders are never overwritten.

See `source_checksums.json`, `phase_manifest.json`, `checksums.json`, `test_predictions.parquet`, and the dataset manifests. Serving and browser checks are recorded separately under `artifacts/verification/<run>/`.
