> Archived v0.1 documentation. For the current build see [RESULTS.md](RESULTS.md).

# Initial local benchmark

Current run: `release`, seed 101, generator 0.1.1. The dataset contains 240 fictional accounts and 6,502 authentication events across 36 simulated days. Forty accounts belong to ten generated fabricated-identity groups. All conclusions below describe this simulator.

The `demo` and `verified` folders retain development runs. During review, we corrected hash-order-dependent generation, chronological device-state updates, and a shared-device scenario label. Use `artifacts/release/evaluation.json` for the current results. Model families, hyperparameters and validation selection rules were not tuned to the final test results.

## Takeover comparison

The future-time test contains 1,477 successful sessions on returning accounts, including 64 unauthorized sessions. Average precision summarizes ranking; recall uses the threshold frozen on validation.

| Model | Average precision | Recall | False alerts / 1,000 legitimate sessions |
|---|---:|---:|---:|
| Behavioral LightGBM | 0.6398 | 60.94% | 4.25 |
| LightGBM with relationship features | 0.6907 | 67.19% | 0.00 |
| Heterogeneous GraphSAGE | 0.6816 | 56.25% | 0.71 |

Relationship features improved average precision in this run. The GNN did not beat relationship-feature LightGBM on this test. These single-run differences do not establish statistical superiority. The JSON report includes account-block bootstrap intervals and describes residual dependence through shared infrastructure.

The separately held-out familiar-session attack family had **zero recall for every model** at its frozen threshold: all 27 unauthorized sessions in that diagnostic slice were missed. Its observable behavior deliberately resembles legitimate activity. This is a measured detection limitation, not evidence of general takeover detection.

## Identity assessment

For returning accounts, relationship-feature LightGBM reached AP 1.000 on 192 latest-account assessments (32 fabricated identities). That result is highly specific to the simulator and known-account setting. On 48 new accounts, including eight fabricated identities, its AP fell to 0.6875 and recall to 62.5%. GraphSAGE had higher new-account identity AP, 0.8052, but recall of 50% at its validation threshold.

The simulator creates ordinary households in groups of three and most fabricated-identity groups in groups of four. Models can exploit that structural difference. Additional household sizes and independent identity-generation mechanisms are needed before treating the identity result as robust. Account identity itself is not a feature, but known-account relationships recur across chronological windows.

Both served targets were selected as relationship-feature LightGBM using later-validation performance. GNN results remain an offline comparison. Its daily graph snapshots and identity training sample weights differ from the baseline setup; see MODEL_DESIGN.md.

## Candidate clusters

The broad relationship candidate pipeline reached nine of ten true fabricated-identity groups. It also included **197 of 200 legitimate accounts**, and pairwise precision was only **0.67%**. One sparse-identity group was wholly missed.

This candidate pipeline is unsuitable as a fraud alert rule. Shared household attributes and reused infrastructure connect unrelated accounts, and connected components amplify those links. The UI therefore labels the output relationship candidates and keeps heuristic linkage strength separate from model risk. This version demonstrates the failure and preserves supporting edges; it does not claim successful suspicious-cluster discrimination.

## Shifted data and interpretation

The independently generated shifted dataset changes seed, device/network churn, missing attributes, attribute corruption and attack prevalence. Frozen relationship-feature LightGBM achieved takeover AP 0.6582 with 58.49% recall and identity AP 0.7500 with 70% recall. This remains a test of another configuration of the same simulator. The GNN was not evaluated on the separate shifted population.

The next research improvements should be evaluated on new reserved data: vary household and fraud-group sizes, distinguish sharing from evidence of fabrication, introduce additional observable signals for familiar-session attacks, and measure candidate specificity without choosing rules from test labels.

Full evidence is in `artifacts/release/evaluation.json`, `test_predictions.parquet`, model metadata, GNN metadata and dataset manifests. Automated verification records are under `artifacts/verification/`.
