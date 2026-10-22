"""Human-readable reports computed from saved experiment results."""
from pathlib import Path


def number(value, percent=False):
    return "n/a" if value is None else f"{100 * value:.1f}%" if percent else f"{value:.4f}"


def write_report(report: dict, output: Path):
    lines = [f"# IdentityGraph AI: {report['run']}", "",
             "All results below use fictional observations and simulated labels. The earlier v0.1 benchmark remains unchanged.", "",
             "## Main chronological comparison", "",
             "| Model | Takeover AP | Takeover recall | False alerts / 1,000 | Identity AP |",
             "|---|---:|---:|---:|---:|"]
    future = report["splits"]["future_time"]
    for model, takeover in future["takeover"].items():
        identity = future["identity"].get(model, {})
        lines.append(f"| {model} | {number(takeover.get('average_precision'))} | {number(takeover.get('recall'), True)} | "
                     f"{number(takeover.get('false_alerts_per_1000'))} | {number(identity.get('average_precision'))} |")
    lines += ["", "Thresholds are frozen using later validation. These single-run differences are not a statistical superiority claim. "
              "The JSON report contains account-block bootstrap intervals.", "", "## Selected model on diagnostic populations", "",
              "| Population | Successful sessions | Takeover positives | AP | Recall | False alerts / 1,000 |",
              "|---|---:|---:|---:|---:|---:|"]
    selected = report["baseline_training"]["targets"]["takeover"]["selected_model"]
    for name, split in report["splits"].items():
        row = split["takeover"][selected]
        lines.append(f"| {name} | {row.get('n', 0)} | {row.get('positives', 0)} | {number(row.get('average_precision'))} | "
                     f"{number(row.get('recall'), True)} | {number(row.get('false_alerts_per_1000'))} |")
    lines += ["", f"Selected takeover model: `{selected}`. Selected identity model: "
              f"`{report['baseline_training']['targets']['identity']['selected_model']}`.", "",
              "Familiar-session attacks intentionally lack distinguishing observable signals. The separate session-replay family has noisy "
              "session-binding and challenge observations. Improved results on that family are not a repair of the old familiar-session experiment.", "",
              "## Candidate specificity", "", "| Candidate experiment | Group coverage | Pair precision | Legitimate accounts included |",
              "|---|---:|---:|---:|"]
    research = report.get("research", {})
    for name, row in {**report["cluster_comparison"], **research.get("cluster_replications", {})}.items():
        lines.append(f"| {name} | {number(row['candidate_group_coverage'], True)} | {number(row['pair_precision'], True)} | "
                     f"{row['benign_accounts_flagged']} / {row['eligible_benign_accounts']} |")
    lines += ["", "The stricter candidates require shared email, another contact type, a device and different established primary networks. "
              "Every pair in a displayed group must qualify. Genuine relatives can still qualify, and sparse fabricated groups can be missed.", "",
              "## Research coverage", ""]
    for name, description in research.get("experiments", {}).items():
        lines.append(f"- {name}: {description}.")
    link = research.get("link_prediction", {})
    lines += ["", f"Future account-device link decoder: {link.get('status', 'not run')}; AP {number(link.get('average_precision'))}. "
              "Its candidate set uses known nodes and sampled window-specific negatives. It predicts a future observed link, not fraud.", "",
              "Node2Vec is frozen after a six-day warmup; unseen entities use zero vectors with missing-node flags. Its matched-warmup "
              "LightGBM comparison uses identical training rows. GraphSAGE, RGCN and GAT use daily graph snapshots. "
              "The relation-ablated RGCN collapses edge types and has fewer relation parameters, so that experiment also changes capacity.", "",
              "Isolation Forest produces a validation percentile, not a fraud probability. The address-removal experiment recomputes "
              "historical features and evaluates frozen baseline models; detailed results are in `research/resolution_model_sensitivity.json`.", "",
              "The IEEE-CIS importer and transaction benchmark are implemented, but source CSVs were not supplied. No external benchmark was run.", "",
              "## Reproduction", "",
              f"Run `identitygraph build --run <new-name> --seed {report['seed']} --accounts {report['datasets']['main']['config']['accounts']}`. "
              "Keep the recorded source and locked environment for exact reproduction. Existing run folders are never overwritten.", "",
              "See `source_checksums.json`, `phase_manifest.json`, `checksums.json`, `test_predictions.parquet`, and the dataset manifests. "
              "Serving and browser checks are recorded separately under `artifacts/verification/<run>/`.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
