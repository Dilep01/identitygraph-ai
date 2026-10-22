"""Reproducible phase runner. Only this training/evaluation layer joins truth."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import cluster_metrics, metrics
from .features import BEHAVIOR_FEATURES, GRAPH_FEATURES, build_feature_frame
from .simulator import SimulationConfig, digest, generate, load_observed


def write_json(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def labeled_features(data_dir: Path) -> tuple[dict, pd.DataFrame, dict]:
    observed = load_observed(data_dir)
    frame = build_feature_frame(**observed)
    event_truth = pd.read_parquet(data_dir / "truth/events.parquet").rename(columns={
        "scenario": "event_scenario", "attack_group": "event_attack_group",
        "label_available_at": "event_label_available_at"})
    account_truth = pd.read_parquet(data_dir / "truth/accounts.parquet").rename(columns={
        "scenario": "identity_scenario", "attack_group": "identity_attack_group",
        "label_available_at": "identity_label_available_at"})
    frame = frame.merge(event_truth, on="event_id", validate="one_to_one").merge(
        account_truth, on="account_id", validate="many_to_one")
    # Normalize time strings across +00:00/Z implementations before comparisons.
    for column in ["event_time", "event_label_available_at", "identity_label_available_at"]:
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Evaluation-only slice marker derived from observable events, never a model feature.
    first_device = observed["events"].groupby("device_id").event_time.min()
    device_events = observed["events"][["event_id", "device_id", "event_time"]].copy()
    device_events["globally_new_device"] = device_events.event_time == device_events.device_id.map(first_device)
    frame = frame.merge(device_events[["event_id", "globally_new_device"]], on="event_id", validate="one_to_one")
    frame = frame.sort_values(["event_time", "event_id"]).reset_index(drop=True)
    names = BEHAVIOR_FEATURES + GRAPH_FEATURES
    if not np.isfinite(frame[names].to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains nonfinite values")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    return observed, frame, manifest


def split_frames(frame: pd.DataFrame, manifest: dict):
    cut = manifest["split_cutoffs"]
    eligible_train = ((frame.event_time < cut["train_cutoff"]) &
                      (frame.event_label_available_at < cut["train_cutoff"]) &
                      (frame.identity_label_available_at < cut["train_cutoff"]))
    eligible_val = ((frame.event_time >= cut["train_cutoff"]) &
                    (frame.event_time < cut["validation_cutoff"]) &
                    (frame.event_label_available_at < cut["validation_cutoff"]) &
                    (frame.identity_label_available_at < cut["validation_cutoff"]))
    test = frame[frame.event_time >= cut["validation_cutoff"]].copy()
    # Cold-start groups never contribute supervised training labels.
    splits = {
        "future_time": test[test.cohort != "cold_start"].copy(),
        "cold_start": test[test.cohort == "cold_start"].copy(),
        "new_device": test[test.new_device == 1].copy(),
        "globally_new_device": test[test.globally_new_device].copy(),
        "unseen_attack": test[(test.is_takeover == 0) | test.event_scenario.isin(["familiar_session", "session_replay"])].copy(),
        "familiar_session": test[(test.is_takeover == 0) | (test.event_scenario == "familiar_session")].copy(),
        "session_replay": test[(test.is_takeover == 0) | (test.event_scenario == "session_replay")].copy(),
        "shared_device_activity": test[test.event_scenario.isin(["household_sharing", "identity_shared_device"])].copy(),
    }
    return frame[eligible_train].copy(), frame[eligible_val].copy(), splits


def build(root: Path, run: str = "demo", seed: int = 41, accounts: int = 240,
          gnn_epochs: int = 12, with_gnn: bool = True, extended: bool = True) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,50}", run):
        raise ValueError("Run name must contain only lowercase letters, numbers, hyphens and underscores")
    root = Path(root).resolve()
    data_dir, output = root / "data" / run, root / "artifacts" / run
    if data_dir.exists() or output.exists():
        raise FileExistsError(f"Run {run!r} already exists. Choose a new run name to preserve evidence.")
    output.mkdir(parents=True)
    source_files = [p for base in (root / "src", root / "scripts", root / "tests")
                    for p in base.rglob("*") if p.is_file() and p.suffix in (".py", ".ps1", ".js", ".html", ".css")]
    source_files += [root / "pyproject.toml", root / "requirements-lock.txt", root / "docs/COMPLETION_PLAN.md"]
    write_json(output / "source_checksums.json", {str(p.relative_to(root)).replace("\\", "/"): digest(p)
                                                 for p in sorted(source_files) if p.exists()})
    began = time.perf_counter()
    journal = {"run": run, "phases": [], "status": "running"}

    def phase(name, details):
        journal["phases"].append({"name": name, "status": "complete", "details": details})
        write_json(output / "phase_manifest.json", journal)
        print(f"Completed {name}: {details}", flush=True)

    config = SimulationConfig(seed=seed, accounts=accounts)
    manifest = generate(data_dir, config)
    phase("1_contracts_and_2_simulator", {"manifest": str(data_dir / "manifest.json"),
                                           "accounts": accounts, "seed": seed})
    observed, frame, manifest = labeled_features(data_dir)
    train, validation, tests = split_frames(frame, manifest)
    (output / "features").mkdir()
    frame.to_parquet(output / "features/labeled_features.parquet", index=False)
    for name, split in {"train": train, "validation": validation, **tests}.items():
        split[["event_id", "account_id", "event_time"]].to_parquet(output / f"features/{name}_ids.parquet", index=False)
    phase("3_historical_features", {"training_sessions": len(train), "validation_sessions": len(validation),
                                     "heldout_sessions": {k: len(v) for k, v in tests.items()},
                                     "label_delay_applied": True})
    from .models import predict_baselines, train_baselines
    model_dir = output / "models"
    baseline_summary = train_baselines(train, validation, model_dir)
    phase("4_baseline_and_relationship_models", {"summary": "models/baseline_comparison.json"})

    shifted_dir = root / "data" / (run + "_shift")
    shifted_config = replace(config, seed=seed + 1000, shifted=True, churn_rate=0.24,
                             missing_rate=0.16, attribute_corruption=0.30, takeover_rate=0.08)
    generate(shifted_dir, shifted_config)
    shift_observed, shift_frame, shift_manifest = labeled_features(shifted_dir)
    tests["simulator_shift"] = shift_frame[shift_frame.event_time >= shift_manifest["replay_cutoff"]].copy()
    populations = {"simulator_shift": shift_observed}
    if extended:
        for replication_seed in (seed + 394, seed + 802):
            name = f"reserved_seed_{replication_seed}"
            directory = root / "data" / f"{run}_{name}"
            generate(directory, replace(config, seed=replication_seed))
            other_observed, other_frame, other_manifest = labeled_features(directory)
            tests[name] = other_frame[other_frame.event_time >= other_manifest["replay_cutoff"]].copy()
            populations[name] = other_observed
    all_predictions = []
    report = {
        "provenance": "Fully synthetic; no external or production validation",
        "run": run, "seed": seed,
        "datasets": {"main": manifest, "shift": shift_manifest},
        "splits": {}, "cluster_metrics": {}, "gnn": {"status": "not_run"},
        "limitations": [
            "Main simulator, independent reserved populations and a shifted seed; generator-based evidence only.",
            "Validation calibration and threshold blocks contain returning accounts, so cold-start calibration may differ.",
            "Test subsets overlap; they are separate diagnostic views, not independent experiments.",
            "New-device test measures account-unfamiliar devices; globally_new_device separately tests first-ever observed devices.",
            "Unseen-attack test includes familiar-session and session-replay attacks; separate slices retain their distinct observability.",
            "Temporal tests allow earlier observed test activity to become history, never its hidden labels.",
            "Identity metrics use the latest successful-session assessment per account in each subset.",
            "Account bootstrap intervals do not remove dependence across shared devices or households.",
            "Cluster heuristics are candidate retrieval, not calibrated fraud probabilities.",
        ],
    }
    for name, split in tests.items():
        predictions = predict_baselines(model_dir, split, split=name)
        all_predictions.append(predictions)
        report["splits"][name] = {}
        for (target, model), group in predictions.groupby(["target", "model"]):
            report["splits"][name].setdefault(target, {})[model] = metrics(group)
    from .graph import broad_candidate_clusters, candidate_clusters, canonical_attribute_mapping
    cutoff = manifest["split_cutoffs"]["test_end"]
    clusters = candidate_clusters(**observed, cutoff=cutoff)
    truth_groups = pd.read_parquet(data_dir / "truth/groups.parquet")
    eligible = set(observed["accounts"].loc[observed["accounts"].created_at < cutoff, "account_id"])
    report["cluster_metrics"] = cluster_metrics(clusters, truth_groups, eligible)
    report["cluster_comparison"] = {"corroborated": report["cluster_metrics"],
        "broad_legacy": cluster_metrics(broad_candidate_clusters(**observed, cutoff=cutoff), truth_groups, eligible),
        "exact_only": cluster_metrics(candidate_clusters(**observed, cutoff=cutoff, fuzzy=False), truth_groups, eligible)}
    write_json(output / "clusters.json", clusters)
    write_json(output / "canonical_attributes.json", canonical_attribute_mapping(observed["attributes"]))
    phase("5_cluster_evaluation", report["cluster_metrics"])
    if with_gnn:
        from .gnn import train_gnn
        gnn_tests = {k: v for k, v in tests.items() if k not in populations}
        extra = {name: (observed_values, tests[name]) for name, observed_values in populations.items()}
        report["graph_models"] = {}
        for architecture in (("graphsage", "rgcn", "gat", "relation_ablated") if extended else ("graphsage",)):
            folder = output / ("gnn" if architecture == "graphsage" else f"gnn_{architecture}")
            gnn_summary = train_gnn(**observed, train_frame=train, validation_frame=validation,
                                    test_frames=gnn_tests, output_dir=folder, epochs=gnn_epochs,
                                    architecture=architecture, extra_populations=extra)
            model_name = "heterogeneous_" + architecture
            report["graph_models"][model_name] = gnn_summary
            if architecture == "graphsage":
                report["gnn"] = gnn_summary
            gnn_wide = pd.read_csv(folder / "gnn_predictions.csv")
            for split_name in tests:
                subset = gnn_wide[gnn_wide.split == split_name]
                for target, label in (("takeover", "is_takeover"), ("identity", "is_synthetic")):
                    long = subset[["event_id", "account_id", "event_time"]].copy()
                    long["target"], long["model"], long["split"] = target, model_name, split_name
                    long["label"], long["score"] = subset[label], subset[f"{target}_score"]
                    long["threshold"] = subset[f"{target}_threshold"]
                    report["splits"][split_name].setdefault(target, {})[model_name] = metrics(long)
                    all_predictions.append(long)
            phase("6_" + architecture, {"epochs": gnn_epochs, "summary": folder.name})
    if extended:
        from .research import (anomaly_scores, link_experiment, train_anomaly,
                               train_representation_experiments)
        from sklearn.metrics import average_precision_score
        research_dir = output / "research"
        warmup = (pd.Timestamp(config.start) + pd.Timedelta(days=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
        representation_predictions, representation_summary = train_representation_experiments(
            observed, train, validation, tests, populations, research_dir, warmup)
        all_predictions.append(representation_predictions)
        for (split, target, model), group in representation_predictions.groupby(["split", "target", "model"]):
            report["splits"][split].setdefault(target, {})[model] = metrics(group)
        anomaly = train_anomaly(train, validation, research_dir / "anomaly.joblib")
        anomaly_report, anomaly_tables = {}, []
        for name, split in tests.items():
            if split.empty:
                continue
            scores = anomaly_scores(anomaly, split)
            anomaly_report[name] = {"n": len(split), "above_95th_percentile": int((scores >= .95).sum()),
                "takeover_average_precision": float(average_precision_score(split.is_takeover, scores)) if split.is_takeover.any() else None,
                "score_type": "validation_percentile_not_fraud_probability"}
            anomaly_tables.append(split[["event_id", "account_id"]].assign(split=name, anomaly_percentile=scores))
        pd.concat(anomaly_tables, ignore_index=True).to_parquet(research_dir / "anomaly_predictions.parquet", index=False)
        def dt(day):
            return (pd.Timestamp(config.start) + pd.Timedelta(days=day)).strftime("%Y-%m-%dT%H:%M:%SZ")
        link_report = link_experiment(observed, {"train": (dt(7), dt(config.train_days)),
            "validation": (dt(config.train_days), dt(config.train_days + config.validation_days)),
            "test": (dt(config.train_days + config.validation_days), dt(config.days))}, research_dir)
        cluster_populations = {}
        for name, values in populations.items():
            directory = shifted_dir if name == "simulator_shift" else root / "data" / f"{run}_{name}"
            truth = pd.read_parquet(directory / "truth/groups.parquet")
            predicted = candidate_clusters(**values, cutoff=cutoff)
            cluster_populations[name] = cluster_metrics(predicted, truth, set(values["accounts"].account_id))
        degraded = {**observed, "attributes": observed["attributes"].copy()}
        address_rows = degraded["attributes"].attribute_type == "address"
        degraded["attributes"].loc[address_rows, "value"] = ""
        resolution = {"as_observed": report["cluster_metrics"], "exact_only": report["cluster_comparison"]["exact_only"],
            "addresses_missing": cluster_metrics(candidate_clusters(**degraded, cutoff=cutoff), truth_groups, eligible)}
        degraded_features = build_feature_frame(**degraded).set_index("event_id")
        feature_columns = BEHAVIOR_FEATURES + GRAPH_FEATURES
        resolution_models = {}
        for name in ("future_time", "cold_start"):
            frame_with_labels = tests[name].copy()
            frame_with_labels[feature_columns] = degraded_features.loc[frame_with_labels.event_id, feature_columns].to_numpy()
            diagnostic = predict_baselines(model_dir, frame_with_labels, split=name)
            resolution_models[name] = {}
            for (target, model), group in diagnostic.groupby(["target", "model"]):
                resolution_models[name].setdefault(target, {})[model] = metrics(group)
        write_json(research_dir / "resolution_model_sensitivity.json", resolution_models)
        research_summary = {"representations": representation_summary, "link_prediction": link_report,
            "anomaly": anomaly_report, "cluster_replications": cluster_populations, "resolution_sensitivity": resolution,
            "resolution_model_sensitivity": "research/resolution_model_sensitivity.json",
            "external_benchmark": {"status": "source_data_not_supplied", "command": "identitygraph ieee-benchmark --data-dir <CSV folder>"},
            "experiments": {"E1": "behavior vs relationship LightGBM vs GraphSAGE/RGCN/GAT",
                "E2": "lightgbm_graph vs lightgbm_no_device", "E3": "heterogeneous_rgcn vs heterogeneous_relation_ablated",
                "E4": "shared_device_activity and candidate false positives", "E5": "cold_start and reserved populations",
                "E6": "exact/fuzzy and missing-address candidate sensitivity"}}
        write_json(research_dir / "summary.json", research_summary)
        monitoring_names = ["account_device_count", "device_other_account_count", "login_count_24h", "new_device",
                            "missing_attribute_type_count", "binding_mismatch"]
        from .models import load_bundle, selected_bundle_predictions
        reference = {name: validation[name].tolist() for name in monitoring_names}
        reference_scores = selected_bundle_predictions(load_bundle(model_dir), validation)
        reference.update({target + "_score": reference_scores[target + "_score"].tolist() for target in ("takeover", "identity")})
        write_json(research_dir / "monitoring_reference.json", reference)
        report["research"] = research_summary
        phase("7_representation_link_anomaly_and_ablation_experiments", {"summary": "research/summary.json"})
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(output / "test_predictions.parquet", index=False)
    report["baseline_training"] = baseline_summary
    report["runtime_seconds"] = round(time.perf_counter() - began, 3)
    report["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "packages": {p: importlib.metadata.version(p) for p in
                                          ["numpy", "pandas", "scikit-learn", "lightgbm", "torch", "torch-geometric", "fastapi"]}}
    write_json(output / "evaluation.json", report)
    from .reporting import write_report
    write_report(report, output)
    phase("7_heldout_evaluation", {"report": "evaluation.json", "runtime_seconds": report["runtime_seconds"]})
    journal["status"] = "complete"
    journal["scope"] = "Offline research pipeline; API, browser and automated checks recorded separately"
    write_json(output / "phase_manifest.json", journal)
    write_json(output / "checksums.json", {str(p.relative_to(output)).replace("\\", "/"): digest(p)
                                           for p in output.rglob("*") if p.is_file() and p.name != "checksums.json"})
    write_json(root / "artifacts/current.json", {"data_dir": f"data/{run}", "artifact_dir": f"artifacts/{run}"})
    return {"run": run, "report": str(output / "evaluation.json"), "runtime_seconds": report["runtime_seconds"]}
