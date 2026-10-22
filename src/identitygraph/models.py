"""Task-specific baselines with chronological validation and simulation-only calibration."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import BEHAVIOR_FEATURES, FEATURE_VERSION, GRAPH_FEATURES

TARGETS = {"takeover": "is_takeover", "identity": "is_synthetic"}
MODEL_VERSION = "identitygraph-baselines-v2"
FPR_LIMIT = 0.02


def _numeric(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Only an explicit observable allowlist may enter a model."""
    missing = set(names).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    values = frame.loc[:, names].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Model features must be finite numeric values")
    return values


def _positive_probability(estimator, values: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(values)
    positive = np.flatnonzero(np.asarray(estimator.classes_) == 1)
    return probabilities[:, positive[0]] if len(positive) else np.zeros(len(values))


class RuleEstimator:
    """Transparent fixed heuristics, deliberately not fitted to generator labels."""

    classes_ = np.array([0, 1])

    def __init__(self, names: list[str], target: str):
        self.names = names
        self.target = target

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        columns = {name: values[:, i] for i, name in enumerate(self.names)}

        def feature(*options: str) -> np.ndarray:
            for name in options:
                if name in columns:
                    return np.maximum(columns[name], 0)
            return np.zeros(len(values))

        new_device = feature("is_new_device", "new_device", "device_is_new")
        failures = feature("failure_count_24h", "account_failures_24h", "failures_24h", "recent_failures")
        sharing = feature("device_other_account_count", "device_account_count", "device_accounts")
        attributes = feature("shared_attribute_account_count", "max_attribute_account_count", "attribute_max_accounts")
        new_region = feature("is_new_region", "new_region", "region_is_new")
        if self.target == "takeover":
            logit = -3.0 + 1.1 * new_device + 0.8 * new_region
            logit += 0.4 * np.minimum(failures, 6) + 0.25 * np.minimum(sharing, 8)
            logit += 1.8 * feature("binding_mismatch") + 0.4 * feature("challenge_failures")
        else:
            logit = -3.0 + 0.45 * np.minimum(attributes, 10)
            logit += 0.35 * np.minimum(sharing, 10)
            logit += 0.5 * feature("shared_contact_different_home_count")
        score = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        return np.column_stack([1 - score, score])


class CalibratedEstimator:
    """Platt scaling on an earlier validation block, never the threshold block."""

    def __init__(self, estimator, calibrator=None):
        self.estimator = estimator
        self.calibrator = calibrator

    def predict_score(self, values: np.ndarray) -> np.ndarray:
        raw = _positive_probability(self.estimator, values)
        if self.calibrator is None:
            return np.clip(raw, 0, 1)
        logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
        return _positive_probability(self.calibrator, logits.reshape(-1, 1))


def _latest_account(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["event_time", "event_id"]).drop_duplicates("account_id", keep="last")


def split_validation(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on unique timestamps so simultaneous events never cross the boundary."""
    times = pd.to_datetime(validation["event_time"], utc=True)
    distinct = np.sort(times.unique())
    if len(distinct) < 2:
        return validation.iloc[0:0].copy(), validation.copy()
    boundary = distinct[len(distinct) // 2]
    return validation.loc[times < boundary].copy(), validation.loc[times >= boundary].copy()


def choose_threshold(labels: np.ndarray, scores: np.ndarray, fpr_limit: float = FPR_LIMIT) -> dict:
    """Maximize recall within a validation FPR budget; never inspect test labels."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not len(labels):
        return {"threshold": 1.0000001, "recall": None, "fpr": None, "precision": None,
                "selection_status": "no_validation_examples"}
    if not np.any(labels == 0):
        return {"threshold": 1.0000001, "recall": 0.0, "fpr": None, "precision": None,
                "selection_status": "no_benign_validation_examples"}
    candidates = np.r_[1.0000001, np.unique(scores)]
    best = None
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fpr = fp / max(1, int(np.sum(labels == 0)))
        recall = tp / max(1, int(np.sum(labels == 1)))
        precision = tp / max(1, int(predicted.sum()))
        rank = (recall, precision, -fpr, float(threshold))
        if fpr <= fpr_limit and (best is None or rank > best[0]):
            best = (rank, {"threshold": float(threshold), "recall": recall, "fpr": fpr,
                           "precision": precision, "selection_status": "validation_only"})
    return best[1]


def _summary(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "count": int(len(labels)), "positives": int(np.sum(labels)),
        "average_precision": float(average_precision_score(labels, scores)) if np.any(labels) else None,
        "brier": float(brier_score_loss(labels, scores)) if len(labels) else None,
    }


def _versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib"]}


def _check_frames(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if train.empty or validation.empty:
        raise ValueError("Nonempty train and validation frames are required")
    for frame in [train, validation]:
        if not frame["event_id"].is_unique:
            raise ValueError("Each modeling frame must have unique event IDs")
        for label in TARGETS.values():
            if not frame[label].isin([0, 1]).all():
                raise ValueError(f"Invalid binary label: {label}")
    if set(train["event_id"]) & set(validation["event_id"]):
        raise ValueError("Train and validation event IDs overlap")
    if pd.to_datetime(train.event_time, utc=True).max() >= pd.to_datetime(validation.event_time, utc=True).min():
        raise ValueError("Training must end strictly before validation begins")


def train_baselines(train: pd.DataFrame, validation: pd.DataFrame, output_dir: Path) -> dict:
    """Fit train-only candidates; calibrate and select using disjoint validation blocks."""
    _check_frames(train, validation)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_frame, selection_frame = split_validation(validation)
    behavior = list(BEHAVIOR_FEATURES)
    relationship = list(dict.fromkeys([*BEHAVIOR_FEATURES, *GRAPH_FEATURES]))
    bundle = {"model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION, "targets": {},
              "candidate_models": {}, "calibration_population": "simulated validation population"}
    summary = {"model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION, "versions": _versions(),
               "fpr_limit": FPR_LIMIT, "selection_policy": "highest later-validation AP, tie-break lowest Brier",
               "calibration_policy": "Platt scaling on earlier half of validation timestamps",
               "calibration_population": "simulated validation population", "targets": {},
               "train_event_count": len(train), "calibration_event_count": len(calibration_frame),
               "selection_event_count": len(selection_frame)}
    predictions = []
    for target, label in TARGETS.items():
        target_train = _latest_account(train) if target == "identity" else train
        calibration = _latest_account(calibration_frame) if target == "identity" else calibration_frame
        selection = _latest_account(selection_frame) if target == "identity" else selection_frame
        all_candidates = {}
        comparisons = {}
        for name, names in [("rules", relationship), ("logistic", behavior),
                            ("lightgbm_behavior", behavior), ("lightgbm_graph", relationship),
                            ("lightgbm_no_device", [n for n in relationship if "device" not in n]),
                            ("lightgbm_no_assurance", [n for n in relationship if n not in
                             {"binding_mismatch", "binding_unverified", "binding_unknown", "challenge_failures", "recovery_used"}])]:
            x_train = _numeric(target_train, names)
            y_train = target_train[label].to_numpy(dtype=int)
            if name == "rules":
                estimator = RuleEstimator(names, target)
            elif len(np.unique(y_train)) < 2:
                estimator = DummyClassifier(strategy="prior").fit(x_train, y_train)
            elif name == "logistic":
                estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
                estimator.fit(x_train, y_train)
            else:
                estimator = LGBMClassifier(n_estimators=100, num_leaves=12, max_depth=5, learning_rate=0.05,
                                           min_child_samples=15, reg_lambda=2.0, verbosity=-1, n_jobs=2,
                                           random_state=42, deterministic=True, force_col_wise=True)
                estimator.fit(x_train, y_train)
            calibrator = None
            calibration_status = "skipped_insufficient_validation_classes"
            if len(calibration) >= 12 and calibration[label].nunique() == 2:
                raw = _positive_probability(estimator, _numeric(calibration, names))
                logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
                calibrator = LogisticRegression(C=1.0, max_iter=300, random_state=42)
                calibrator.fit(logits.reshape(-1, 1), calibration[label].to_numpy(dtype=int))
                calibration_status = "platt_earlier_validation"
            calibrated = CalibratedEstimator(estimator, calibrator)
            select_scores = calibrated.predict_score(_numeric(selection, names))
            threshold_result = choose_threshold(selection[label].to_numpy(), select_scores)
            descriptor = {"estimator": calibrated, "feature_names": names, "model_name": name,
                          "threshold": threshold_result["threshold"], "label": label,
                          "calibration_status": calibration_status}
            all_candidates[name] = descriptor
            comparison = {**_summary(selection[label].to_numpy(), select_scores), **threshold_result,
                          "feature_names": names, "calibration_status": calibration_status,
                          "calibration_examples": len(calibration), "fit_examples": len(target_train)}
            comparisons[name] = comparison
            for split_name, frame in [("train", train), ("validation_calibration", calibration_frame),
                                      ("validation_selection", selection_frame)]:
                if frame.empty:
                    continue
                result = frame[["event_id", "account_id", "event_time"]].copy()
                result["target"] = target
                result["label"] = frame[label].to_numpy(dtype=int)
                result["model"] = name
                result["split"] = split_name
                result["score"] = calibrated.predict_score(_numeric(frame, names))
                result["threshold"] = descriptor["threshold"]
                predictions.append(result)
        def rank(model_name):
            metrics = comparisons[model_name]
            ap = metrics["average_precision"]
            return (-1 if ap is None else ap, -(metrics["brier"] or 0))
        selected_name = max((n for n in all_candidates if n not in ("lightgbm_no_device", "lightgbm_no_assurance")), key=rank)
        bundle["targets"][target] = all_candidates[selected_name]
        bundle["candidate_models"][target] = all_candidates
        summary["targets"][target] = {"selected_model": selected_name, "models": comparisons}
    bundle["metadata"] = summary
    joblib.dump(bundle, output_dir / "bundle.joblib")
    (output_dir / "model_metadata.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "baseline_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "baseline_predictions.csv", index=False)
    return summary


def load_bundle(path) -> dict:
    """Load only locally generated, trusted joblib artifacts (pickle is executable)."""
    path = Path(path)
    bundle = joblib.load(path / "bundle.joblib" if path.is_dir() else path)
    if bundle.get("model_version") not in (MODEL_VERSION, "identitygraph-baselines-v1") or set(bundle.get("targets", {})) != set(TARGETS):
        raise ValueError("Unsupported model bundle")
    return bundle


def predict_bundle(bundle: dict, features: dict) -> dict:
    result = {}
    for target, descriptor in bundle["targets"].items():
        values = _numeric(pd.DataFrame([features]), descriptor["feature_names"])
        score = float(descriptor["estimator"].predict_score(values)[0])
        threshold = float(descriptor["threshold"])
        result[target] = {"score": score, "threshold": threshold,
                          "decision": "manual_review" if score >= threshold else "low_risk"}
    return result


def predict_baselines(bundle_or_dir, frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Long-form held-out predictions for every fitted candidate, without refitting."""
    bundle = bundle_or_dir if isinstance(bundle_or_dir, dict) else load_bundle(bundle_or_dir)
    rows = []
    for target, candidates in bundle["candidate_models"].items():
        for name, descriptor in candidates.items():
            result = frame[["event_id", "account_id", "event_time"]].copy()
            result["target"] = target
            result["label"] = frame[TARGETS[target]].to_numpy() if TARGETS[target] in frame else np.nan
            result["model"] = name
            result["split"] = split
            result["score"] = descriptor["estimator"].predict_score(_numeric(frame, descriptor["feature_names"])) if len(frame) else []
            result["threshold"] = descriptor["threshold"]
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def selected_bundle_predictions(bundle: dict, frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    result = frame[[name for name in ["event_id", "account_id", "event_time", *TARGETS.values()] if name in frame]].copy()
    result["split"] = split
    for target, descriptor in bundle["targets"].items():
        result[f"{target}_score"] = descriptor["estimator"].predict_score(_numeric(frame, descriptor["feature_names"])) if len(frame) else []
        result[f"{target}_threshold"] = descriptor["threshold"]
    return result
