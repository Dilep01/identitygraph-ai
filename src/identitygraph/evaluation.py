"""Held-out metrics and candidate coverage; truth is used only in evaluation."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def metrics(predictions: pd.DataFrame, review_budget: int = 20, bootstrap: int = 100) -> dict:
    if predictions.empty:
        return {"n": 0, "positives": 0, "status": "no_examples"}
    frame = predictions.sort_values(["event_time", "event_id"]).copy()
    if frame.target.iloc[0] == "identity":
        frame = frame.drop_duplicates("account_id", keep="last")
    y, p = frame.label.to_numpy(dtype=int), frame.score.to_numpy(dtype=float)
    threshold = float(frame.threshold.iloc[0])
    positive = p >= threshold
    tp, fp = int(np.sum(positive & (y == 1))), int(np.sum(positive & (y == 0)))
    positives, negatives = int(y.sum()), int((y == 0).sum())
    top = np.argsort(-p, kind="stable")[:min(review_budget, len(p))]
    calibration = []
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            confidence, rate = float(p[mask].mean()), float(y[mask].mean())
            ece += mask.mean() * abs(confidence - rate)
            calibration.append({"lower": float(lo), "upper": float(hi), "n": int(mask.sum()),
                                "mean_score": confidence, "observed_rate": rate})
    # Resample whole accounts rather than treating repeated sessions as independent.
    grouped = {a: g.index.to_numpy() for a, g in frame.reset_index(drop=True).groupby("account_id")}
    account_keys = list(grouped)
    rng = np.random.default_rng(734)
    estimates = []
    if positives and negatives and len(account_keys) > 1:
        for _ in range(bootstrap):
            indices = np.concatenate([grouped[account_keys[int(k)]] for k in
                                      rng.integers(0, len(account_keys), len(account_keys))])
            if 0 < y[indices].sum() < len(indices):
                estimates.append(float(average_precision_score(y[indices], p[indices])))
    days = max(1, frame.event_time.str[:10].nunique())
    return {
        "n": len(y), "positives": positives, "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if positives else None,
        "average_precision_account_bootstrap_95ci": list(map(float, np.quantile(estimates, [0.025, 0.975])))
        if estimates else None,
        "brier": float(brier_score_loss(y, p)), "calibration_error_10_bins": float(ece),
        "calibration_bins": calibration, "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
        "false_alerts_per_1000": 1000 * fp / negatives if negatives else None,
        "alerts": int(positive.sum()), "alerts_per_observed_day": float(positive.sum() / days),
        "precision_at_k": float(y[top].mean()), "k": len(top),
        "unit": "latest_assessment_per_account" if frame.target.iloc[0] == "identity" else "successful_session",
        "threshold_policy": "Frozen from later validation; target FPR <= 2% there, not guaranteed on test",
        "uncertainty_note": "Account-block bootstrap; dependence through shared infrastructure remains.",
    }


def cluster_metrics(clusters: list[dict], truth_groups: pd.DataFrame,
                    eligible_accounts: set[str]) -> dict:
    truth = truth_groups[truth_groups.account_id.isin(eligible_accounts)]
    groups = {str(key): set(g.account_id) for key, g in truth.groupby("attack_group")}
    predicted = [set(c["account_ids"]) & eligible_accounts for c in clusters]
    predicted = [p for p in predicted if len(p) >= 2]
    truth_pairs = {tuple(sorted(pair)) for g in groups.values() for pair in combinations(g, 2)}
    candidate_pairs = {tuple(sorted(pair)) for g in predicted for pair in combinations(g, 2)}
    matched = candidate_pairs & truth_pairs
    reached = {gid for gid, g in groups.items() if any(len(g & p) >= 2 for p in predicted)}
    fully_recovered = {gid for gid, g in groups.items() if any(g <= p for p in predicted)}
    malicious = set().union(*groups.values()) if groups else set()
    benign = eligible_accounts - malicious
    benign_flagged = set().union(*predicted) & benign if predicted else set()
    return {
        "ground_truth_groups": len(groups), "candidate_clusters": len(predicted),
        "groups_reached": len(reached), "candidate_group_coverage": len(reached) / len(groups) if groups else None,
        "fully_contained_groups": len(fully_recovered),
        "pair_precision": len(matched) / len(candidate_pairs) if candidate_pairs else None,
        "pair_recall": len(matched) / len(truth_pairs) if truth_pairs else None,
        "benign_accounts_flagged": len(benign_flagged), "eligible_benign_accounts": len(benign),
        "benign_account_flag_rate": len(benign_flagged) / len(benign) if benign else None,
        "missed_groups": sorted(set(groups) - reached),
        "coverage_definition": "At least two members of a true group in one candidate; all eligible true groups in denominator",
        "containment_note": "Containment is not purity; pair precision penalizes unrelated accounts merged into candidates",
    }
