"""Observable drift summaries and explicit, delayed feedback evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def population_shift(reference, current, features):
    results = {}
    for name in features:
        old = np.asarray(reference.get(name, []), dtype=float)
        new = np.asarray(current.get(name, []), dtype=float)
        old, new = old[np.isfinite(old)], new[np.isfinite(new)]
        if not len(old) or not len(new):
            results[name] = {"status": "insufficient_data", "reference_n": len(old), "current_n": len(new)}
            continue
        boundaries = np.unique(np.quantile(old, np.linspace(0, 1, 11)))
        if len(boundaries) == 1:
            boundaries = np.array([-np.inf, boundaries[0] + 1e-9, np.inf])
        else:
            boundaries[0], boundaries[-1] = -np.inf, np.inf
        a, b = np.histogram(old, boundaries)[0].astype(float), np.histogram(new, boundaries)[0].astype(float)
        a, b = (a + .5) / (len(old) + .5 * len(a)), (b + .5) / (len(new) + .5 * len(b))
        psi = float(np.sum((b - a) * np.log(b / a)))
        results[name] = {"reference_n": len(old), "current_n": len(new), "reference_mean": float(old.mean()),
                         "current_mean": float(new.mean()), "population_stability_index": psi,
                         "status": "descriptive_only" if len(new) < 100 else "shift_observed" if psi >= .2 else "stable_by_heuristic"}
    return {"features": results, "note": "PSI uses frozen reference bins and a 0.2 descriptive threshold. Shift is not proof of accuracy loss."}


def feedback_metrics(results, feedback, as_of):
    records = []
    for item in feedback:
        result = results.get(item["event_id"])
        if item["observed_at"] <= as_of and result and result["status"] == "assessed":
            records.append({**item, "account_id": result["account_id"], "event_time": result["event_time"],
                            "score": result[item["target"]]["score"], "threshold": result[item["target"]]["threshold"]})
    output = {}
    for target in ("takeover", "identity"):
        frame = pd.DataFrame([row for row in records if row["target"] == target])
        if frame.empty:
            output[target] = {"status": "no_available_labels", "n": 0}
            continue
        if target == "identity":
            frame = frame.sort_values(["event_time", "event_id"]).drop_duplicates("account_id", keep="last")
        y, alerts = frame.label.to_numpy(), (frame.score >= frame.threshold).to_numpy()
        output[target] = {"status": "available_feedback_only", "n": len(frame), "positives": int(y.sum()),
                          "average_precision": float(average_precision_score(y, frame.score)) if y.sum() else None,
                          "brier": float(brier_score_loss(y, frame.score)),
                          "recall": float((alerts & (y == 1)).sum() / y.sum()) if y.sum() else None,
                          "false_positive_rate": float((alerts & (y == 0)).sum() / (y == 0).sum()) if (y == 0).any() else None}
    return {"as_of": as_of, "targets": output,
            "note": "Only submitted labels available by this cutoff are evaluated. Feedback may be selectively reported; it never updates a prediction or model."}
