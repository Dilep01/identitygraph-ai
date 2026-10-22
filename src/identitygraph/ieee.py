"""Optional local IEEE-CIS transaction benchmark; never relabel it as takeover data."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss

from .models import choose_threshold
from .simulator import digest

TRANSACTION_COLUMNS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
                       "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
                       "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
IDENTITY_COLUMNS = ["TransactionID", "DeviceType", "DeviceInfo", "id_01", "id_02", "id_03", "id_05",
                    "id_06", "id_09", "id_10", "id_11", "id_12", "id_13", "id_14", "id_15", "id_16"]


def csv_audit(path):
    missing, rows, ids, duplicates = Counter(), 0, set(), 0
    fields = list(pd.read_csv(path, nrows=0).columns)
    if "TransactionID" not in fields:
        raise ValueError(f"{path.name} is missing TransactionID")
    for chunk in pd.read_csv(path, chunksize=20000, low_memory=False):
        rows += len(chunk)
        missing.update(chunk.isna().sum().to_dict())
        if chunk.TransactionID.isna().any():
            raise ValueError(f"{path.name} has missing join IDs")
        for identifier in chunk.TransactionID:
            duplicates += int(identifier in ids)
            ids.add(identifier)
    return {"rows": rows, "columns": fields, "missing_fraction": {k: v / max(rows, 1) for k, v in missing.items()},
            "duplicate_join_ids": duplicates, "sha256": digest(path)}, ids


def temporal_relationship_features(frame):
    """Counts use earlier timestamp buckets; missing values never form shared identities."""
    counts = {name: Counter() for name in ("card1", "DeviceInfo", "P_emaildomain")}
    rows = np.zeros((len(frame), len(counts)), dtype=float)
    for _, group in frame.groupby("TransactionDT", sort=True):
        for index, row in group.iterrows():
            for column, name in enumerate(counts):
                value = row.get(name)
                rows[index, column] = counts[name][str(value)] if pd.notna(value) else 0
        for _, row in group.iterrows():
            for name in counts:
                value = row.get(name)
                if pd.notna(value):
                    counts[name][str(value)] += 1
    return pd.DataFrame(rows, columns=[name + "_prior_count" for name in counts])


def run_ieee(data_dir: Path, output: Path, audit_only=False):
    files = [Path(data_dir) / name for name in ("train_transaction.csv", "train_identity.csv")]
    absent = [p.name for p in files if not p.is_file()]
    if absent:
        raise FileNotFoundError("IEEE-CIS source files are not supplied: " + ", ".join(absent) +
                                ". Obtain them through the dataset's authorized access; no benchmark result was generated.")
    if output.exists():
        raise FileExistsError("Choose a new output run to preserve benchmark evidence")
    audits = [csv_audit(p) for p in files]
    if any(audit[0]["duplicate_join_ids"] for audit in audits):
        raise ValueError("Duplicate TransactionID values prevent a validated one-to-one join")
    common = audits[0][1] & audits[1][1]
    report = {"dataset": "IEEE-CIS", "target": "transaction_fraud", "takeover_labels_available": False,
              "synthetic_identity_labels_available": False, "files": {p.name: a[0] for p, a in zip(files, audits)},
              "join_coverage": len(common) / max(1, len(audits[0][1])),
              "identity_orphans": len(audits[1][1] - audits[0][1]), "status": "audit_complete"}
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if audit_only:
        return report
    transaction = pd.read_csv(files[0], usecols=lambda name: name in TRANSACTION_COLUMNS, low_memory=False)
    identity = pd.read_csv(files[1], usecols=lambda name: name in IDENTITY_COLUMNS, low_memory=False)
    for required in ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud"):
        if required not in transaction:
            raise ValueError(f"Missing required column {required}")
    frame = transaction.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    frame = frame.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    if frame.TransactionDT.isna().any() or not frame.isFraud.isin([0, 1]).all():
        raise ValueError("Invalid timestamps or transaction-fraud labels")
    times = np.sort(frame.TransactionDT.unique())
    if len(times) < 10:
        raise ValueError("At least ten distinct chronological timestamps are required")
    train_cut, val_cut = times[int(len(times) * .6)], times[int(len(times) * .8)]
    masks = {"train": frame.TransactionDT < train_cut,
             "validation": (frame.TransactionDT >= train_cut) & (frame.TransactionDT < val_cut),
             "test": frame.TransactionDT >= val_cut}
    excluded = {"TransactionID", "TransactionDT", "isFraud"}
    numeric = pd.DataFrame(index=frame.index)
    encoding = {}
    for name in frame.columns.difference(list(excluded)):
        if pd.api.types.is_numeric_dtype(frame[name]):
            median = float(frame.loc[masks["train"], name].median())
            median = median if np.isfinite(median) else 0.0
            numeric[name] = frame[name].fillna(median)
            numeric[name + "_missing"] = frame[name].isna().astype(float)
            encoding[name] = {"median": median}
        else:
            frequency = frame.loc[masks["train"], name].fillna("__MISSING__").value_counts(normalize=True).to_dict()
            numeric[name + "_training_frequency"] = frame[name].fillna("__MISSING__").map(frequency).fillna(0)
            encoding[name] = {"training_frequency": frequency}
    numeric["hour_utc_offset"] = (frame.TransactionDT % 86400) / 3600
    relationship = pd.concat([numeric, temporal_relationship_features(frame)], axis=1)
    predictions, scores = [], {}
    for name, values in (("tabular", numeric), ("historical_relationships", relationship)):
        model = LGBMClassifier(n_estimators=150, num_leaves=20, max_depth=6, learning_rate=.05,
                              verbosity=-1, n_jobs=2, random_state=42, deterministic=True, force_col_wise=True)
        model.fit(values[masks["train"]].to_numpy(), frame.loc[masks["train"], "isFraud"])
        val = model.predict_proba(values[masks["validation"]].to_numpy())[:, 1]
        chosen = choose_threshold(frame.loc[masks["validation"], "isFraud"].to_numpy(), val)
        test = model.predict_proba(values[masks["test"]].to_numpy())[:, 1]
        y = frame.loc[masks["test"], "isFraud"].to_numpy()
        alerts = test >= chosen["threshold"]
        scores[name] = {"average_precision": float(average_precision_score(y, test)) if y.any() else None,
                       "brier_uncalibrated": float(brier_score_loss(y, test)), "threshold_selection": chosen,
                       "recall": float((alerts & (y == 1)).sum() / max(1, y.sum())),
                       "false_positive_rate": float((alerts & (y == 0)).sum() / max(1, (y == 0).sum()))}
        predictions.append(pd.DataFrame({"TransactionID": frame.loc[masks["test"], "TransactionID"],
                                          "label": y, "score": test, "model": name}))
        joblib.dump({"model": model, "features": list(values.columns), "encoding": encoding,
                     "threshold": chosen}, output / f"{name}.joblib")
    pd.concat(predictions).to_parquet(output / "predictions.parquet", index=False)
    report.update(status="benchmark_complete", models=scores,
                  split_counts={k: int(v.sum()) for k, v in masks.items()},
                  limitations=["Transaction-fraud labels are not identity or account-takeover labels.",
                               "Fixed chronological 60/20/20 timestamp split; no test tuning.",
                               "Historical recurrence counts are relationship proxies, not verified people.",
                               "Scores are not calibrated; earlier observed test transactions update counts without labels."])
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
