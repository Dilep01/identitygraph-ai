"""Fictional observable records and physically separate simulation truth.

The simulator defines actor intent; classifiers never import this module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 41
    accounts: int = 240
    days: int = 36
    train_days: int = 18
    validation_days: int = 8
    cold_fraction: float = 0.20
    synthetic_fraction: float = 0.18
    takeover_rate: float = 0.045
    daily_sessions: float = 0.85
    churn_rate: float = 0.10
    missing_rate: float = 0.04
    attribute_corruption: float = 0.12
    shifted: bool = False
    start: str = "2026-01-01T00:00:00Z"

    def validate(self):
        if self.accounts < 40 or self.days <= self.train_days + self.validation_days:
            raise ValueError("Use at least 40 accounts and a nonempty chronological test window")
        for field in ("cold_fraction", "synthetic_fraction", "takeover_rate", "churn_rate",
                      "missing_rate", "attribute_corruption"):
            if not 0 <= getattr(self, field) < 1:
                raise ValueError(f"{field} must be in [0, 1)")
        if self.daily_sessions <= 0 or self.train_days < 4 or self.validation_days < 4:
            raise ValueError("Session rate must be positive and training/validation need >=4 days")


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output_dir: Path, config: SimulationConfig = SimulationConfig()) -> dict:
    config.validate()
    output_dir = Path(output_dir)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Dataset already exists: {output_dir}; use a new run name")
    rng = np.random.default_rng(config.seed)
    start = datetime.fromisoformat(config.start.replace("Z", "+00:00"))
    train_end = start + timedelta(days=config.train_days)
    validation_end = train_end + timedelta(days=config.validation_days)
    end = start + timedelta(days=config.days)

    def token(prefix):
        return prefix + "_" + rng.bytes(8).hex()

    n = config.accounts
    cold_n = int(n * config.cold_fraction)
    permutation = rng.permutation(n)
    cold = set(int(i) for i in permutation[:cold_n])
    warm = [int(i) for i in permutation[cold_n:]]
    cold_list = sorted(cold)
    def partition(pool):
        result, position = [], 0
        while position < len(pool):
            size = int(rng.integers(2, 7))
            result.append(list(map(int, pool[position:position + size])))
            position += size
        if len(result) > 1 and len(result[-1]) == 1:
            result[-2].extend(result.pop())
        return result

    malicious_groups = []
    for pool in (warm, cold_list):
        count = max(2, int(len(pool) * config.synthetic_fraction)) if config.synthetic_fraction else 0
        selected = list(rng.permutation(pool)[:count])
        malicious_groups.extend(partition(selected))
    synthetic = {i for group in malicious_groups for i in group}
    benign = [int(i) for i in rng.permutation(n) if int(i) not in synthetic]
    household = {}
    for members in partition(benign):
        group_id = token("group")
        for i in members:
            household[i] = group_id
    fraud_group = {}
    sparse_groups = set()
    for j, members in enumerate(malicious_groups):
        group_id = token("group")
        # One cold family intentionally has weak linkage; never guarantee candidate recovery.
        if all(i in cold for i in members) and j == len(malicious_groups) - 1:
            sparse_groups.add(group_id)
        for i in members:
            fraud_group[i] = group_id
            household[i] = token("group")

    account_ids = [token("acct") for _ in range(n)]
    personal_devices = [token("dev") for _ in range(n)]
    household_devices = {g: token("dev") for g in sorted(set(household.values()))}
    household_network = {g: token("net") for g in sorted(set(household.values()))}
    # Some legitimate relatives live separately; different networks are not proof of fraud.
    home_network = [token("net") if rng.random() < 0.16 else household_network[household[i]] for i in range(n)]
    public_networks = [token("net") for _ in range(8)]
    attack_devices = [token("dev") for _ in range(7)]
    attack_networks = [token("net") for _ in range(6)]
    group_devices = {g: token("dev") for g in sorted(set(fraud_group.values()))}
    shared_email = {}
    for key in sorted(set(household.values()) | set(fraud_group.values())):
        rate = 0.75 if key in group_devices else 0.12
        if rng.random() < rate:
            shared_email[key] = f"person{rng.bytes(5).hex()}@example.test"
    attributes, links, accounts, account_truth, groups = [], [], [], [], []
    attribute_by_key = {}
    birth = {}

    def attribute(kind, value):
        key = (kind, value)
        if key not in attribute_by_key:
            aid = token("attr")
            attribute_by_key[key] = aid
            attributes.append({"attribute_id": aid, "attribute_type": kind, "value": value})
        return attribute_by_key[key]

    for i in range(n):
        created = (validation_end if i in cold else start) + timedelta(hours=int(rng.integers(0, 36)))
        birth[i] = created
        accounts.append({"account_id": account_ids[i], "created_at": iso(created)})
        group = fraud_group.get(i, "")
        shared = group if group else household[i]
        values = {"email": f"person{rng.bytes(5).hex()}@example.test",
                  "phone": f"phone-token-{shared[-9:]}", "address": f"fictional lane {shared[-8:]}"}
        if shared in shared_email:
            values["email"] = shared_email[shared]
        if group in sparse_groups:
            values["address"] = f"fictional lane {rng.bytes(5).hex()}"
        for kind, val in values.items():
            if rng.random() < config.missing_rate:
                continue
            if kind == "address" and rng.random() < config.attribute_corruption:
                val = val.replace("lane", "ln")
            aid = attribute(kind, val)
            links.append({"account_id": account_ids[i], "attribute_id": aid, "observed_at": iso(created)})
        scenario = "sparse_identity" if group in sparse_groups else ("fabricated_identity" if group else "legitimate")
        account_truth.append({"account_id": account_ids[i], "is_synthetic": int(i in synthetic),
                              "owner_group": token("owner"), "household_group": household[i],
                              "attack_group": group, "scenario": scenario,
                              "cohort": "cold_start" if i in cold else "train",
                              "label_available_at": iso(created + timedelta(days=3))})
        if group:
            groups.append({"attack_group": group, "account_id": account_ids[i]})

    events, event_truth = [], []

    def emit(i, when, device, network, region, outcome, auth, browser, quality, attack, scenario):
        eid = token("evt")
        # These are simulated pre-decision observations, not labels returned by an oracle.
        # Legitimate failures/mismatches and silent attacks both occur.
        visible_attack = attack and scenario != "familiar_session"
        mismatch_rate = 0.60 if visible_attack else 0.012
        binding = "mismatch" if rng.random() < mismatch_rate else ("unverified" if rng.random() < 0.12 else "verified")
        challenge_failures = int(rng.integers(1, 4)) if rng.random() < (0.40 if visible_attack else 0.035) else 0
        recovery = bool(rng.random() < (0.28 if visible_attack else 0.025))
        if rng.random() < config.missing_rate:
            binding, challenge_failures, recovery = "unknown", 0, False
        events.append({"event_id": eid, "account_id": account_ids[i], "device_id": device,
                       "ip_id": network, "event_time": iso(when), "received_at": iso(when),
                       "region": region, "outcome": outcome, "auth_method": auth,
                       "browser": browser, "device_quality": quality,
                       "session_binding": binding, "challenge_failures": challenge_failures,
                       "recovery_used": recovery})
        event_truth.append({"event_id": eid, "is_takeover": int(attack and outcome == "success"),
                            "scenario": scenario, "attack_group": "episode_" + network if attack else "",
                            "label_available_at": iso(when + timedelta(days=1))})

    for day in range(config.days):
        for i in rng.permutation(n):
            i = int(i)
            if start + timedelta(days=day + 1) <= birth[i]:
                continue
            offsets = sorted(rng.integers(0, 86400, size=int(rng.poisson(config.daily_sessions))).tolist())
            for offset in offsets:
                when = start + timedelta(days=day, seconds=int(offset))
                if when < birth[i]:
                    continue
                attack = rng.random() < config.takeover_rate and day >= 4
                device, network = personal_devices[i], home_network[i]
                region, scenario, quality = f"region_{i % 4}", "routine", "stable"
                browser = str(rng.choice(["chrome", "firefox", "safari"], p=[0.6, 0.2, 0.2]))
                auth = "mfa" if rng.random() < 0.65 else "password"
                if attack:
                    unseen = day >= config.train_days + config.validation_days and rng.random() < 0.35
                    if unseen:
                        scenario = "familiar_session" if rng.random() < 0.5 else "session_replay"
                        # Unauthorized session with ordinary-looking infrastructure: deliberate hard case.
                    else:
                        scenario = "credential_reuse" if rng.random() < 0.5 else "new_infrastructure"
                        device, network = str(rng.choice(attack_devices)), str(rng.choice(attack_networks))
                        if rng.random() < 0.75:
                            region = f"region_{int(rng.integers(0, 8))}"
                        if scenario == "credential_reuse":
                            for k in range(int(rng.integers(1, 4))):
                                failed_at = when - timedelta(seconds=10 + 15 * k)
                                if failed_at >= birth[i]:
                                    emit(i, failed_at, device, network, region, "failure", auth,
                                         browser, quality, True, scenario)
                else:
                    roll = rng.random()
                    if roll < config.churn_rate:
                        device = token("dev")
                        if rng.random() < 0.55:
                            personal_devices[i] = device
                        quality, scenario = "reset", "new_phone"
                        if rng.random() < 0.3:
                            network, region = token("net"), f"region_{int(rng.integers(0, 8))}"
                    elif roll < config.churn_rate + 0.13:
                        device, scenario = household_devices[household[i]], "household_sharing"
                    elif roll < config.churn_rate + 0.26:
                        network, scenario = str(rng.choice(public_networks)), "public_network"
                    elif roll < config.churn_rate + 0.38:
                        network, scenario = token("net"), "mobile_network"
                    group = fraud_group.get(i)
                    if group and group not in sparse_groups and rng.random() < 0.30:
                        device = group_devices[group]
                        quality, scenario = "stable", "identity_shared_device"
                outcome = "failure" if not attack and rng.random() < 0.07 else "success"
                emit(i, when, device, network, region, outcome, auth, browser, quality, attack, scenario)

    observed_dir, truth_dir = output_dir / "observed", output_dir / "truth"
    observed_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "observed/accounts": pd.DataFrame(accounts).sort_values("account_id"),
        "observed/attributes": pd.DataFrame(attributes).sort_values("attribute_id"),
        "observed/account_attributes": pd.DataFrame(links).sort_values(["account_id", "attribute_id"]),
        "observed/events": pd.DataFrame(events).sort_values(["received_at", "event_time", "event_id"]),
        "truth/accounts": pd.DataFrame(account_truth).sort_values("account_id"),
        "truth/events": pd.DataFrame(event_truth).sort_values("event_id"),
        "truth/groups": pd.DataFrame(groups, columns=["attack_group", "account_id"]).sort_values(["attack_group", "account_id"]),
    }
    files = {}
    for name, frame in frames.items():
        path = output_dir / (name + ".parquet")
        frame.to_parquet(path, index=False)
        files[name + ".parquet"] = {"rows": len(frame), "sha256": digest(path)}
    manifest = {
        "schema_version": "2.0", "generator_version": "1.0.0", "provenance": "fully_synthetic",
        "config": asdict(config), "files": files,
        "split_cutoffs": {"train_cutoff": iso(train_end), "validation_cutoff": iso(validation_end),
                          "test_end": iso(end)}, "replay_cutoff": iso(validation_end),
        "known_attack_families": ["new_infrastructure", "credential_reuse"],
        "heldout_attack_family": "familiar_session",
        "heldout_attack_families": ["familiar_session", "session_replay"],
        "group_sizes": {"fabricated": [len(g) for g in malicious_groups],
                        "legitimate_households": list(pd.Series([household[i] for i in benign]).value_counts().astype(int))},
        "limitations": ["Actor intent is generated, not independently observed.",
                        "No production identity, authentication, fraud, or personal data.",
                        "Hard familiar-session attacks may be indistinguishable from legitimate activity.",
                        "All scenarios are generated by one simulator; shift tests are not external validation."],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_observed(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(Path(data_dir) / "observed" / f"{name}.parquet")
            for name in ("accounts", "attributes", "account_attributes", "events")}
