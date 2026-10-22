"""Observable, strictly historical features for ordered synthetic event replay.

Current successful-session context may be used, but outcomes and relationships
from events at the same timestamp are excluded from every historical feature.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import re

import pandas as pd

FEATURE_VERSION = "2.0.0"
BEHAVIOR_FEATURES = [
    "account_age_hours", "prior_login_count", "prior_failure_count",
    "prior_success_rate", "login_count_1h", "login_count_24h",
    "failure_count_24h", "hours_since_last_login", "new_device", "new_ip",
    "new_region", "new_browser", "password_auth", "device_quality_reset",
    "binding_mismatch", "binding_unverified", "binding_unknown", "challenge_failures", "recovery_used",
]
GRAPH_FEATURES = [
    "account_device_count", "account_ip_count", "device_other_account_count",
    "ip_other_account_count", "device_event_count_24h", "ip_event_count_24h",
    "shared_attribute_account_count", "shared_attribute_type_count",
    "attribute_count", "missing_attribute_type_count", "device_account_overlap",
    "ip_account_overlap",
    "shared_contact_different_home_count",
]
EVENT_COLUMNS = [
    "event_id", "account_id", "device_id", "ip_id", "event_time", "received_at",
    "region", "outcome", "auth_method", "browser", "device_quality",
]
ATTRIBUTE_TYPES = ("email", "phone", "address")


def utc_time(value) -> datetime:
    """Require timezone-aware input; normalize equivalent UTC representations."""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Timestamps must include a UTC offset")
    return value.astimezone(timezone.utc)


def iso_time(value) -> str:
    return utc_time(value).isoformat()


def normalize_attribute(kind: str, value: str) -> str:
    """Normalize fictional strings without inferring that accounts are identical."""
    text = " ".join(str(value).strip().casefold().split())
    if kind == "address":
        text = re.sub(r"\bln\b", "lane", text)
    if kind == "phone" and not any(character.isalpha() for character in text):
        return re.sub(r"[^0-9]", "", text)
    return text


def observable_event(event: dict) -> dict:
    """Copy only the declared observed schema, excluding any attached truth."""
    missing = [key for key in EVENT_COLUMNS if key not in event]
    if missing:
        raise ValueError(f"Missing event fields: {', '.join(missing)}")
    result = {key: str(event[key]) for key in EVENT_COLUMNS}
    for key in ("event_id", "account_id", "device_id", "ip_id"):
        if not result[key].strip():
            raise ValueError(f"{key} must be nonempty")
    result["event_time"] = iso_time(result["event_time"])
    result["received_at"] = iso_time(result["received_at"])
    if result["received_at"] != result["event_time"]:
        raise ValueError("This ordered-replay version requires receipt time equal to event time")
    if result["outcome"] not in ("success", "failure"):
        raise ValueError("outcome must be success or failure")
    if result["auth_method"] not in ("password", "mfa"):
        raise ValueError("auth_method must be password or mfa")
    if result["device_quality"] not in ("stable", "reset"):
        raise ValueError("device_quality must be stable or reset")
    def optional(key, default):
        value = event.get(key, default)
        return default if value is None or isinstance(value, float) and pd.isna(value) else value

    binding = optional("session_binding", "unknown")
    failures = optional("challenge_failures", 0)
    recovery = optional("recovery_used", False)
    if binding not in ("verified", "unverified", "mismatch", "unknown"):
        raise ValueError("Invalid session_binding")
    if isinstance(failures, bool) or not isinstance(failures, (int, float)) or not 0 <= failures <= 10 or int(failures) != failures:
        raise ValueError("challenge_failures must be an integer between 0 and 10")
    if not isinstance(recovery, bool):
        raise ValueError("recovery_used must be boolean")
    result.update(session_binding=binding, challenge_failures=int(failures), recovery_used=recovery)
    return result


def primary_network(events):
    from collections import Counter
    counts = Counter(e["ip_id"] for e in events if e["outcome"] == "success")
    if not counts or sum(counts.values()) < 3:
        return None
    key, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return key if count / sum(counts.values()) >= 0.4 else None


class _Timeline:
    def __init__(self):
        self.times: list[datetime] = []
        self.events: list[dict] = []

    def add(self, at: datetime, event: dict):
        index = bisect_right(self.times, at)
        self.times.insert(index, at)
        self.events.insert(index, event)

    def before(self, at: datetime) -> list[dict]:
        return self.events[:bisect_left(self.times, at)]

    def between(self, start: datetime, end: datetime) -> list[dict]:
        return self.events[bisect_left(self.times, start):bisect_left(self.times, end)]

    def count_between(self, start: datetime, end: datetime) -> int:
        return bisect_left(self.times, end) - bisect_left(self.times, start)


class HistoryState:
    """History contains observed schema only; no label tables are accepted.

    Timeline filtering makes equal-time behavior safe even when a caller has
    already inserted another member of the current timestamp bucket.
    """
    def __init__(self, accounts: pd.DataFrame, attributes: pd.DataFrame,
                 account_attributes: pd.DataFrame):
        if accounts["account_id"].astype(str).duplicated().any():
            raise ValueError("Duplicate account_id")
        if attributes["attribute_id"].astype(str).duplicated().any():
            raise ValueError("Duplicate attribute_id")
        self.account_created = {
            str(row.account_id): utc_time(row.created_at)
            for row in accounts[["account_id", "created_at"]].itertuples(index=False)
        }
        self.attribute_values = {}
        for row in attributes[["attribute_id", "attribute_type", "value"]].itertuples(index=False):
            kind = str(row.attribute_type)
            if kind not in ATTRIBUTE_TYPES:
                raise ValueError(f"Unsupported attribute type: {kind}")
            self.attribute_values[str(row.attribute_id)] = (kind, normalize_attribute(kind, row.value))
        self._links = []
        for row in account_attributes[["account_id", "attribute_id", "observed_at"]].itertuples(index=False):
            account, attribute = str(row.account_id), str(row.attribute_id)
            if account not in self.account_created or attribute not in self.attribute_values:
                raise ValueError("Attribute link references an unknown entity")
            available = max(utc_time(row.observed_at), self.account_created[account])
            self._links.append((available, account, attribute))
        self._links.sort()
        self._link_times = [row[0] for row in self._links]
        self._metadata_size = -1
        self._metadata = None
        self._account_history = defaultdict(_Timeline)
        self._device_history = defaultdict(_Timeline)
        self._ip_history = defaultdict(_Timeline)
        self._events: dict[str, dict] = {}

    def _metadata_at(self, at: datetime):
        size = bisect_right(self._link_times, at)
        if size != self._metadata_size:
            account_keys, key_accounts = defaultdict(set), defaultdict(set)
            for _, account, attribute in self._links[:size]:
                key = self.attribute_values[attribute]
                if key[1]:  # Missing or blank values never establish links.
                    account_keys[account].add(key)
                    key_accounts[key].add(account)
            self._metadata = (account_keys, key_accounts)
            self._metadata_size = size
        return self._metadata

    def _validate_account(self, event: dict, at: datetime):
        account = event["account_id"]
        if account not in self.account_created:
            raise ValueError(f"Unknown account: {account}")
        if self.account_created[account] > at:
            raise ValueError("Account does not exist at event time")

    def features(self, event: dict) -> dict[str, float]:
        event = observable_event(event)
        at = utc_time(event["event_time"])
        self._validate_account(event, at)
        account = event["account_id"]
        account_events = self._account_history[account].before(at)
        device_events = self._device_history[event["device_id"]].before(at)
        ip_events = self._ip_history[event["ip_id"]].before(at)
        recent_start = at - timedelta(hours=24)
        recent = self._account_history[account].between(recent_start, at)
        hour_start = at - timedelta(hours=1)
        devices = {e["device_id"] for e in account_events}
        ips = {e["ip_id"] for e in account_events}
        device_accounts = {e["account_id"] for e in device_events} - {account}
        ip_accounts = {e["account_id"] for e in ip_events} - {account}
        account_keys, key_accounts = self._metadata_at(at)
        keys = account_keys.get(account, set())
        shared_accounts, shared_types = set(), set()
        for key in keys:
            others = key_accounts[key] - {account}
            if others:
                shared_accounts.update(others)
                shared_types.add(key[0])
        failures = sum(e["outcome"] == "failure" for e in account_events)
        home = primary_network(account_events)
        differing_contacts = 0
        for other in sorted(shared_accounts):
            shared = {k[0] for k in keys & account_keys.get(other, set())}
            other_home = primary_network(self._account_history[other].before(at))
            if len(shared) >= 2 and home is not None and other_home is not None and home != other_home:
                differing_contacts += 1
        values = {
            "account_age_hours": (at - self.account_created[account]).total_seconds() / 3600,
            "prior_login_count": len(account_events),
            "prior_failure_count": failures,
            "prior_success_rate": (len(account_events) - failures) / len(account_events) if account_events else 0,
            "login_count_1h": self._account_history[account].count_between(hour_start, at),
            "login_count_24h": len(recent),
            "failure_count_24h": sum(e["outcome"] == "failure" for e in recent),
            "hours_since_last_login": min(720, (at - utc_time(account_events[-1]["event_time"])).total_seconds() / 3600) if account_events else 720,
            "new_device": event["device_id"] not in devices,
            "new_ip": event["ip_id"] not in ips,
            "new_region": event["region"] not in {e["region"] for e in account_events},
            "new_browser": event["browser"] not in {e["browser"] for e in account_events},
            "password_auth": event["auth_method"] == "password",
            "device_quality_reset": event["device_quality"] == "reset",
            "binding_mismatch": event["session_binding"] == "mismatch",
            "binding_unverified": event["session_binding"] == "unverified",
            "binding_unknown": event["session_binding"] == "unknown",
            "challenge_failures": event["challenge_failures"],
            "recovery_used": event["recovery_used"],
            "account_device_count": len(devices),
            "account_ip_count": len(ips),
            "device_other_account_count": len(device_accounts),
            "ip_other_account_count": len(ip_accounts),
            "device_event_count_24h": self._device_history[event["device_id"]].count_between(recent_start, at),
            "ip_event_count_24h": self._ip_history[event["ip_id"]].count_between(recent_start, at),
            "shared_attribute_account_count": len(shared_accounts),
            "shared_attribute_type_count": len(shared_types),
            "attribute_count": len(keys),
            "missing_attribute_type_count": len(set(ATTRIBUTE_TYPES) - {key[0] for key in keys}),
            "device_account_overlap": len(device_accounts & shared_accounts),
            "ip_account_overlap": len(ip_accounts & shared_accounts),
            "shared_contact_different_home_count": differing_contacts,
        }
        return {name: float(values[name]) for name in BEHAVIOR_FEATURES + GRAPH_FEATURES}

    def update(self, event: dict) -> None:
        event = observable_event(event)
        at = utc_time(event["event_time"])
        self._validate_account(event, at)
        old = self._events.get(event["event_id"])
        if old is not None:
            if old != event:
                raise ValueError("Duplicate event_id has changed contents")
            return
        self._events[event["event_id"]] = event
        self._account_history[event["account_id"]].add(at, event)
        self._device_history[event["device_id"]].add(at, event)
        self._ip_history[event["ip_id"]].add(at, event)


def build_feature_frame(accounts, attributes, account_attributes, events) -> pd.DataFrame:
    state = HistoryState(accounts, attributes, account_attributes)
    observed = [observable_event(row) for row in events.to_dict("records")]
    observed.sort(key=lambda e: (e["received_at"], e["event_time"], e["event_id"]))
    if len({e["event_id"] for e in observed}) != len(observed):
        raise ValueError("Duplicate event IDs in the feature dataset")
    rows, index = [], 0
    while index < len(observed):
        end = index + 1
        while end < len(observed) and observed[end]["event_time"] == observed[index]["event_time"]:
            end += 1
        bucket = observed[index:end]
        for event in bucket:
            if event["outcome"] == "success":
                rows.append({**{key: event[key] for key in ("event_id", "account_id", "event_time")},
                             **state.features(event)})
        for event in bucket:
            state.update(event)
        index = end
    return pd.DataFrame(rows, columns=["event_id", "account_id", "event_time"] + BEHAVIOR_FEATURES + GRAPH_FEATURES)
