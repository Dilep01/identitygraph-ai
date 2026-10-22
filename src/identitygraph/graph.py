"""Historical heterogeneous graphs and reviewable synthetic linkage evidence."""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from hashlib import sha256
from itertools import combinations
from math import log1p

from .features import HistoryState, iso_time, observable_event, primary_network, utc_time

NODE_TYPES = ("account", "device", "ip", "attribute")
NODE_FEATURE_DIM = 4
GRAPH_FEATURE_VERSION = "2.0.0"
EDGE_TYPES = (
    ("account", "used_device", "device"),
    ("account", "used_ip", "ip"),
    ("account", "has_attribute", "attribute"),
    ("device", "used_ip", "ip"),
)
MAX_GRAPH_NODES = 80
MAX_GRAPH_EDGES = 160


def _view(accounts, attributes, account_attributes, events, cutoff):
    """Select available metadata and strictly prior observable events."""
    at = utc_time(cutoff)
    state = HistoryState(accounts, attributes, account_attributes)
    account_ids = {key for key, created in state.account_created.items() if created <= at}
    links = sorted({(account, attribute) for available, account, attribute in state._links
                    if available <= at and account in account_ids and state.attribute_values[attribute][1]})
    attribute_ids = {attribute for _, attribute in links}
    selected_events = []
    seen = set()
    for raw in events.to_dict("records"):
        # Future events, including their malformed or hidden fields, are outside
        # this historical snapshot's information boundary.
        if utc_time(raw["event_time"]) >= at:
            continue
        event = observable_event(raw)
        if event["account_id"] not in account_ids:
            raise ValueError("Historical event references unavailable account")
        state._validate_account(event, utc_time(event["event_time"]))
        if event["event_id"] in seen:
            raise ValueError("Duplicate historical event_id")
        seen.add(event["event_id"])
        selected_events.append(event)
    selected_events.sort(key=lambda row: (row["event_time"], row["event_id"]))
    ids = {
        "account": sorted(account_ids),
        "device": sorted({e["device_id"] for e in selected_events}),
        "ip": sorted({e["ip_id"] for e in selected_events}),
        "attribute": sorted(attribute_ids),
    }
    edge_pairs = {
        EDGE_TYPES[0]: sorted({(e["account_id"], e["device_id"]) for e in selected_events}),
        EDGE_TYPES[1]: sorted({(e["account_id"], e["ip_id"]) for e in selected_events}),
        EDGE_TYPES[2]: links,
        EDGE_TYPES[3]: sorted({(e["device_id"], e["ip_id"]) for e in selected_events}),
    }
    return at, state, ids, selected_events, edge_pairs


def build_snapshot(accounts, attributes, account_attributes, events, cutoff: str):
    """Return PyG data and {node_type: {observable_id: tensor_row}} mappings.

    Only observed events strictly before cutoff contribute. Node features never
    include IDs, raw attribute strings, labels, or simulator group membership.
    """
    import torch
    from torch_geometric.data import HeteroData

    at, state, ids, historical, edge_pairs = _view(
        accounts, attributes, account_attributes, events, cutoff)
    mappings = {kind: {value: index for index, value in enumerate(values)}
                for kind, values in ids.items()}
    data = HeteroData()
    by_account, by_device, by_ip = defaultdict(list), defaultdict(list), defaultdict(list)
    for event in historical:
        by_account[event["account_id"]].append(event)
        by_device[event["device_id"]].append(event)
        by_ip[event["ip_id"]].append(event)
    attribute_accounts = defaultdict(set)
    for account, attribute in edge_pairs[EDGE_TYPES[2]]:
        attribute_accounts[attribute].add(account)

    def infrastructure_features(rows):
        age = (at - utc_time(rows[0]["event_time"])).total_seconds() / 86400 if rows else 0
        return [log1p(len(rows)), log1p(len({e["account_id"] for e in rows})),
                min(age / 30, 12), sum(e["outcome"] == "failure" for e in rows) / max(1, len(rows))]

    x = {kind: [] for kind in NODE_TYPES}
    for account in ids["account"]:
        rows = by_account[account]
        age = (at - state.account_created[account]).total_seconds() / 86400
        x["account"].append([
            min(age / 30, 12), log1p(sum(e["outcome"] == "success" for e in rows)),
            log1p(sum(e["outcome"] == "failure" for e in rows)),
            log1p(len({e["device_id"] for e in rows})),
        ])
    x["device"] = [infrastructure_features(by_device[key]) for key in ids["device"]]
    x["ip"] = [infrastructure_features(by_ip[key]) for key in ids["ip"]]
    for attribute in ids["attribute"]:
        kind, _ = state.attribute_values[attribute]
        x["attribute"].append([float(kind == "email"), float(kind == "phone"),
                               float(kind == "address"), log1p(len(attribute_accounts[attribute]))])
    for kind in NODE_TYPES:
        data[kind].x = torch.tensor(x[kind], dtype=torch.float32).reshape(-1, NODE_FEATURE_DIM)
        data[kind].num_nodes = len(ids[kind])
    for (source, relation, target), pairs in edge_pairs.items():
        indices = [[mappings[source][left], mappings[target][right]] for left, right in pairs]
        edge_index = torch.tensor(indices, dtype=torch.long).reshape(-1, 2).t().contiguous()
        data[(source, relation, target)].edge_index = edge_index
        data[(target, "rev_" + relation, source)].edge_index = edge_index.flip(0).contiguous()
    return data, mappings


def _similarity(kind, left, right):
    # A shared email domain is weak evidence and must not inflate similarity.
    if kind == "email":
        if "@" not in left or "@" not in right:
            return 0.0
        left_local, left_domain = left.rsplit("@", 1)
        right_local, right_domain = right.rsplit("@", 1)
        if left_domain != right_domain:
            return 0.0
        left, right = left_local, right_local
    # Phone numbers are exact-only: neighboring fictional numbers are common.
    if kind == "phone" or min(len(left), len(right)) < 6:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def broad_candidate_clusters(accounts, attributes, account_attributes, events, cutoff: str, fuzzy: bool = True):
    """Return review candidates with a 0-100 heuristic score, not probability.

    Qualifying pairs have >=2 attribute types, an attribute plus a device,
    or >=2 devices. IP sharing never qualifies a pair. Similarities are
    candidate links only; no canonical accounts or identities are merged.
    """
    _, state, ids, historical, edge_pairs = _view(
        accounts, attributes, account_attributes, events, cutoff)
    value_accounts, value_ids = defaultdict(set), defaultdict(set)
    for account, attribute in edge_pairs[EDGE_TYPES[2]]:
        key = state.attribute_values[attribute]
        value_accounts[key].add(account)
        value_ids[key].add(attribute)
    pair_evidence = defaultdict(list)
    for key, owners in sorted(value_accounts.items()):
        kind, _ = key
        for pair in combinations(sorted(owners), 2):
            pair_evidence[pair].append({
                "type": "exact_attribute", "attribute_type": kind,
                "attribute_ids": sorted(value_ids[key]),
                "description": f"Same normalized fictional {kind}",
            })
    # Blocking bounds fuzzy comparisons and deliberately accepts missed links.
    # The first three characters must agree; this is documented as a limitation.
    blocks = defaultdict(list)
    for key in sorted(value_accounts):
        kind, value = key
        if kind in ("email", "address"):
            blocks[(kind, value[:3])].append(key)
    for keys in blocks.values():
        for left, right in combinations(keys, 2):
            similarity = _similarity(left[0], left[1], right[1])
            if not fuzzy or similarity < 0.92:
                continue
            for a in sorted(value_accounts[left]):
                for b in sorted(value_accounts[right]):
                    if a == b:
                        continue
                    pair_evidence[tuple(sorted((a, b)))].append({
                        "type": "similar_attribute", "attribute_type": left[0],
                        "attribute_ids": sorted(value_ids[left] | value_ids[right]),
                        "similarity": round(similarity, 4),
                        "description": f"Similar fictional {left[0]} strings; unconfirmed link",
                    })
    device_accounts = defaultdict(set)
    for event in historical:
        device_accounts[event["device_id"]].add(event["account_id"])
    for device, owners in sorted(device_accounts.items()):
        for pair in combinations(sorted(owners), 2):
            pair_evidence[pair].append({"type": "shared_device", "device_id": device,
                                       "description": "Historical use of the same device"})

    qualifying, adjacency = {}, defaultdict(set)
    for pair, evidence in sorted(pair_evidence.items()):
        attribute_types = {e["attribute_type"] for e in evidence if "attribute_type" in e}
        devices = {e["device_id"] for e in evidence if "device_id" in e}
        if len(attribute_types) >= 2 or (attribute_types and devices) or len(devices) >= 2:
            qualifying[pair] = evidence
            adjacency[pair[0]].add(pair[1])
            adjacency[pair[1]].add(pair[0])
    clusters, visited = [], set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        members, pending = set(), [start]
        while pending:
            account = pending.pop()
            if account in members:
                continue
            members.add(account)
            visited.add(account)
            pending.extend(adjacency[account] - members)
        evidence, pair_scores = [], []
        for pair, items in qualifying.items():
            if pair[0] not in members or pair[1] not in members:
                continue
            for item in items:
                evidence.append({"account_ids": list(pair), **item})
            exact = {e["attribute_type"] for e in items if e["type"] == "exact_attribute"}
            fuzzy = {e["attribute_type"] for e in items if e["type"] == "similar_attribute"} - exact
            device_count = len({e["device_id"] for e in items if e["type"] == "shared_device"})
            pair_scores.append(min(100, 25 * len(exact) + 15 * len(fuzzy) + 20 * min(2, device_count)))
        member_ids = sorted(members)
        digest = sha256("\x00".join(member_ids).encode()).hexdigest()[:12]
        clusters.append({"cluster_id": f"cluster-{digest}", "account_ids": member_ids,
                         "evidence": evidence, "score": float(max(pair_scores)),
                         "score_type": "heuristic_0_100", "status": "candidate_for_review",
                         "history_cutoff": iso_time(cutoff)})
    return sorted(clusters, key=lambda row: (-row["score"], row["cluster_id"]))


def candidate_clusters(accounts, attributes, account_attributes, events, cutoff: str, fuzzy: bool = True):
    """Corroborated candidates with complete pair support, never transitive closure.

    Require an exact shared email plus another attribute type and a device link, plus different established
    primary networks. This can miss sparse groups and still flag relatives who
    live separately. Thresholds are fixed in the development contract.
    """
    broad = broad_candidate_clusters(accounts, attributes, account_attributes, events, cutoff, fuzzy)
    _, _, _, historical, _ = _view(accounts, attributes, account_attributes, events, cutoff)
    histories = defaultdict(list)
    for event in historical:
        histories[event["account_id"]].append(event)
    homes = {key: primary_network(value) for key, value in histories.items()}
    pairs = defaultdict(list)
    for cluster in broad:
        for evidence in cluster["evidence"]:
            pairs[tuple(evidence["account_ids"])].append(evidence)
    qualified, adjacency = {}, defaultdict(set)
    for pair, evidence in sorted(pairs.items()):
        kinds = {e["attribute_type"] for e in evidence if "attribute_type" in e}
        exact_types = {e["attribute_type"] for e in evidence if e["type"] == "exact_attribute"}
        devices = {e["device_id"] for e in evidence if "device_id" in e}
        left, right = pair
        if "email" not in exact_types or len(kinds) < 2 or not devices or not homes.get(left) or not homes.get(right) or homes[left] == homes[right]:
            continue
        qualified[pair] = evidence + [{"account_ids": list(pair), "type": "different_primary_networks",
                                      "description": "Shared contacts and device across different established networks",
                                      "ip_ids": [homes[left], homes[right]]}]
        adjacency[left].add(right)
        adjacency[right].add(left)
    # Deterministic complete-link covers: an A-B-C bridge cannot imply A-C evidence.
    groups = set()
    for pair in qualified:
        members = set(pair)
        possible = adjacency[pair[0]] & adjacency[pair[1]]
        for node in sorted(possible):
            if all(node in adjacency[member] for member in members):
                members.add(node)
        groups.add(tuple(sorted(members)))
    groups = [g for g in sorted(groups) if not any(set(g) < set(h) for h in groups)]
    result = []
    for members in groups:
        evidence = [e for pair in combinations(members, 2) for e in qualified[pair]]
        result.append({"cluster_id": "corroborated-" + sha256("\x00".join(members).encode()).hexdigest()[:12],
                       "account_ids": list(members), "evidence": evidence, "score": 80.0,
                       "score_type": "heuristic_0_100", "status": "candidate_for_review",
                       "history_cutoff": iso_time(cutoff), "policy": "corroborated_complete_link_v2",
                       "warning": "Shared ownership is unverified; relatives living separately can qualify."})
    return result


def canonical_attribute_mapping(attributes):
    """Alias map for normalized observed attributes; accounts are never merged."""
    from .features import normalize_attribute
    aliases = []
    for row in attributes.sort_values("attribute_id").itertuples():
        value = normalize_attribute(row.attribute_type, row.value)
        canonical = sha256((row.attribute_type + "\x00" + value).encode()).hexdigest()[:20]
        aliases.append({"attribute_id": row.attribute_id, "canonical_attribute_id": canonical if value else None,
                        "attribute_type": row.attribute_type, "normalized_value": value,
                        "status": "normalized_attribute_alias" if value else "missing"})
    return aliases


def account_graph(account_id, accounts, attributes, account_attributes, events, cutoff: str):
    """Return a deterministic two-hop view, capped at 80 nodes and 160 edges."""
    _, state, ids, _, edge_pairs = _view(accounts, attributes, account_attributes, events, cutoff)
    account_id = str(account_id)
    if account_id not in ids["account"]:
        raise ValueError(f"Unknown account at cutoff: {account_id}")
    # This explorer displays observed relationships, including weak IP evidence.
    # Candidate creation applies stricter rules in candidate_clusters().
    all_edges, adjacency = [], defaultdict(set)
    for (source, relation, target), pairs in edge_pairs.items():
        for left, right in pairs:
            source_id, target_id = f"{source}:{left}", f"{target}:{right}"
            all_edges.append({"source": source_id, "target": target_id, "type": relation})
            adjacency[source_id].add(target_id)
            adjacency[target_id].add(source_id)
    root = f"account:{account_id}"
    chosen, frontier = [root], [root]
    seen = {root}
    for _ in range(2):
        next_frontier = []
        for node in frontier:
            for neighbor in sorted(adjacency[node]):
                if neighbor in seen:
                    continue
                if len(chosen) >= MAX_GRAPH_NODES:
                    break
                seen.add(neighbor)
                chosen.append(neighbor)
                next_frontier.append(neighbor)
        frontier = next_frontier
    edges = [edge for edge in all_edges if edge["source"] in seen and edge["target"] in seen]
    edges.sort(key=lambda edge: (root not in (edge["source"], edge["target"]),
                                 edge["source"], edge["target"], edge["type"]))
    edges = edges[:MAX_GRAPH_EDGES]
    connected = {root} | {edge[key] for edge in edges for key in ("source", "target")}
    nodes = []
    for identifier in chosen:
        if identifier not in connected:
            continue
        kind, raw_id = identifier.split(":", 1)
        node = {"id": identifier, "type": kind, "entity_id": raw_id, "label": raw_id}
        if kind == "attribute":
            attribute_type, value = state.attribute_values[raw_id]
            node.update(attribute_type=attribute_type, label=f"{attribute_type}: {value}")
        nodes.append(node)
    return {"nodes": nodes, "edges": edges}
