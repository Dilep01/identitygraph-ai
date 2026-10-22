# Historical features and identity graph

This component uses fictional observable records only. It never imports the
simulator, truth tables, attack groups, or label columns. There is no identity
verification claim and no automatic account merge or account blocking.

## Information boundary

`features.py` exports `FEATURE_VERSION = "2.0.0"`, `BEHAVIOR_FEATURES`, and
`GRAPH_FEATURES`. Only those 32 numeric feature names are model inputs (19 behavior,
13 graph). Event,
account, device, and network identifiers remain lookup keys. Raw attribute
strings do not become model inputs.

All timestamps must have an explicit timezone and are normalized to UTC. This
version requires receipt time to equal event time. For a query at time `t`,
historical events satisfy `event_time < t`; account creation and attribute-link
availability satisfy `available_at <= t`. Attribute availability is the later
of account creation and the link's `observed_at` time. Static attribute values
are immutable in this schema; a correction needs a new attribute and link.

`build_feature_frame` scores all successful sessions in one timestamp bucket
before inserting any event in that bucket. Failures become history but are not
prediction rows. `HistoryState.features` also excludes same-time and future
events if the caller has already inserted them. This guards replay ordering
without permitting late events through the serving API. The current outcome
does not become a numeric feature. Its success value determines eligibility for
the successful-session model. The current auth method, region, browser, device,
and IP are query context; novelty and counts use prior history only.

The behavioral inputs cover account age, prior session/failure counts, prior
success rate, one-hour and 24-hour activity, time since the last session, device,
IP, region and browser novelty, auth method, and device quality. Missing history
uses zero counts/rates and 720 hours since last login; elapsed time is also
capped at 720 hours. Optional current binding, challenge and recovery observations
add five numeric inputs. These are noisy observable fields, not an actor-identity oracle.

Graph inputs cover account device/IP degrees, other accounts previously using
the query device/IP, 24-hour device/IP activity, exact shared-attribute contacts,
shared attribute types, attribute completeness, and overlap between attribute
contacts and device/IP contacts. Every set/count is derived from available
observable data. A further feature counts shared contacts with different established
primary networks. A primary network requires at least three earlier successful sessions
and at least 40% on that network. Attributes with empty values never create links.

## Heterogeneous snapshot

`build_snapshot(..., cutoff)` returns `(HeteroData, mappings)`, where each mapping
is `{observable_entity_id: tensor_row}` under `account`, `device`, `ip`, and
`attribute`. All four node stores and all eight relation stores exist even when
some contain no nodes or edges. Edges are deduplicated. Relation types are:

- account → used_device → device and its reverse;
- account → used_ip → IP and its reverse;
- account → has_attribute → attribute and its reverse;
- device → used_ip → IP and its reverse.

Each node has four finite numeric values:

| Node | Features |
| --- | --- |
| Account | Age in 30-day units, capped at 12; log1p prior successes; log1p prior failures; log1p distinct devices |
| Device | log1p events; log1p accounts; age since first event in 30-day units, capped at 12; failure fraction |
| IP | Same definitions as device |
| Attribute | Email, phone and address indicator values; log1p linked accounts |

The snapshot uses all strictly earlier events. This is suitable for the bounded
local dataset, but is not an unbounded streaming storage design. Its historical
interaction edges summarize repeated sessions; there are no per-session nodes.
Unknown query devices and IPs are absent from a historical snapshot. The model
must handle these through its documented fallback, without constructing future
edges. A coarser snapshot used by a GNN experiment must report its earlier
cutoff; this builder never silently rounds a cutoff.

## Candidate linkage

Exact string normalization trims whitespace and ignores case. Digit-based phone
fixtures additionally ignore punctuation. Opaque `phone-token-...` fictional
identifiers retain their letters and full normalized value. An exact shared
attribute may have multiple source attribute IDs; candidate evidence records
them without merging accounts.

Fuzzy comparison applies only to fictional email and address strings. Email
domains must agree, and similarity is computed on local parts so a common
domain cannot inflate the score. Phone tokens use exact comparison only. Fuzzy
comparisons require at least six characters, share the first three characters
as a blocking rule, and use `SequenceMatcher` similarity of at least 0.92. A
typo in the blocking prefix can therefore be missed. Similarity is evidence for
review, not evidence that a person or attribute has been verified.

The v1 candidate policy requires an exact shared email, another shared attribute
type, a historical shared device, and different established primary networks.
The deterministic complete-link grouping requires direct support for every pair
inside each displayed group. A bridge between A-B and B-C cannot by itself join
A and C. Evidence retains exact/similar attributes, devices and primary networks.

Qualifying candidates receive a fixed heuristic score of 80, not a probability
or a meaningful ranking among qualifying groups. Stable group IDs hash member
IDs. Canonical attribute aliases hash normalized type/value pairs; they merge
equivalent attribute keys for analysis, never account identities.

`broad_candidate_clusters` preserves the older connected-component policy for
comparison: two attribute types, an attribute plus a device, or two devices.
It can join accounts without direct pair support and has many false positives.
The API uses the stricter policy. Genuine relatives living separately can still
qualify; sparsely linked fabricated identities can be missed. Every true group
is included in the coverage denominator. These algorithms are for a bounded
local research dataset, not large-scale entity resolution.

## Investigator graph

`account_graph` returns at most 80 nodes and 160 edges from a deterministic
two-hop neighborhood. It includes observed IP relationships because those can
help an investigation, even though IP-only sharing cannot create a candidate.
Node IDs are namespaced (`account:...`, `device:...`, `ip:...`, `attribute:...`).
Every edge endpoint is present in the response. The view is a capped excerpt;
it must not be presented as a complete account neighborhood.

## Focused verification

`tests/test_features_graph.py` checks strict history and metadata cutoffs,
equal-time buckets, future/label invariance, duplicate handling, cold history,
opaque phone token preservation, empty graph shape, reverse relations,
legacy candidate rules, separate fuzzy evidence, and bounded explorer endpoints.
`tests/test_completion.py` checks the stricter policy, legitimate household
exclusion, direct pair support, aliases, optional telemetry and hidden/future-data invariance.
Test execution results are recorded separately from this design document.
