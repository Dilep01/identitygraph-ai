import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from identitygraph.simulator import SimulationConfig, generate, load_observed


def test_reproducible_separate_truth_and_unseen_family(tmp_path):
    cfg = SimulationConfig(accounts=60, daily_sessions=0.6, takeover_rate=0.2)
    one, two = tmp_path / "one", tmp_path / "two"
    m1, m2 = generate(one, cfg), generate(two, cfg)
    assert m1 == m2
    obs = load_observed(one)
    forbidden = {"is_takeover", "is_synthetic", "attack_group", "scenario", "owner_group", "cohort"}
    for df in obs.values():
        assert not forbidden.intersection(df.columns)
    events = obs["events"]
    assert events.event_id.is_unique
    assert events.event_time.is_monotonic_increasing
    assert set(events.account_id) <= set(obs["accounts"].account_id)
    truth = pd.read_parquet(one / "truth/events.parquet")
    joined = events.merge(truth, on="event_id", validate="one_to_one")
    assert not ((joined.outcome == "failure") & (joined.is_takeover == 1)).any()
    unseen = joined[joined.scenario == "familiar_session"]
    assert len(unseen) > 0
    assert (unseen.event_time >= m1["replay_cutoff"]).all()
    assert all("@example.test" in v for v in obs["attributes"].query("attribute_type == 'email'").value)
    assert json.loads((one / "manifest.json").read_text())["provenance"] == "fully_synthetic"


def test_preserve_existing_dataset_and_reject_bad_config(tmp_path):
    generate(tmp_path / "run", SimulationConfig(accounts=40))
    with pytest.raises(FileExistsError):
        generate(tmp_path / "run", SimulationConfig(accounts=40))
    with pytest.raises(ValueError):
        generate(tmp_path / "bad", SimulationConfig(accounts=3))


def test_reproducible_across_python_hash_seeds(tmp_path):
    for seed in ("1", "999"):
        output = tmp_path / seed
        code = ("from pathlib import Path; from identitygraph.simulator import generate, SimulationConfig; "
                "import sys; generate(Path(sys.argv[1]), SimulationConfig(accounts=40, daily_sessions=0.2))")
        subprocess.run([sys.executable, "-c", code, str(output)],
                       env={**os.environ, "PYTHONHASHSEED": seed}, check=True, capture_output=True, timeout=120)
    manifests = [json.loads((tmp_path / seed / "manifest.json").read_text()) for seed in ("1", "999")]
    assert manifests[0] == manifests[1]


def test_new_phone_never_affects_an_earlier_session(tmp_path):
    run = tmp_path / "chronology"
    generate(run, SimulationConfig(accounts=40, daily_sessions=4, churn_rate=0.7, takeover_rate=0.0))
    events = pd.read_parquet(run / "observed/events.parquet")
    truth = pd.read_parquet(run / "truth/events.parquet")
    joined = events.merge(truth[["event_id", "scenario"]], on="event_id")
    first_seen = joined.groupby(["account_id", "device_id"]).event_time.min()
    new_phones = joined[joined.scenario == "new_phone"]
    assert len(new_phones) > 100
    for row in new_phones.itertuples():
        assert first_seen.loc[(row.account_id, row.device_id)] == row.event_time
