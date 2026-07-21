"""
test_integration.py — is your derived bundle READY to drive the simulator? (torch-free)

This does NOT wire the env for you (that is your Day-6 capstone). It checks that the
bundle `build_env_data.derive_all()` produces carries everything the TTC env's DataLoader
contract needs, with sane, real values — so that whatever integration YOU write
(docs/06_integrate_into_simulator.md) has correct data underneath it.

It imports only build_env_data (torch-free); no simulator/agent import, so it stays green
everywhere. The actual run is exercised by `day6_integrate.py --run-sim` / `quick_run.py`.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_env_data  # noqa: E402

PLACEHOLDER_HEADWAY = 300.0

# The keys the TTC env pulls off the DataLoader (see setup/ttc_route_29_data/dataloader.py
# + setup/ttc_route_29.py). A student's real DataLoader reads these off the bundle.
REQUIRED_KEYS = {
    "node_ids", "terminal_start_id", "terminal_end_id",
    "stop_pax_arrival_rate", "link_time_info", "od_rate_table",
    "spacing", "dispatching_headway", "link_info", "stop_info",
}


@pytest.fixture(scope="module")
def bundle():
    return build_env_data.derive_all("NORTH")


def test_bundle_has_dataloader_contract_keys(bundle):
    missing = REQUIRED_KEYS - set(bundle)
    assert not missing, f"derived bundle is missing DataLoader keys: {missing}"


def test_node_ids_and_terminals(bundle):
    node_ids = bundle["node_ids"]
    assert isinstance(node_ids, list) and node_ids
    assert all(isinstance(n, str) for n in node_ids)
    assert bundle["terminal_start_id"] == node_ids[0]
    assert bundle["terminal_end_id"] == node_ids[-1]


def test_headway_is_real_not_placeholder(bundle):
    mean, std = bundle["dispatching_headway"]
    # GAP 1: the real headway must have replaced the (300, 60) placeholder.
    assert mean > PLACEHOLDER_HEADWAY, f"headway {mean} still looks like the placeholder"
    assert std >= 0


def test_spacing_real_positive_and_keyed_by_stops(bundle):
    spacing = bundle["spacing"]
    node_ids = set(bundle["node_ids"])
    assert spacing, "no spacing in bundle"
    assert all(v > 0 and v < 20000 for v in spacing.values()), "spacing not real/positive"
    # spacing is keyed by the downstream stop of each link -> a subset of node_ids
    assert set(map(str, spacing)) <= node_ids
    total_km = sum(spacing.values()) / 1000.0
    assert 3.0 < total_km < 30.0, f"route length {total_km:.1f} km is implausible"


def test_link_time_info_sane(bundle):
    for tail, params in bundle["link_time_info"].items():
        assert params["loc"] > 0, f"non-positive travel time at link -> {tail}"
        assert params["scale"] >= 0


def test_demand_and_od_nonneg_downstream(bundle):
    assert all(v >= 0 for v in bundle["stop_pax_arrival_rate"].values())
    order = {n: i for i, n in enumerate(bundle["node_ids"])}
    for o, dests in bundle["od_rate_table"].items():
        for d, rate in dests.items():
            assert rate >= 0
            # passengers only travel downstream
            if rate > 0:
                assert order[str(d)] > order[str(o)], f"upstream OD flow {o}->{d}"
