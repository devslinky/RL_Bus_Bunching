"""
test_derived_data.py — validate the *derived* Route 29 environment data.

These tests run the real ETL pipeline you (the student) build in the
`scripts/` folder and check that its output honours the DataLoader contract
described in docs/03_data_to_env_mapping.md.  They are the tests that prove
GAP 1 (headway) and the OD / spacing / travel-time derivations are sane.

WHAT THEY EXERCISE
------------------
* The heavy lifting (reading ~400k rows of APC data and building the whole
  NORTH environment dict) happens once, inside the ``derived_north`` fixture
  defined in ``conftest.py``.  Every invariant test below simply consumes that
  cached dict, so the suite stays fast even though the data is real.
* One extra test calls ``derive_travel_time`` and ``derive_headway`` directly
  to confirm the two building blocks behave on their own.

HOW TO RUN
----------
    # from the repo root:  /home/jiahao/Documents/busoperation
    python -m pytest student_project/tests/test_derived_data.py -v

These tests read the real CSVs, so expect them to take a few seconds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Make the day-scripts importable even when pytest is invoked from elsewhere.
# conftest.py normally does this too, but adding it here keeps this file
# runnable on its own (e.g. `python -m pytest <this file>`).
# --------------------------------------------------------------------------- #
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# --------------------------------------------------------------------------- #
# Invariants on the assembled NORTH environment dict (derive_all("NORTH")).
# All of these consume the shared ``derived_north`` fixture from conftest.py.
# --------------------------------------------------------------------------- #
def test_node_ids_is_nonempty_str_list(derived_north):
    """node_ids must be a non-empty, duplicate-free list of *string* stop ids."""
    node_ids = derived_north["node_ids"]
    assert isinstance(node_ids, list), "node_ids should be a list"
    assert len(node_ids) > 0, "node_ids must not be empty"
    assert all(isinstance(n, str) for n in node_ids), "every node id must be a str"
    assert len(set(node_ids)) == len(node_ids), "node_ids contains duplicates"


def test_terminals_are_the_endpoints(derived_north):
    """terminal_start/end must be the first / last entries of node_ids."""
    d = derived_north
    node_ids = d["node_ids"]
    assert d["terminal_start_id"] == node_ids[0], (
        f"terminal_start_id {d['terminal_start_id']!r} != first node {node_ids[0]!r}"
    )
    assert d["terminal_end_id"] == node_ids[-1], (
        f"terminal_end_id {d['terminal_end_id']!r} != last node {node_ids[-1]!r}"
    )


def test_spacing_positive_and_finite(derived_north):
    """Every inter-stop spacing must be a positive, finite distance in metres.

    This is the whole point of GAP 2: real haversine spacing, not the old
    ``tt_mean * 20 km/h`` placeholder — so no zeros and no NaNs.
    """
    d = derived_north
    spacing = d["spacing"]
    node_ids = set(d["node_ids"])
    assert len(spacing) > 0, "spacing dict is empty"
    for tail_id, meters in spacing.items():
        assert tail_id in node_ids, f"spacing keyed by unknown stop {tail_id!r}"
        assert math.isfinite(meters), f"spacing[{tail_id}] is not finite: {meters}"
        assert meters > 0, f"spacing[{tail_id}] must be > 0, got {meters}"


def test_link_time_info_params_valid(derived_north):
    """Each link's travel-time distribution needs loc > 0 and scale >= 0."""
    lti = derived_north["link_time_info"]
    assert len(lti) > 0, "link_time_info is empty"
    for tail_id, params in lti.items():
        assert "loc" in params and "scale" in params, (
            f"link_time_info[{tail_id}] missing loc/scale: {params}"
        )
        loc = params["loc"]
        scale = params["scale"]
        assert math.isfinite(loc) and loc > 0, (
            f"link_time_info[{tail_id}] loc must be > 0, got {loc}"
        )
        assert math.isfinite(scale) and scale >= 0, (
            f"link_time_info[{tail_id}] scale must be >= 0, got {scale}"
        )


def test_stop_pax_arrival_rate_nonneg_finite(derived_north):
    """Every per-stop lambda (pax/sec) must be finite and non-negative."""
    lam = derived_north["stop_pax_arrival_rate"]
    assert len(lam) > 0, "stop_pax_arrival_rate is empty"
    for stop_id, rate in lam.items():
        assert math.isfinite(rate), f"lambda[{stop_id}] is not finite: {rate}"
        assert rate >= 0, f"lambda[{stop_id}] must be >= 0, got {rate}"


def test_od_rate_table_downstream_only_and_nonneg(derived_north):
    """The OD rate table must be non-negative and strictly downstream-only.

    "Downstream-only" means an entry origin -> dest is allowed only when dest
    comes strictly *after* origin in the canonical stop order (upper-triangular
    in node_ids order).  No self loops, no upstream flows.
    """
    d = derived_north
    node_ids = d["node_ids"]
    pos = {sid: i for i, sid in enumerate(node_ids)}
    od = d["od_rate_table"]
    assert len(od) > 0, "od_rate_table is empty"
    for origin, dests in od.items():
        assert origin in pos, f"OD origin {origin!r} not in node_ids"
        for dest, rate in dests.items():
            assert dest in pos, f"OD dest {dest!r} not in node_ids"
            assert math.isfinite(rate), f"OD[{origin}->{dest}] not finite: {rate}"
            assert rate >= 0, f"OD[{origin}->{dest}] must be >= 0, got {rate}"
            assert pos[dest] > pos[origin], (
                f"OD[{origin}->{dest}] is not strictly downstream "
                f"(origin idx {pos[origin]} >= dest idx {pos[dest]})"
            )


def test_od_row_sums_match_lambda(derived_north):
    """Each origin's OD row-sum must approximate that stop's lambda.

    The IPF that builds the OD table distributes each stop's boarding rate
    (lambda_i) across its downstream destinations, so the row-sum should equal
    lambda_i.  The final (terminal_end) stop has no downstream stop, so its
    demand cannot be placed in an upper-triangular matrix; we skip it.

    Tolerance is deliberately modest (a few percent): the IPF rescales the
    boarding / alighting margins so their totals agree, which can nudge a row
    away from the raw lambda by the (small) route-level boarding/alighting
    imbalance.
    """
    d = derived_north
    node_ids = d["node_ids"]
    od = d["od_rate_table"]
    lam = d["stop_pax_arrival_rate"]

    # Skip the last stop: it is the terminal and has no downstream destination.
    for origin in node_ids[:-1]:
        row_sum = sum(od.get(origin, {}).values())
        expected = lam.get(origin, 0.0)
        assert np.isclose(row_sum, expected, rtol=5e-2, atol=1e-6), (
            f"OD row-sum for origin {origin!r} = {row_sum} does not match "
            f"lambda = {expected}"
        )


def test_dispatching_headway_is_real_not_placeholder(derived_north):
    """The measured dispatching headway must be meaningfully above 300 s.

    GAP 1: the old DataLoader hard-coded (300, 60).  The real NORTH headway
    measured from APC terminal departures is ~536 s mean, so a correct
    pipeline lands well inside (300, 900).
    """
    mean_sec, std_sec = derived_north["dispatching_headway"]
    assert math.isfinite(mean_sec) and math.isfinite(std_sec)
    assert 300 < mean_sec < 900, (
        f"dispatching headway mean {mean_sec}s should be in (300, 900) — "
        f"clearly larger than the 300s placeholder"
    )
    assert std_sec >= 0, f"headway std must be >= 0, got {std_sec}"


# --------------------------------------------------------------------------- #
# Direct checks on the two building-block derivations (no fixture needed).
# These re-read the APC data, so this test takes a few seconds on its own.
# The day modules are imported lazily so a problem in one of them cannot break
# collection of the fixture-based tests above.
# --------------------------------------------------------------------------- #
def test_travel_time_and_headway_building_blocks():
    """derive_travel_time gives positive tt_mean where data exists; headway>300."""
    import day3_travel_time
    import day4_headway

    tt = day3_travel_time.derive_travel_time("NORTH")
    expected_cols = {"from_stop_id", "to_stop_id", "tt_mean", "tt_std", "tt_cv", "tt_count"}
    assert expected_cols.issubset(tt.columns), (
        f"derive_travel_time missing columns: {expected_cols - set(tt.columns)}"
    )

    observed = tt[tt["tt_count"] > 0]
    assert len(observed) > 0, "no links had any observed travel times"
    assert (observed["tt_mean"] > 0).all(), (
        "every link with tt_count > 0 must have a positive tt_mean"
    )

    hw = day4_headway.derive_headway("NORTH")
    assert "overall" in hw, "derive_headway result missing 'overall' key"
    overall_mean = hw["overall"][0]
    assert overall_mean > 300, (
        f"real NORTH headway mean {overall_mean}s should exceed the 300s placeholder"
    )
