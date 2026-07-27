"""
setup/ttc_route_29_data/dataloader.py

PLACEHOLDER DataLoader for TTC Route 29 — the "shipped" baseline that
day6_integrate.py's `preview()` calls "OLD (placeholder)" and compares
against your Day 1-5 derived bundle.

This file intentionally ships with two known-wrong placeholders, called out
explicitly by name in the docs and in day6_integrate.py:

  * GAP 1 — dispatching_headway is hardcoded to (300.0, 60.0) seconds, not
    fit from real terminal-departure gaps (see docs/03, day4_headway.py).
  * GAP 2 — spacing is faked as tt_mean * (20 km/h), not real haversine
    distance between stops (see docs/02 section 9, day2_geometry.py).

Everything else here (node_ids, link_time_info, stop_pax_arrival_rate) is
ALSO a rough stand-in, not real TTC data — there's no raw_data/ bundle in
this repo to source it from. Treat the whole file as scaffolding: it exists
so `setup.blueprint` imports cleanly and `Blueprint("ttc_route_29_north")`
is buildable today, not because any of these numbers are correct yet.

Your Day 1-6 work should progressively replace each property below with one
that reads from your derived pickle bundle
(student_project/outputs/derived/{direction}/data_{direction}.pickle,
written by build_env_data.py) instead of the hardcoded values here.
"""
from typing import Dict, List, Optional, Tuple


# Placeholder route geometry: 2 terminals + 46 stops = 48 nodes, matching the
# verified `Blueprint("ttc_route_29_north")` node count from the docs.
# TODO (Day 1-2): replace with real TTC stop IDs for Route 29, in visiting
# order, pulled from GTFS stops.txt / trips.txt / stop_times.txt (or from
# ttc_avl_seg_clean_data.csv's stop-matching columns).
_PLACEHOLDER_NODE_IDS = [f"29_STOP_{i:02d}" for i in range(48)]

# Placeholder per-link travel time mean (seconds) / std (seconds).
# TODO (Day 3): replace with norm.fit() on real stop->stop travel times,
# i.e. consecutive StopArrivalTime within one TripID (see docs/02 section 5,
# day3_travel_time.py). Remember to filter on SteadyState and drop
# Abnormal*-flagged rows before fitting (docs/02 section 7.6).
_PLACEHOLDER_TT_MEAN_SEC = 50.0
_PLACEHOLDER_TT_STD_SEC = 10.0

# Placeholder dispatching headway (GAP 1). This is deliberately the same
# (300.0, 60.0) value day6_integrate.py's PLACEHOLDER_HEADWAY checks against.
# TODO (Day 4): replace with norm.fit() on consecutive StopDepartureTime
# where StopSeq == 1 (docs/02 section 6, day4_headway.py).
_PLACEHOLDER_HEADWAY_MEAN_SEC = 300.0
_PLACEHOLDER_HEADWAY_STD_SEC = 60.0

# Placeholder boarding rate (pax/sec) at every stop, flat.
# TODO (Day 5): replace with per-stop lambda_i = (mean Boarding per trip at
# stop i) / dispatching_headway_sec (docs/02 section 10, day5_demand_od.py).
_PLACEHOLDER_PAX_ARRIVAL_RATE = 0.02


class DataLoader:
    """
    Placeholder DataLoader for TTC Route 29.

    Supports both directions via the `direction` argument, mirroring how
    setup/ttc_route_29.py (TTC_Route_29_Network / TTC_Route_29_Route_Schema)
    constructs it: `DataLoader(direction)`.
    """

    def __init__(self, direction: str = "north") -> None:
        self.direction = direction.lower()
        if self.direction not in ("north", "south"):
            raise ValueError(f"direction must be 'north' or 'south', got {direction!r}")

    @property
    def node_ids(self) -> List[str]:
        """
        Ordered node ids: terminal -> stop -> stop -> ... -> stop -> terminal.

        NORTH and SOUTH should be reverses of each other once real stop
        sequences are wired in (Day 1-2); for now both directions share the
        same placeholder sequence.
        """
        if self.direction == "north":
            return list(_PLACEHOLDER_NODE_IDS)
        return list(reversed(_PLACEHOLDER_NODE_IDS))

    @property
    def link_time_info(self) -> Dict[str, Dict[str, float]]:
        """
        {tail_node_id -> {'loc': mean_travel_time_sec, 'scale': std_travel_time_sec}}

        Keyed by tail_node, matching how TTC_Route_29_Network._define_network
        looks this up: "the link arriving at this stop."
        """
        node_ids = self.node_ids
        return {
            node_id: {"loc": _PLACEHOLDER_TT_MEAN_SEC, "scale": _PLACEHOLDER_TT_STD_SEC}
            for node_id in node_ids[1:]  # no incoming link for the start terminal
        }

    def get_spacing(self) -> Dict[str, float]:
        """
        {tail_node_id -> spacing_meters}

        GAP 2 placeholder: spacing faked as tt_mean * 20 km/h, NOT real
        haversine distance between stop coordinates. Replace with cumulative
        haversine spacing along the canonical stop sequence (docs/02
        section 9, day2_geometry.py) once you have real lat/lon per stop.
        """
        KMH_TO_MS = 1000.0 / 3600.0
        assumed_speed_ms = 20.0 * KMH_TO_MS
        link_time_info = self.link_time_info
        return {
            node_id: info["loc"] * assumed_speed_ms
            for node_id, info in link_time_info.items()
        }

    @property
    def stop_pax_arrival_rate(self) -> Dict[str, float]:
        """{stop_id -> passengers/sec}, flat placeholder for every non-terminal stop."""
        node_ids = self.node_ids
        visit_seq_stop_ids = node_ids[1:-1]
        return {stop_id: _PLACEHOLDER_PAX_ARRIVAL_RATE for stop_id in visit_seq_stop_ids}

    @property
    def od_rate_table(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        {origin_stop -> {dest_stop -> pax/sec}}, or falsy to signal "not yet
        derived" so Route_Schema._define_od_table() falls back to a uniform
        downstream-distribution assumption.

        TODO (Day 5): replace with the real IPF-derived OD table divided by
        (num_trips * headway_sec) (docs/02 section 10, day5_demand_od.py).
        Returning None here is intentional for now -- it signals "not
        derived yet" to the fallback logic in Route_Schema.
        """
        return None

    @property
    def dispatching_headway(self) -> Tuple[float, float]:
        """
        (mean_headway_sec, std_headway_sec).

        GAP 1 placeholder. TODO (Day 4): replace with norm.fit() on
        consecutive StopDepartureTime where StopSeq == 1, per direction.
        """
        return (_PLACEHOLDER_HEADWAY_MEAN_SEC, _PLACEHOLDER_HEADWAY_STD_SEC)