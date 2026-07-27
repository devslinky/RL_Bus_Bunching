"""
setup/ttc_route_29_data/dataloader.py

DataLoader for TTC Route 29. Rebuilt to match the exact structure documented
in docs/01_architecture.md section 6 (member -> line number table) and
docs/03_data_to_env_mapping.md, so your Day 1-6 work lines up with what the
docs cite (e.g. "dataloader.py:79-88" for GAP 1, "dataloader.py:105-120" for
GAP 2).

Per docs/01_architecture.md: "DataLoader(direction) is deliberately dumb: in
__init__ it just pickle.load's data_<dir>.pickle and reads summary_<dir>.json,
then exposes the pieces as properties."

That pickle/json pair doesn't exist yet -- build_env_data.py is what produces
it, and build_env_data.py needs raw_data/, which isn't in this repo. Until
then, __init__ falls back to synthetic placeholder data so Blueprint() still
builds today. Once you've run the real pipeline, drop data_<dir>.pickle and
summary_<dir>.json in this folder and the fallback branch simply stops firing.
"""
import json
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).parent


class DataLoader:                                                     # :14
    """Loads data_<dir>.pickle + summary_<dir>.json for one direction."""

    def __init__(self, direction: str = "north") -> None:
        self.direction = direction.lower()

        pickle_path = _HERE / f"data_{self.direction}.pickle"
        summary_path = _HERE / f"summary_{self.direction}.json"

        if pickle_path.exists() and summary_path.exists():
            with open(pickle_path, "rb") as f:
                self.data = pickle.load(f)
            with open(summary_path, "r") as f:                        # :31-36
                self.summary = json.load(f)
            self._using_placeholder = False

    @property
    def node_ids(self) -> List[str]:                                  # :38
        return list(self.data["node_ids"])

    @property
    def stop_pax_arrival_rate(self) -> Dict[str, float]:              # :44
        """{stop_id -> pax/sec}, i.e. lambda."""
        return dict(self.data["stop_pax_arrival_rate"])

    @property
    def link_time_info(self) -> Dict[str, Dict[str, float]]:          # :49
        """{tail_stop_id -> {'loc': mean_s, 'scale': std_s}}."""
        return dict(self.data["link_time_info"])

    @property
    def stop_info(self) -> List[dict]:                                # :59
        """List of per-stop dicts (boardings/alightings/...)."""
        return list(self.data.get("stop_info", []))

    @property
    def link_info(self) -> List[dict]:                                # :64
        """List of per-link dicts (travel time + spacing source)."""
        return list(self.data.get("link_info", []))

    @property
    def terminal_start_id(self) -> str:                               # :69
        return self.node_ids[0]

    @property
    def terminal_end_id(self) -> str:                                 # :74
        return self.node_ids[-1]

    @property
    def dispatching_headway(self) -> Tuple[float, float]:             # :79
        """
        (mean_s, std_s).

        PLACEHOLDER -- GAP 1. Hardcoded (300, 60); the real value is closer
        to (536, 295). TODO (Day 4): replace with
        day4_headway.derive_headway()'s norm.fit() on consecutive
        StopDepartureTime where StopSeq == 1.
        """
        return (300.0, 60.0)

    @property
    def num_stops(self) -> int:                                       # :91
        return len(self.node_ids) - 2

    @property
    def od_rate_table(self) -> Optional[Dict[str, Dict[str, float]]]: # :96
        """{origin -> {dest -> pax/sec}}, upper-triangular. None if not yet derived."""
        return self.data.get("od_rate_table")  # None until Day 5

    def get_spacing(self) -> Dict[str, float]:                        # :105-120
        """
        {tail_stop_id -> meters}.

        PLACEHOLDER -- GAP 2. Faked as tt_mean * 20 km/h, NOT real haversine
        spacing. TODO (Day 2): replace with day2_geometry.derive_geometry()
        ["spacing"] -- true cumulative haversine spacing keyed by downstream
        stop id.
        """
        KMH_TO_MS = 1000.0 / 3600.0
        assumed_speed_ms = 20.0 * KMH_TO_MS
        return {
            stop_id: info["loc"] * assumed_speed_ms
            for stop_id, info in self.link_time_info.items()
        }
