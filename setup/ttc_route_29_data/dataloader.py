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
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).parent # data should be in the ttc_route_29_data folder alongside this dataloader.py


class DataLoader:                                                     
    """Loads data_<dir>.pickle + summary_<dir>.json for one direction."""

    def __init__(self, direction: str = "north") -> None:
        self.direction = direction.lower()

        pickle_path = _HERE / f"data_{self.direction}.pickle"
        summary_path = _HERE / f"summary_{self.direction}.json"

        if pickle_path.exists() and summary_path.exists():
            with open(pickle_path, "rb") as f:
                self.data = pickle.load(f)
            with open(summary_path, "r") as f:                        
                self.summary = json.load(f)
            self._using_placeholder = False

    @property
    def node_ids(self) -> List[str]:                                 
        return list(self.data["node_ids"])

    @property
    def stop_pax_arrival_rate(self) -> Dict[str, float]:              
        """{stop_id -> pax/sec}, i.e. lambda."""
        return dict(self.data["stop_pax_arrival_rate"])

    @property
    def link_time_info(self) -> Dict[str, Dict[str, float]]:          
        """{tail_stop_id -> {'loc': mean_s, 'scale': std_s}}."""
        return dict(self.data["link_time_info"])

    @property
    def stop_info(self) -> List[dict]:                                
        """List of per-stop dicts (boardings/alightings/...)."""
        return list(self.data.get("stop_info", []))

    @property
    def link_info(self) -> List[dict]:                                
        """List of per-link dicts (travel time + spacing source)."""
        return list(self.data.get("link_info", []))

    @property
    def terminal_start_id(self) -> str:                             
        return self.node_ids[0]

    @property
    def terminal_end_id(self) -> str:                                
        return self.node_ids[-1]

    @property
    def dispatching_headway(self) -> Tuple[float, float]:             
        """
        (mean_s, std_s).

        currently returns 300,60 
        commented out is a better alternative with headway derived from real ttc data (see docs/03_data_to_env_mapping.md) (see ttc_route_29_data_real/dataloader.py)
        """
        return (300.0,60.0)
        # return tuple(self.data["dispatching_headway"])

    @property
    def num_stops(self) -> int:                                      
        return len(self.node_ids) - 2

    @property
    def od_rate_table(self) -> Optional[Dict[str, Dict[str, float]]]: 
        """{origin -> {dest -> pax/sec}}, upper-triangular. None if not yet derived."""
        return self.data.get("od_rate_table") 

    def get_spacing(self) -> Dict[str, float]:                        
        """
        {tail_stop_id -> meters}.
        currently returns tt_mean * 20 km/h
        commented out is a better alternative that returns real haversine spacing keyed by downstream stop id. (see ttc_route_29_data_real/dataloader.py)
        """
        KMH_TO_MS = 1000.0 / 3600.0
        assumed_speed_ms = 20.0 * KMH_TO_MS

        return {
            stop_id: info["loc"] * assumed_speed_ms
            for stop_id, info in self.link_time_info.items()
        }
        #return dict(self.data["spacing"])
        