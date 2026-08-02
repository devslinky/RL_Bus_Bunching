from setup.ttc_route_29_data.dataloader import DataLoader
from typing import Dict, Tuple, override

class DataLoaderReal(DataLoader):
    """Same as ttc_route_29_data.dataloader.DataLoader, but replaces placeholder values with real data."""

    @property
    @override
    def dispatching_headway(self) -> Tuple[float, float]:
        return tuple(self.data["dispatching_headway"])
    
    @override
    def get_spacing(self) -> Dict[str, float]:
        return dict(self.data["spacing"])