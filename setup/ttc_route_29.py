"""
TTC Route 29 (Dufferin) Simulation Environment

This module defines the network and route schema for TTC Route 29,
based on real APC and AVL data from November 2023.

Route 29 runs along Dufferin Street in Toronto:
- NORTH: Fleet St (Exhibition) -> Wilson Station
- SOUTH: Wilson Station -> Fleet St (Exhibition)

The simulation supports both directions as separate environments.
"""
from collections import defaultdict
from typing import List, Dict, Tuple
from typing_extensions import override
from setup.ttc_route_29_data.dataloader import DataLoader
from setup.ttc_route_29_data_real.dataloader import DataLoaderReal # replace old dataloader with placeholder headway and spacing values with real data derived values
from setup.network import Network
from setup.route import Route_Schema
from setup.config_dataclass import TerminalNodeGeometry, StopNodeGeometry, LinkGeometry, LinkDistribution


class TTC_Route_29_Network(Network):
    """
    Network definition for TTC Route 29.

    Supports both NORTH and SOUTH directions.
    """

    def __init__(self, direction: str = 'north') -> None:
        self.direction = direction.lower()
        self.data_loader = DataLoader(direction)
        super().__init__()

    @override
    def _define_network(self):
        node_ids = self.data_loader.node_ids
        link_time_info = self.data_loader.link_time_info
        spacing = self.data_loader.get_spacing()
        berth_num = 2
        node_x_cum = {}
        x_cum = 0.0
        node_x_cum[node_ids[0]] = x_cum

        for i in range(1, len(node_ids)):
            curr_node = node_ids[i]
            if curr_node in spacing:
                x_cum += spacing[curr_node]
            else:
                x_cum += 250 # arbritrary default spacing if not specified
            node_x_cum[curr_node] = x_cum

        y = 0
        start_terminal_id = node_ids[0]
        end_terminal_id = node_ids[-1]

        for node_id in node_ids:
            x = node_x_cum[node_id]
            if node_id == start_terminal_id or node_id == end_terminal_id:
                terminal_node_geometry = TerminalNodeGeometry(x, y)
                self._G.add_node(node_id, node_type='terminal', terminal_node_geometry=terminal_node_geometry)
            else:
                stop_node_geometry = StopNodeGeometry(x, y, berth_num)
                self._G.add_node(node_id, node_type='stop', stop_node_geometry=stop_node_geometry)
            self._name_coordinates[node_id] = (x, y)

        head_x_cum = 0
        link_id = 0
        for head_node, tail_node in zip(node_ids[:-1], node_ids[1:]):
            if tail_node in spacing:
                link_spacing = spacing[tail_node]
            else:
                link_spacing = 250

            if tail_node in link_time_info:
                tt_info = link_time_info[tail_node]
                tt_mean = tt_info['loc']
                tt_std = tt_info['scale']
                if tt_mean > 0:
                    tt_cv = tt_std / tt_mean
                else:
                    tt_cv = 0.2
            else:
                tt_mean = 50
                tt_cv = 0.2
            tt_type = 'normal'

            link_distribution = LinkDistribution(tt_mean, tt_cv, tt_type)
            link_geometry = LinkGeometry(str(head_node), str(tail_node), head_x_cum, 0, link_spacing)
            self._G.add_edge(
                str(head_node),
                str(tail_node),
                link_id=link_id,
                link_geometry=link_geometry,
                link_distribution=link_distribution,
            )
            link_id += 1
            head_x_cum += link_spacing


class TTC_Route_29_Route_Schema(Route_Schema):
    """
    Route schema for TTC Route 29.

    Defines OD demand, schedule headway, and other route parameters.
    """

    def __init__(self, direction: str = 'north') -> None:
        self.direction = direction.lower()
        self.data_loader = DataLoader(direction)
        self._node_ids = self.data_loader.node_ids
        self._stop_pax_arrival_rate = self.data_loader.stop_pax_arrival_rate
        self._start_terminal_id = self._node_ids[0]
        self._end_terminal_id = self._node_ids[-1]
        self._visit_seq_stop_ids = self._node_ids[1:-1]
        self._H_mean, self._H_std = self.data_loader.dispatching_headway
        super().__init__()

    @override
    def _define_od_table(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Define OD demand table.

        Uses OD rate table inferred from APC boarding/alighting data
        via IPF (Iterative Proportional Fitting) method.
        Falls back to uniform distribution if inferred data is not available.
        """
        inferred_od = self.data_loader.od_rate_table
        if inferred_od:
            route_id = '29N' if self.direction == 'north' else '29S'
            return {route_id: inferred_od}

        od_rate_table = defaultdict(dict)
        for idx, origin_stop in enumerate(self._visit_seq_stop_ids):
            pax_arrival_rate = self._stop_pax_arrival_rate.get(origin_stop, 0.0)
            dest_stops = self._visit_seq_stop_ids[idx + 1:]
            dest_stop_num = len(dest_stops)
            if dest_stop_num > 0:
                for dest_stop in dest_stops:
                    od_rate_table[origin_stop][dest_stop] = pax_arrival_rate / dest_stop_num
            else:
                for dest_stop in self._visit_seq_stop_ids:
                    od_rate_table[origin_stop][dest_stop] = 0.0

        route_id = '29N' if self.direction == 'north' else '29S'
        return {route_id: dict(od_rate_table)}

    @override
    def _define_route_ids(self) -> List[str]:
        route_id = '29N' if self.direction == 'north' else '29S'
        return [route_id]

    @override
    def _define_schedule_headway(self) -> Dict[str, Tuple[float, float]]:
        route_id = '29N' if self.direction == 'north' else '29S'
        return {route_id: (self._H_mean, self._H_std)}

    @override
    def _define_terminal(self) -> Dict[str, str]:
        route_id = '29N' if self.direction == 'north' else '29S'
        return {route_id: self._start_terminal_id}

    @override
    def _define_visit_seq_stops(self) -> Dict[str, List[str]]:
        route_id = '29N' if self.direction == 'north' else '29S'
        return {route_id: self._visit_seq_stop_ids}

    @override
    def _define_end_terminal(self) -> Dict[str, str]:
        route_id = '29N' if self.direction == 'north' else '29S'
        return {route_id: self._end_terminal_id}

    @override
    def _define_boarding_rate(self) -> Dict[str, Dict[str, float]]:
        """
        Define boarding rate (passengers per second) at each stop.

        Based on TTC average boarding time of ~4 seconds per passenger.
        """
        route_id = '29N' if self.direction == 'north' else '29S'
        boarding_rate = 0.25
        return {route_id: {stop_id: boarding_rate for stop_id in self._visit_seq_stop_ids}}

    @override
    def _define_hold_stops(self) -> Dict[str, List[str]]:
        """All stops are holdable."""
        route_id = '29N' if self.direction == 'north' else '29S'
        return {route_id: self._visit_seq_stop_ids.copy()}


class TTC_Route_29_North_Network(TTC_Route_29_Network):
    """Network for Route 29 NORTH (Fleet St -> Wilson Station)."""

    def __init__(self):
        super().__init__('north')


class TTC_Route_29_South_Network(TTC_Route_29_Network):
    """Network for Route 29 SOUTH (Wilson Station -> Fleet St)."""

    def __init__(self):
        super().__init__('south')


class TTC_Route_29_North_Route_Schema(TTC_Route_29_Route_Schema):
    """Route schema for Route 29 NORTH."""

    def __init__(self):
        super().__init__('north')


class TTC_Route_29_South_Route_Schema(TTC_Route_29_Route_Schema):
    """Route schema for Route 29 SOUTH."""

    def __init__(self):
        super().__init__('south')


###### Student Changes: adding ttc_route_29_<dir>_real subclasses to use real data derived headway and spacing values instead of placeholder values

class TTC_Route_29_North_Network_real(TTC_Route_29_Network):
    """Network for Route 29 NORTH with real data derived headway and spacing values."""

    def __init__(self) -> None:
        self.direction = 'north'
        self.data_loader = DataLoaderReal('north')
        Network.__init__(self)          

class TTC_Route_29_South_Network_real(TTC_Route_29_Network):
    """Network for Route 29 SOUTH with real data derived headway and spacing values."""

    def __init__(self) -> None:
        self.direction = 'south'
        self.data_loader = DataLoaderReal('south')
        Network.__init__(self)   

class TTC_Route_29_North_Route_Schema_real(TTC_Route_29_Route_Schema):
    """Route schema for Route 29 NORTH with real data derived headway and spacing values."""

    def __init__(self) -> None:
        self.direction = 'north'
        self.data_loader = DataLoaderReal('north')
        self._node_ids = self.data_loader.node_ids
        self._stop_pax_arrival_rate = self.data_loader.stop_pax_arrival_rate
        self._start_terminal_id = self._node_ids[0]
        self._end_terminal_id = self._node_ids[-1]
        self._visit_seq_stop_ids = self._node_ids[1:-1]
        self._H_mean, self._H_std = self.data_loader.dispatching_headway
        Route_Schema.__init__(self)     

class TTC_Route_29_South_Route_Schema_real(TTC_Route_29_Route_Schema):
    """Route schema for Route 29 SOUTH with real data derived headway and spacing values."""

    def __init__(self) -> None:
        self.direction = 'south'
        self.data_loader = DataLoaderReal('south')
        self._node_ids = self.data_loader.node_ids
        self._stop_pax_arrival_rate = self.data_loader.stop_pax_arrival_rate
        self._start_terminal_id = self._node_ids[0]
        self._end_terminal_id = self._node_ids[-1]
        self._visit_seq_stop_ids = self._node_ids[1:-1]
        self._H_mean, self._H_std = self.data_loader.dispatching_headway
        Route_Schema.__init__(self)   
