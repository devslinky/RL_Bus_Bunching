from typing import List, Dict, Tuple, Literal, Optional
from collections import defaultdict
from copy import deepcopy

from .pax import Pax
from .bus import Bus


class PaxQueue:
    ''' A queue that holds paxs that are waiting for buses at a stop.

    Methods:

    '''

    def __init__(self, stop_id: str, board_truncation: Literal['arrival', 'rtd']):
        # Key is a group (tuple) of routes, value is a list of paxs that can be served by any route in the tuple
        # if the tuple contains only one route, then the paxs are exclusive
        # The name `group`` is used to indicate the paxs that share totally the same routes
        self._route_group_paxs: Dict[Tuple[str, ...],
                                     List[Pax]] = defaultdict(list)

        # the stop id that the queue belongs to
        self._stop_id: str = stop_id

        # the filter type to filter paxs that can be served by the bus
        self._board_truncation: Literal['arrival', 'rtd'] = board_truncation

        # rejection tracking
        self._rejected_pax_count: int = 0
        self._last_rejected_bus_id: Optional[str] = None
        self._rejection_events: List[Tuple[str, str, int, int]] = []  # (bus_id, route_id, time, num_rejected)

    def add_pax(self, pax: Pax):
        ''' Add a pax to the queue.

        Args:
            pax: the pax to be added to the queue

        '''
        routes = tuple(pax.routes)
        self._route_group_paxs[routes].append(pax)

    def board(self, bus: Bus, t: int):
        ''' Board paxs in this queue to a bus

        Args:
            bus: the bus to board paxs

        Note:
            Passengers can only board if the bus has remaining capacity.
            If the bus is full, passengers stay in the queue and wait for the next bus.
        '''
        # the buse has two `board_status`: boarding and idle
        # if the bus is boarding, then it is in the middle of boarding
        if bus.board_status == 'boarding':
            # board a fraction of a pax in the queue
            bus.accumate_board_fraction()
            return
        # if the bus is idle, then it is ready to board
        elif bus.board_status == 'idle':
            # Check if bus has room for more passengers
            if not bus.has_room:
                # Bus is full, count rejected passengers (deduplicated per bus)
                if bus.bus_id != self._last_rejected_bus_id:
                    served_groups = self._get_served_groups(bus.route_id)
                    for group in served_groups:
                        paxs = self._route_group_paxs[group]
                        if self._board_truncation == 'arrival':
                            paxs = self._filter_pax_arrival_after_bus_arrival(bus, paxs)
                        num_rejected = len(paxs)
                        self._rejected_pax_count += num_rejected
                        if num_rejected > 0:
                            self._rejection_events.append((bus.bus_id, bus.route_id, t, num_rejected))
                    self._last_rejected_bus_id = bus.bus_id
                return

            # Reset dedup tracker when a bus successfully boards
            self._last_rejected_bus_id = None

            # 0. find pax groups that the bus can serve
            served_groups = self._get_served_groups(bus.route_id)
            # no group can be served
            if len(served_groups) == 0:
                return

            # 1. serve the exlusive groups first
            exclusive_group = served_groups[0]
            assert len(exclusive_group) == 1, 'Only one exclusive group for now'
            paxs = self._route_group_paxs[exclusive_group]
            board_paxs = []
            # if `self._boarding_runcation` is 'arrival', then only board paxs that arrive before the bus arrives
            # i.e., filter out paxs that arrive after the bus arrives
            if self._board_truncation == 'arrival':
                board_paxs = self._filter_pax_arrival_after_bus_arrival(
                    bus, paxs)
            else:
                board_paxs = paxs

            if len(board_paxs) == 0:
                return
            # put the pax in the head of the queue on board, but the boarding process is not finished
            head_pax = board_paxs[0]
            # the bus's boarding status will be set to 'boarding' in this bus's board method
            bus.board(head_pax, t)
            paxs.remove(head_pax)
            bus.accumate_board_fraction()

        # TODO 2. serve common-line groups, for now, there is only one exclusive group

    def accumulate_out_vehicle_delay(self):
        for group, paxs in self._route_group_paxs.items():
            for pax in paxs:
                pax.accumulate_out_vehicle_delay()

    def get_pax_arrival_times(self) -> List[int]:
        '''Get the arrival times of all waiting passengers.

        Returns:
            A list of arrival times for all passengers across all route groups
        '''
        arrival_times = []
        for paxs in self._route_group_paxs.values():
            for pax in paxs:
                arrival_times.append(pax.arrival_time)
        return arrival_times

    def get_total_pax_num(self) -> int:
        ''' Get the total number of paxs for all the routes

        Returns:
            The total number of paxs for all the routes
        '''
        total_pax_sum = sum([len(self._route_group_paxs[group])
                             for group in self._route_group_paxs.keys()])
        return total_pax_sum

    def check_remaining_pax_num(self, bus: Bus) -> int:
        '''Check how many paxs can still board this bus.

        This considers both:
        1. Passengers waiting at the stop who can take this bus
        2. The remaining capacity of the bus

        If the bus is full, returns 0 even if passengers are waiting
        (they will wait for the next bus).

        Args:
            bus: the bus to be checked

        Returns:
            The number of remaining paxs that can board the bus
            (limited by bus capacity)
        '''
        # If bus is full, no more passengers can board
        if not bus.has_room:
            return 0

        served_groups = self._get_served_groups(bus.route_id)
        waiting_pax_num = 0
        for group in served_groups:
            paxs = self._route_group_paxs[group]
            board_paxs = []
            if self._board_truncation == 'arrival':
                board_paxs = self._filter_pax_arrival_after_bus_arrival(
                    bus, paxs)
            else:
                board_paxs = paxs
            waiting_pax_num += len(board_paxs)

        # Return the minimum of waiting passengers and remaining capacity
        return min(waiting_pax_num, bus.remaining_capacity)

    def _get_served_groups(self, bus_route_id: str) -> List[Tuple[str, ...]]:
        '''Get all the served groups of paxs for a bus given its route id

        Args:
            bus_route_id: the route id of the bus

        Returns:
            A list of tuple, where each tuple is a group of routes that the bus can serve
        '''
        served_groups = []
        for route_group in self._route_group_paxs.keys():
            if bus_route_id in route_group:
                served_groups.append(route_group)
        return served_groups

    def get_rejected_pax_count(self) -> int:
        return self._rejected_pax_count

    def get_rejection_events(self) -> List[Tuple[str, str, int, int]]:
        return self._rejection_events

    def reset_rejected_pax_count(self):
        self._rejected_pax_count = 0
        self._last_rejected_bus_id = None
        self._rejection_events = []

    def _filter_pax_arrival_after_bus_arrival(self, bus: Bus, paxs: List[Pax]) -> List[Pax]:
        filtered_paxs = [pax for pax in paxs if pax.arrival_time <
                         bus.bus_log.stop_arrival_time[self._stop_id]]
        return filtered_paxs
