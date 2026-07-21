from typing import Literal, List, Optional, Dict, Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


@dataclass
class TrajectoryPoint:
    spot_type: str
    spot_id: str
    distance_from_terminal: float
    status: Literal['running_on_link', 'queueing_at_stop',
                    'dwelling_at_stop', 'holding', 'finished']


def plot_time_space_diagram(
    buses,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (14, 8),
    dpi: int = 150,
    time_unit: Literal['seconds', 'minutes', 'hours'] = 'minutes',
    distance_unit: Literal['meters', 'kilometers'] = 'kilometers',
    show_stop_lines: bool = True,
    show_legend: bool = True,
    show_stats: bool = True,
    color_by_bus: bool = False,
    alpha: float = 0.8,
    metrics: Optional[Dict[str, float]] = None,
    left_paxs: Optional[List] = None
):
    """
    Plot a time-space diagram for bus trajectories.

    Args:
        buses: List of Bus objects with trajectory data
        save_path: Path to save the figure (None to display)
        title: Custom title for the plot
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
        time_unit: Unit for time axis ('seconds', 'minutes', 'hours')
        distance_unit: Unit for distance axis ('meters', 'kilometers')
        show_stop_lines: Whether to show horizontal lines at stop locations
        show_legend: Whether to show the legend
        show_stats: Whether to show statistics box
        color_by_bus: Whether to use different colors for each bus
        alpha: Transparency for trajectory lines
        metrics: Optional dict of metrics to display (e.g., {'headway_std': 45.2})
        left_paxs: Optional list of Pax objects that have finished their trips
    """
    # Set up the figure with better styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Time and distance conversion factors
    time_factor = {'seconds': 1, 'minutes': 60, 'hours': 3600}[time_unit]
    time_label = {'seconds': 'Time (s)', 'minutes': 'Time (min)', 'hours': 'Time (h)'}[time_unit]

    dist_factor = {'meters': 1, 'kilometers': 1000}[distance_unit]
    dist_label = {'meters': 'Distance from Terminal (m)', 'kilometers': 'Distance from Terminal (km)'}[distance_unit]

    # Color palette for buses
    if color_by_bus:
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(len(buses))]
    else:
        colors = ['#2c3e50'] * len(buses)  # Dark blue-gray for all buses

    # Collect stop locations for horizontal lines
    stop_locations: Dict[str, float] = {}

    # Statistics tracking
    all_hold_times = []
    all_queue_times = []
    all_dwell_times = []
    trip_times = []

    # Track time range
    min_time, max_time = float('inf'), float('-inf')
    max_distance = 0

    for bus_idx, bus in enumerate(buses):
        if not bus.trajectory:
            continue

        color = colors[bus_idx]

        # Extract trajectory data
        times = []
        distances = []
        for t, point in sorted(bus.trajectory.items()):
            times.append(t / time_factor)
            distances.append(point.distance_from_terminal / dist_factor)

            # Collect stop locations
            if point.spot_type == 'stop' or point.spot_type == 'holder':
                stop_locations[point.spot_id] = point.distance_from_terminal / dist_factor

        if times:
            min_time = min(min_time, min(times))
            max_time = max(max_time, max(times))
            max_distance = max(max_distance, max(distances))

            # Plot main trajectory
            ax.plot(times, distances, color=color, linewidth=1.0, alpha=alpha, zorder=2)

            # Calculate trip time
            if len(times) > 1:
                trip_times.append((max(times) - min(times)) * time_factor)

        # Plot holding durations (green)
        hold_segments = defaultdict(list)
        for t, point in bus.trajectory.items():
            if point.spot_type == 'holder':
                hold_segments[point.spot_id].append((t / time_factor, point.distance_from_terminal / dist_factor))

        for spot_id, points in hold_segments.items():
            if len(points) >= 2:
                ts = [p[0] for p in points]
                y = points[0][1]
                start, end = min(ts), max(ts)
                hold_duration = (end - start) * time_factor
                all_hold_times.append(hold_duration)
                ax.hlines(y=y, xmin=start, xmax=end, color='#27ae60', linewidth=3.5, zorder=3, alpha=0.9)

        # Plot queueing durations (red)
        queue_segments = defaultdict(list)
        for t, point in bus.trajectory.items():
            if point.status == 'queueing_at_stop':
                queue_segments[point.spot_id].append((t / time_factor, point.distance_from_terminal / dist_factor))

        for spot_id, points in queue_segments.items():
            if len(points) >= 2:
                ts = [p[0] for p in points]
                y = points[0][1]
                start, end = min(ts), max(ts)
                queue_duration = (end - start) * time_factor
                all_queue_times.append(queue_duration)
                ax.hlines(y=y, xmin=start, xmax=end, color='#e74c3c', linewidth=3.5, zorder=4, alpha=0.9)

        # Plot dwelling durations (orange)
        dwell_segments = defaultdict(list)
        for t, point in bus.trajectory.items():
            if point.status == 'dwelling_at_stop':
                dwell_segments[point.spot_id].append((t / time_factor, point.distance_from_terminal / dist_factor))

        for spot_id, points in dwell_segments.items():
            if len(points) >= 2:
                ts = [p[0] for p in points]
                y = points[0][1]
                start, end = min(ts), max(ts)
                dwell_duration = (end - start) * time_factor
                all_dwell_times.append(dwell_duration)
                ax.hlines(y=y, xmin=start, xmax=end, color='#f39c12', linewidth=2.5, zorder=2, alpha=0.7)

    # Draw horizontal lines at stop locations
    if show_stop_lines and stop_locations:
        sorted_stops = sorted(stop_locations.items(), key=lambda x: x[1])
        for i, (stop_id, dist) in enumerate(sorted_stops):
            ax.axhline(y=dist, color='#bdc3c7', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
            # Label every nth stop to avoid clutter
            if i % max(1, len(sorted_stops) // 10) == 0:
                ax.text(max_time * 1.01, dist, f'S{stop_id}', fontsize=7, va='center', alpha=0.7)

    # Styling
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(dist_label, fontsize=12, fontweight='bold')

    # Set axis limits with padding
    if min_time != float('inf'):
        ax.set_xlim(min_time - (max_time - min_time) * 0.02, max_time * 1.05)
        ax.set_ylim(-max_distance * 0.02, max_distance * 1.02)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title('Time-Space Diagram', fontsize=14, fontweight='bold', pad=15)

    # Legend
    if show_legend:
        legend_elements = [
            Line2D([0], [0], color='#2c3e50', linewidth=1.5, label='Bus Trajectory'),
            Line2D([0], [0], color='#f39c12', linewidth=3, label='Dwelling'),
            Line2D([0], [0], color='#27ae60', linewidth=3, label='Holding'),
            Line2D([0], [0], color='#e74c3c', linewidth=3, label='Queueing'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    # Statistics box
    if show_stats:
        stats_text = f"Buses: {len(buses)}\n"

        if trip_times:
            avg_trip = np.mean(trip_times)
            if time_unit == 'minutes':
                stats_text += f"Avg Trip: {avg_trip/60:.1f} min\n"
            elif time_unit == 'hours':
                stats_text += f"Avg Trip: {avg_trip/3600:.2f} h\n"
            else:
                stats_text += f"Avg Trip: {avg_trip:.0f} s\n"

        if all_hold_times:
            stats_text += f"Avg Hold: {np.mean(all_hold_times):.1f} s\n"
            stats_text += f"Total Holds: {len(all_hold_times)}\n"

        if all_queue_times:
            stats_text += f"Avg Queue: {np.mean(all_queue_times):.1f} s\n"
            stats_text += f"Queue Events: {len(all_queue_times)}\n"

        if all_dwell_times:
            stats_text += f"Avg Dwell: {np.mean(all_dwell_times):.1f} s"

        # Add passenger wait time stats
        if left_paxs:
            out_vehicle = [
                pax.board_time - pax.arrival_time for pax in left_paxs
                if pax.board_time is not None]
            in_vehicle = [
                pax.alight_time - pax.board_time for pax in left_paxs
                if pax.board_time is not None and pax.alight_time is not None]
            stats_text += "\n" + "-" * 15 + "\n"
            stats_text += f"Pax Served: {len(left_paxs)}\n"
            if out_vehicle:
                stats_text += f"Avg Wait: {np.mean(out_vehicle):.1f} s\n"
            if in_vehicle:
                stats_text += f"Avg In-Veh: {np.mean(in_vehicle):.1f} s\n"
            if out_vehicle and in_vehicle:
                total = [o + i for o, i in zip(out_vehicle[:len(in_vehicle)], in_vehicle)]
                stats_text += f"Avg Total: {np.mean(total):.1f} s"

        # Add provided metrics
        if metrics:
            stats_text += "\n" + "-" * 15 + "\n"
            for name, value in metrics.items():
                stats_text += f"{name}: {value:.2f}\n"
            stats_text = stats_text.rstrip('\n')

        # Position stats box in upper right
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    # Tight layout
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Time-space diagram saved to: {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_headway_analysis(
    buses,
    stop_id: str,
    save_path: Optional[str] = None,
    schedule_headway: Optional[float] = None,
    figsize: tuple = (12, 5),
    dpi: int = 150
):
    """
    Plot headway analysis for a specific stop.

    Args:
        buses: List of Bus objects with trajectory data
        stop_id: Stop ID to analyze
        save_path: Path to save the figure
        schedule_headway: Expected headway in seconds (for reference line)
        figsize: Figure size
        dpi: Resolution
    """
    # Collect arrival times at the stop
    arrival_times = []

    for bus in buses:
        for t, point in bus.trajectory.items():
            if point.spot_id == stop_id and point.status == 'dwelling_at_stop':
                arrival_times.append(t)
                break  # First arrival at this stop

    if len(arrival_times) < 2:
        print(f"Not enough arrivals at stop {stop_id} to analyze headways")
        return None, None

    arrival_times = sorted(arrival_times)
    headways = [arrival_times[i+1] - arrival_times[i] for i in range(len(arrival_times)-1)]
    bus_indices = list(range(2, len(arrival_times) + 1))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Left: Headway over time
    ax1.bar(bus_indices, headways, color='#3498db', alpha=0.8, edgecolor='#2980b9')
    if schedule_headway:
        ax1.axhline(y=schedule_headway, color='#e74c3c', linestyle='--', linewidth=2, label=f'Scheduled ({schedule_headway}s)')
    ax1.set_xlabel('Bus Number', fontsize=11)
    ax1.set_ylabel('Headway (seconds)', fontsize=11)
    ax1.set_title(f'Headways at Stop {stop_id}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Headway distribution
    ax2.hist(headways, bins=20, color='#3498db', alpha=0.8, edgecolor='#2980b9')
    if schedule_headway:
        ax2.axvline(x=schedule_headway, color='#e74c3c', linestyle='--', linewidth=2, label=f'Scheduled ({schedule_headway}s)')
    ax2.axvline(x=np.mean(headways), color='#27ae60', linestyle='-', linewidth=2, label=f'Mean ({np.mean(headways):.1f}s)')
    ax2.set_xlabel('Headway (seconds)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Headway Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add statistics
    stats_text = (f"Mean: {np.mean(headways):.1f}s\n"
                  f"Std: {np.std(headways):.1f}s\n"
                  f"Min: {np.min(headways):.1f}s\n"
                  f"Max: {np.max(headways):.1f}s\n"
                  f"CV: {np.std(headways)/np.mean(headways):.2f}")
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9)
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Headway analysis saved to: {save_path}")
    else:
        plt.show()

    return fig, (ax1, ax2)


def save_experiment_data(
    buses,
    output_dir: str = "experiment_data",
    prefix: str = "",
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    left_paxs: Optional[List] = None,
    rejection_events: Optional[List] = None
) -> Dict[str, str]:
    """
    Save experiment raw data to CSV files for further analysis.

    Creates multiple CSV files:
    - trajectories.csv: Time-space trajectory data for all buses
    - bus_summary.csv: Per-bus summary statistics
    - stop_events.csv: Arrival, RTD, departure times at each stop
    - holding_events.csv: All holding events with durations
    - passengers.csv: Per-passenger wait time data
    - metrics.csv: Episode metrics and configuration

    Args:
        buses: List of Bus objects with trajectory data
        output_dir: Directory to save CSV files
        prefix: Optional prefix for filenames (e.g., "episode_001_")
        metrics: Optional dict of episode metrics
        config: Optional dict of run configuration
        left_paxs: Optional list of Pax objects that have finished their trips

    Returns:
        Dict mapping data type to saved file path
    """
    import csv
    import os
    from datetime import datetime

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    saved_files = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = f"{prefix}" if prefix else f"{timestamp}_"

    # 1. Trajectory data (time-space points)
    trajectory_file = os.path.join(output_dir, f"{file_prefix}trajectories.csv")
    with open(trajectory_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bus_id', 'route_id', 'time_sec', 'distance_m',
            'spot_type', 'spot_id', 'status'
        ])

        for bus in buses:
            for t, point in sorted(bus.trajectory.items()):
                writer.writerow([
                    bus.bus_id,
                    bus.route_id,
                    t,
                    point.distance_from_terminal,
                    point.spot_type,
                    point.spot_id,
                    point.status
                ])

    saved_files['trajectories'] = trajectory_file
    print(f"Saved: {trajectory_file}")

    # 2. Bus summary statistics
    bus_summary_file = os.path.join(output_dir, f"{file_prefix}bus_summary.csv")
    with open(bus_summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bus_id', 'route_id', 'dispatch_time', 'end_time', 'trip_time_sec',
            'total_hold_time_sec', 'total_dwell_time_sec', 'total_queue_time_sec',
            'num_stops_visited', 'is_need_to_hold'
        ])

        for bus in buses:
            if not bus.trajectory:
                continue

            times = sorted(bus.trajectory.keys())
            dispatch_time = min(times) if times else None
            end_time = max(times) if times else None
            trip_time = (end_time - dispatch_time) if dispatch_time and end_time else None

            # Calculate total holding time
            hold_time = 0
            for t, point in bus.trajectory.items():
                if point.spot_type == 'holder':
                    hold_time += 1  # Each timestep in holder = 1 second

            # Calculate total dwell time
            dwell_time = 0
            for t, point in bus.trajectory.items():
                if point.status == 'dwelling_at_stop':
                    dwell_time += 1

            # Calculate total queue time
            queue_time = 0
            for t, point in bus.trajectory.items():
                if point.status == 'queueing_at_stop':
                    queue_time += 1

            # Count stops visited
            stops_visited = set()
            for t, point in bus.trajectory.items():
                if point.spot_type in ['stop', 'holder']:
                    stops_visited.add(point.spot_id)

            writer.writerow([
                bus.bus_id,
                bus.route_id,
                dispatch_time,
                end_time,
                trip_time,
                hold_time,
                dwell_time,
                queue_time,
                len(stops_visited),
                bus.is_need_to_hold
            ])

    saved_files['bus_summary'] = bus_summary_file
    print(f"Saved: {bus_summary_file}")

    # 3. Stop events (arrival, RTD, departure at each stop)
    stop_events_file = os.path.join(output_dir, f"{file_prefix}stop_events.csv")
    with open(stop_events_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bus_id', 'route_id', 'stop_id', 'arrival_time', 'rtd_time',
            'departure_time', 'dwell_time_sec', 'hold_time_sec',
            'epsilon_arrival', 'epsilon_rtd', 'epsilon_departure'
        ])

        for bus in buses:
            if not hasattr(bus, 'bus_log'):
                continue

            log = bus.bus_log
            all_stops = set(log.stop_arrival_time.keys()) | set(log.stop_rtd_time.keys()) | set(log.stop_departure_time.keys())

            for stop_id in sorted(all_stops):
                arrival = log.stop_arrival_time.get(stop_id)
                rtd = log.stop_rtd_time.get(stop_id)
                departure = log.stop_departure_time.get(stop_id)

                dwell = (rtd - arrival) if arrival and rtd else None
                hold = (departure - rtd) if rtd and departure else None

                eps_arr = log.stop_epsilon_arrival.get(stop_id)
                eps_rtd = log.stop_epsilon_rtd.get(stop_id)
                eps_dep = log.stop_epsilon_departure.get(stop_id)

                writer.writerow([
                    bus.bus_id,
                    bus.route_id,
                    stop_id,
                    arrival,
                    rtd,
                    departure,
                    dwell,
                    hold,
                    eps_arr,
                    eps_rtd,
                    eps_dep
                ])

    saved_files['stop_events'] = stop_events_file
    print(f"Saved: {stop_events_file}")

    # 4. Holding events with details
    holding_events_file = os.path.join(output_dir, f"{file_prefix}holding_events.csv")
    with open(holding_events_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bus_id', 'route_id', 'stop_id', 'hold_start_time',
            'hold_end_time', 'hold_duration_sec', 'distance_m'
        ])

        for bus in buses:
            # Group holding segments by stop
            hold_segments: Dict[str, List] = defaultdict(list)
            for t, point in bus.trajectory.items():
                if point.spot_type == 'holder':
                    hold_segments[point.spot_id].append((t, point.distance_from_terminal))

            for stop_id, points in hold_segments.items():
                if len(points) >= 1:
                    times = [p[0] for p in points]
                    distance = points[0][1]
                    start_time = min(times)
                    end_time = max(times)
                    duration = end_time - start_time + 1  # +1 because inclusive

                    writer.writerow([
                        bus.bus_id,
                        bus.route_id,
                        stop_id,
                        start_time,
                        end_time,
                        duration,
                        distance
                    ])

    saved_files['holding_events'] = holding_events_file
    print(f"Saved: {holding_events_file}")

    # 4b. Rejection events
    if rejection_events:
        rejection_file = os.path.join(output_dir, f"{file_prefix}rejection_events.csv")
        with open(rejection_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'stop_id', 'bus_id', 'route_id', 'time_sec', 'num_rejected_pax'
            ])
            for stop_id, bus_id, route_id, t, num_rejected in rejection_events:
                writer.writerow([stop_id, bus_id, route_id, t, num_rejected])

        saved_files['rejection_events'] = rejection_file
        print(f"Saved: {rejection_file}")

    # 5. Headway data at each stop
    headway_file = os.path.join(output_dir, f"{file_prefix}headways.csv")
    with open(headway_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'stop_id', 'route_id', 'bus_id', 'prev_bus_id',
            'arrival_time', 'prev_arrival_time', 'headway_sec'
        ])

        # Group arrivals by stop
        stop_arrivals: Dict[str, List] = defaultdict(list)
        for bus in buses:
            if not hasattr(bus, 'bus_log'):
                continue
            for stop_id, arr_time in bus.bus_log.stop_arrival_time.items():
                stop_arrivals[stop_id].append((arr_time, bus.bus_id, bus.route_id))

        for stop_id, arrivals in stop_arrivals.items():
            arrivals.sort(key=lambda x: x[0])  # Sort by arrival time
            for i in range(1, len(arrivals)):
                curr_time, curr_bus, route_id = arrivals[i]
                prev_time, prev_bus, _ = arrivals[i-1]
                headway = curr_time - prev_time

                writer.writerow([
                    stop_id,
                    route_id,
                    curr_bus,
                    prev_bus,
                    curr_time,
                    prev_time,
                    headway
                ])

    saved_files['headways'] = headway_file
    print(f"Saved: {headway_file}")

    # 6. Passenger wait time data
    if left_paxs:
        pax_file = os.path.join(output_dir, f"{file_prefix}passengers.csv")
        with open(pax_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pax_id', 'origin', 'destination', 'routes',
                'arrival_time', 'board_time', 'alight_time',
                'out_vehicle_wait_sec', 'in_vehicle_wait_sec', 'total_wait_sec'
            ])

            for pax in left_paxs:
                out_wait = (pax.board_time - pax.arrival_time
                            if pax.board_time is not None else None)
                in_wait = (pax.alight_time - pax.board_time
                           if pax.board_time is not None and pax.alight_time is not None
                           else None)
                total_wait = (out_wait + in_wait
                              if out_wait is not None and in_wait is not None
                              else None)

                writer.writerow([
                    pax.pax_id,
                    pax.origin,
                    pax.destination,
                    ';'.join(pax.routes),
                    pax.arrival_time,
                    pax.board_time,
                    pax.alight_time,
                    out_wait,
                    in_wait,
                    total_wait
                ])

        saved_files['passengers'] = pax_file
        print(f"Saved: {pax_file}")

    # 7. Metrics and configuration
    metrics_file = os.path.join(output_dir, f"{file_prefix}metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'name', 'value'])

        # Add metrics
        if metrics:
            for name, value in metrics.items():
                writer.writerow(['metric', name, value])

        # Add config
        if config:
            for name, value in config.items():
                if not isinstance(value, (dict, list)):
                    writer.writerow(['config', name, value])

        # Add summary stats
        writer.writerow(['summary', 'total_buses', len(buses)])
        writer.writerow(['summary', 'timestamp', timestamp])

        # Calculate aggregate stats
        total_hold = 0
        total_trips = 0
        for bus in buses:
            if bus.trajectory:
                times = sorted(bus.trajectory.keys())
                if len(times) > 1:
                    total_trips += 1
                for t, point in bus.trajectory.items():
                    if point.spot_type == 'holder':
                        total_hold += 1

        writer.writerow(['summary', 'completed_trips', total_trips])
        writer.writerow(['summary', 'total_hold_time_sec', total_hold])

    saved_files['metrics'] = metrics_file
    print(f"Saved: {metrics_file}")

    print(f"\nAll experiment data saved to: {output_dir}/")
    return saved_files


def load_trajectory_csv(filepath: str) -> List[Dict]:
    """
    Load trajectory data from CSV file.

    Args:
        filepath: Path to trajectories.csv file

    Returns:
        List of trajectory records as dictionaries
    """
    import csv

    records = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['time_sec'] = int(row['time_sec'])
            row['distance_m'] = float(row['distance_m'])
            records.append(row)

    return records

