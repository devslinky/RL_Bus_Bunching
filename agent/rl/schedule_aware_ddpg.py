"""
Schedule-Aware DDPG Agent

This agent addresses the key weaknesses of naive DDPG approaches:
1. Rich state representation including schedule deviation (epsilon)
2. Better reward function based on schedule adherence
3. Domain knowledge integration via VirtualBus
4. Improved training stability with layer normalization and gradient clipping

Key differences from Local_Spacing_DDPG:
- Uses epsilon (schedule deviation) instead of just spacing
- Incorporates passenger arrival rate and boarding rate (beta)
- Uses VirtualBus for schedule-based control
- Multi-component reward function

"""

from copy import deepcopy
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import random
import numpy as np
import torch
import torch.nn as nn

from simulator.snapshot import Snapshot
from setup.blueprint import Blueprint
from simulator.virtual_bus import VirtualBus
from simulator.simulator import Simulator

from .rl_agent import RLAgent
from .net import Actor_Net, Critic_Net


@dataclass(frozen=True)
class SAR:
    """State-Action-Reward tuple for a single decision point"""
    state: List[float]
    action: float
    reward: Optional[float]


@dataclass(frozen=True)
class SARS:
    """State-Action-Reward-NextState tuple for experience replay"""
    state: List[float]
    action: float
    reward: float
    next_state: List[float]


class EnhancedActorNet(nn.Module):
    """Actor network with layer normalization for training stability"""

    def __init__(self, state_size: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()

        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EnhancedCriticNet(nn.Module):
    """Critic network with layer normalization"""

    def __init__(self, state_size: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()

        # State + Action as input
        input_size = state_size + 1

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.network(state_action)


class Schedule_Aware_DDPG(RLAgent):
    """
    Schedule-Aware DDPG Agent with TD3 Improvements

    This agent uses rich state representation including:
    - epsilon_arrival: schedule deviation at arrival
    - epsilon_last: previous bus's schedule deviation
    - beta: passenger arrival rate / boarding rate
    - normalized_headway: current headway / scheduled headway
    - normalized_forward_spacing
    - normalized_backward_spacing

    Key improvements over basic DDPG:
    - Twin critics (TD3) to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing
    - Improvement-based reward function
    - Reward normalization for stability
    """

    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint,
                 run_config: Dict[str, Any]) -> None:
        super().__init__(agent_config, blueprint)

        self._blueprint = blueprint
        self._run_config = run_config

        # State and action configuration
        self._state_size = agent_config['state_size']  # Should be 6 for full state
        self._max_hold_time = agent_config['max_hold_time']
        self._H = agent_config['schedule_headway']

        # Reward weights (rebalanced for stability)
        self._w_epsilon = agent_config.get('w_epsilon', 0.5)  # Weight for epsilon improvement
        self._w_spacing = agent_config.get('w_spacing', 0.1)  # Weight for spacing penalty
        self._w_action = agent_config.get('w_action', 0.01)   # Weight for action penalty
        self._reward_baseline = agent_config.get('reward_baseline', 0.1)  # Positive baseline
        self._reward_clip = agent_config.get('reward_clip', 2.0)  # Clip extreme rewards

        # Schedule-based control parameters (from SimpleControlNonlinear)
        self._slack = agent_config.get('slack', 10)
        self._base_type = agent_config.get('base_type', 'arrival')
        self._episode_num_for_stabilize = agent_config.get(
            'episode_num_for_stabilize_average_hold', 0)
        self._episode_duration_for_stabilize = agent_config.get(
            'episode_duration_for_stabilize_average_hold', 10800)

        # Calculate route information
        self._route_stop_arrival_rate = self._calculate_total_arrival_rate()
        self._route_schedule = self._set_schedule_headway()

        # Initialize VirtualBus for epsilon calculation
        self._generate_virtual_bus()

        # Use enhanced networks with layer normalization
        hidden_sizes = tuple(agent_config['hidden_size'])
        self._actor_net = EnhancedActorNet(self._state_size, hidden_sizes)

        # TD3: Twin critics to reduce overestimation
        self._critic_net_1 = EnhancedCriticNet(self._state_size, hidden_sizes)
        self._critic_net_2 = EnhancedCriticNet(self._state_size, hidden_sizes)

        # Target networks
        self._target_actor_net = deepcopy(self._actor_net)
        self._target_critic_net_1 = deepcopy(self._critic_net_1)
        self._target_critic_net_2 = deepcopy(self._critic_net_2)
        for param in self._target_actor_net.parameters():
            param.requires_grad = False
        for param in self._target_critic_net_1.parameters():
            param.requires_grad = False
        for param in self._target_critic_net_2.parameters():
            param.requires_grad = False

        # Optimizers
        self._actor_lr = agent_config['actor_lr']
        self._critic_lr = agent_config['critic_lr']
        self._lr_decay = agent_config.get('lr_decay', 0.999)  # Learning rate decay

        self._actor_optim = torch.optim.Adam(
            self._actor_net.parameters(), lr=self._actor_lr)
        self._critic_optim = torch.optim.Adam(
            list(self._critic_net_1.parameters()) + list(self._critic_net_2.parameters()),
            lr=self._critic_lr)

        # Learning rate schedulers
        self._actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._actor_optim, gamma=self._lr_decay)
        self._critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._critic_optim, gamma=self._lr_decay)

        # Training hyperparameters
        self._gamma = agent_config['gamma']
        self._polyak = agent_config['polya']
        self._memory = deque(maxlen=agent_config['memory_size'])
        self._update_cycle = agent_config['update_cycle']
        self._batch_size = agent_config['batch_size']
        self._grad_clip = agent_config.get('grad_clip', 0.5)  # Stronger clipping

        # TD3: Policy delay and target smoothing
        self._policy_delay = agent_config.get('policy_delay', 2)  # Update actor every N critic updates
        self._target_noise = agent_config.get('target_noise', 0.2)  # Noise added to target actions
        self._noise_clip = agent_config.get('noise_clip', 0.5)  # Clip target noise

        # Exploration noise
        self._init_noise_level = agent_config['init_noise_level']
        self._decay_rate = agent_config['decay_rate']
        self._noise_level = self._init_noise_level
        self._min_noise = agent_config.get('min_noise', 0.01)

        # State tracking for transition construction
        self._bus_stop_sar: Dict[Tuple[str, str], List[Tuple[str, SAR]]] = defaultdict(list)
        self._add_event_count = 0
        self._learn_count = 0
        self._critic_update_count = 0  # Track critic updates for policy delay

        # Normalization constants
        self._epsilon_scale = agent_config.get('epsilon_scale', 100.0)
        self._spacing_scale = agent_config.get('spacing_scale', 1000.0)

        # Reward normalization (running statistics)
        self._use_reward_norm = agent_config.get('use_reward_normalization', True)
        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

        # Track previous epsilon for improvement-based reward
        self._bus_prev_epsilon: Dict[Tuple[str, str], float] = {}

    @property
    def virtual_bus(self) -> VirtualBus:
        """Expose virtual_bus for Simulator to use"""
        return self._virtual_bus

    def _set_schedule_headway(self) -> Dict[str, float]:
        """Get schedule headway for each route"""
        route_schedule = {}
        for route_id, route in self._blueprint.route_schema.route_details_by_id.items():
            route_schedule[route_id] = route.schedule_headway
        return route_schedule

    def _calculate_total_arrival_rate(self) -> Dict[str, Dict[str, float]]:
        """Calculate total passenger arrival rate at each stop"""
        route_total_arrival_rate = defaultdict(dict)
        for route_id, route in self._blueprint.route_schema.route_details_by_id.items():
            for origin_stop_id, destination_rate in route.od_rate_table.items():
                total_origin_demand = sum(destination_rate.values())
                route_total_arrival_rate[route_id][origin_stop_id] = total_origin_demand

            # Last stop has no arrivals
            last_stop_id = route.visit_seq_stops[-1]
            route_total_arrival_rate[route_id][last_stop_id] = 0.0

        return dict(route_total_arrival_rate)

    def _generate_virtual_bus(self):
        """Generate VirtualBus for schedule-based epsilon calculation"""
        self._virtual_bus = VirtualBus(self._blueprint)
        self._virtual_bus.initialize_with_perfect_schedule(
            self._route_stop_arrival_rate, self._slack)

        if self._episode_num_for_stabilize == 0:
            print('Schedule_Aware_DDPG: Using initial virtual bus schedule')
            return

        # Iteratively refine VirtualBus schedule
        print(f'Schedule_Aware_DDPG: Stabilizing virtual bus over {self._episode_num_for_stabilize} episodes...')
        for ep in range(self._episode_num_for_stabilize):
            simulator = Simulator(self._blueprint, self, self._run_config)
            stop_bus_hold_action: Dict[Tuple[str, str, str], float] = {}

            for t in range(self._episode_duration_for_stabilize):
                snapshot = simulator.step(t, stop_bus_hold_action)
                # Use simple heuristic for stabilization
                stop_bus_hold_action = self._calculate_hold_time_heuristic(snapshot)
                snapshot.record_holding_time(stop_bus_hold_action)

            route_stop_average_hold_time = simulator.get_stop_average_hold_time()
            self._virtual_bus.update_trajectory(route_stop_average_hold_time)

        print('Schedule_Aware_DDPG: Virtual bus stabilization complete')

    def _calculate_hold_time_heuristic(self, snapshot: Snapshot) -> Dict[Tuple[str, str, str], float]:
        """Simple heuristic for VirtualBus stabilization (similar to SimpleControlNonlinear)"""
        stop_bus_hold_time = {}
        action_buses = snapshot.holder_snapshot.action_buses

        for (stop_id, route_id, bus_id) in action_buses:
            if not snapshot.bus_snapshots[(route_id, bus_id)].is_need_to_hold:
                stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0
                continue

            try:
                epsilon_arrival, epsilon_rtd = snapshot.get_bus_epsilon(route_id, bus_id, stop_id)
                last_epsilon_arrival, last_epsilon_rtd = snapshot.get_stop_epsilon(route_id, stop_id, bus_id)

                # Get beta
                stop_boarding_rate = self._blueprint.route_schema.route_details_by_id[route_id].boarding_rate
                arrival_rate = self._route_stop_arrival_rate[route_id][stop_id]
                beta = arrival_rate / stop_boarding_rate[stop_id] if stop_boarding_rate[stop_id] > 0 else 0

                # Simple control formula
                f0 = 0.2
                hold_time = -epsilon_arrival + f0 * epsilon_arrival
                hold_time += beta * (last_epsilon_arrival - epsilon_arrival)
                hold_time += self._slack
                hold_time = max(0, hold_time)

            except Exception:
                hold_time = self._slack

            stop_bus_hold_time[(stop_id, route_id, bus_id)] = hold_time

        return stop_bus_hold_time

    def _extract_state(self, snapshot: Snapshot, route_id: str, bus_id: str,
                       stop_id: str) -> Tuple[List[float], bool]:
        """
        Extract rich state representation

        State components:
        0. epsilon_arrival (normalized): current bus's schedule deviation
        1. epsilon_last (normalized): previous bus's schedule deviation
        2. beta: passenger arrival rate / boarding rate
        3. normalized_headway: current headway / scheduled headway
        4. normalized_forward_spacing
        5. normalized_backward_spacing

        Returns:
            state: list of state values
            is_valid: whether the state is valid (no inf values)
        """
        try:
            # Get epsilon values
            epsilon_arrival, epsilon_rtd = snapshot.get_bus_epsilon(route_id, bus_id, stop_id)
            last_epsilon_arrival, last_epsilon_rtd = snapshot.get_stop_epsilon(route_id, stop_id, bus_id)

            # Choose based on base_type
            if self._base_type == 'arrival':
                epsilon_curr = epsilon_arrival
                epsilon_last = last_epsilon_arrival
            else:
                epsilon_curr = epsilon_rtd
                epsilon_last = last_epsilon_rtd

            # Normalize epsilon
            epsilon_curr_norm = epsilon_curr / self._epsilon_scale
            epsilon_last_norm = epsilon_last / self._epsilon_scale

            # Get beta (passenger dynamics)
            stop_boarding_rate = self._blueprint.route_schema.route_details_by_id[route_id].boarding_rate
            arrival_rate = self._route_stop_arrival_rate[route_id][stop_id]
            beta = arrival_rate / stop_boarding_rate[stop_id] if stop_boarding_rate[stop_id] > 0 else 0

            # Get headway
            stop_snapshots = snapshot.stop_snapshots
            arrival_times = stop_snapshots[stop_id].route_arrival_time_seq[route_id]
            if len(arrival_times) >= 2:
                headway = arrival_times[-1] - arrival_times[-2]
                normalized_headway = headway / self._route_schedule[route_id]
            else:
                normalized_headway = 1.0

            # Get spacing (from parent class method)
            _, forward_spacing, _, backward_spacing = self.extract_local_info_from_snapshot(
                bus_id, snapshot, ['spacing'])

            # Check for invalid spacing
            if forward_spacing == float('inf') or backward_spacing == float('inf'):
                return [0.0] * self._state_size, False

            # Normalize spacing
            forward_spacing_norm = forward_spacing / self._spacing_scale
            backward_spacing_norm = backward_spacing / self._spacing_scale

            state = [
                np.clip(epsilon_curr_norm, -5, 5),
                np.clip(epsilon_last_norm, -5, 5),
                np.clip(beta, 0, 2),
                np.clip(normalized_headway, 0, 3),
                np.clip(forward_spacing_norm, 0, 5),
                np.clip(backward_spacing_norm, 0, 5),
            ]

            return state, True

        except Exception as e:
            # Return zero state if extraction fails
            return [0.0] * self._state_size, False

    def _calculate_reward(self, snapshot: Snapshot, epsilon_curr: float, epsilon_last: float,
                          forward_spacing: float, backward_spacing: float,
                          stop_id: str, acting_bus: Tuple[str, str],
                          action: float, prev_epsilon: Optional[float] = None) -> float:
        """
        Calculate improvement-based reward with positive baseline

        Key changes from original:
        1. Rewards IMPROVEMENT in schedule deviation (not absolute value)
        2. Adds positive baseline to prevent Q → -∞
        3. Clips rewards to prevent extreme values

        Components:
        1. Epsilon improvement: reward for reducing schedule deviation
        2. Spacing balance penalty: -|forward - backward| (smaller is better)
        3. Action penalty: small cost for holding (regularization)
        4. Positive baseline: ensures some positive reward signal
        """
        # 1. Improvement-based epsilon reward (primary objective)
        # Reward getting closer to schedule (reducing |epsilon|)
        if prev_epsilon is not None:
        # if False:
            # Positive if |prev_epsilon| > |epsilon_curr| (improvement)
            epsilon_improvement = (abs(prev_epsilon) - abs(epsilon_curr)) / self._epsilon_scale
        else:
            # For first observation, use negative of absolute deviation (but smaller weight)
            epsilon_improvement = -abs(epsilon_curr) / self._epsilon_scale * 0.5

        # 2. Spacing balance penalty (secondary objective)
        # Penalize unbalanced spacing between forward and backward buses
        spacing_diff = abs(forward_spacing - backward_spacing) / self._spacing_scale
        spacing_penalty = -spacing_diff

        # 3. Headway penalty (secondary objective)
        # Penalize deviation from scheduled headway
        # stop_snapshots = snapshot.stop_snapshots
        route_id = acting_bus[0]
        # # all the buses' arrival time at this stop
        # current_stop_arrival_info = stop_snapshots[stop_id].route_arrival_time_seq[acting_bus[0]]
        # # current_stop_departure_info = holder_snapshots.route_stop_departure_time_seq[acting_bus[0]][stop_id]
        # # the pervious bus's arrival time at this stop
        # pervious_bus_arrival_time = current_stop_arrival_info[-2]
        # # the current bus's arrival time at this stop
        # current_bus_arrival_time = current_stop_arrival_info[-1]
        # headway = current_bus_arrival_time - pervious_bus_arrival_time
        # normalized_headway = headway / self._H

        # headway_penalty = -abs((self._H - headway) / self._H)

        sched_H = self._H

        arr = snapshot.stop_snapshots[stop_id].route_arrival_time_seq[route_id]
        if len(arr) >= 2:
            headway = arr[-1] - arr[-2]
            headway_penalty = -abs((sched_H - headway) / max(sched_H, 1e-6))
        else:
            headway_penalty = 0.0  # or a small penalty; but avoid indexing crash

        # 3. Action penalty (regularization)
        # Small cost for holding to discourage excessive intervention
        action_penalty = -action

        # 4. Combine with positive baseline
        reward = (
            # self._reward_baseline +
            headway_penalty +
            self._w_epsilon * epsilon_improvement +
            self._w_spacing * spacing_penalty +
            self._w_action * action_penalty
            )

        # Clip reward to prevent extreme values
        reward = np.clip(reward, -self._reward_clip, self._reward_clip)

        return reward

    def _normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using running statistics (Welford's algorithm)

        This helps stabilize training by keeping rewards in a consistent range
        regardless of the absolute scale of the reward function.
        """
        if not self._use_reward_norm:
            return reward

        self._reward_count += 1

        # Update running mean and variance (Welford's online algorithm)
        delta = reward - self._reward_running_mean
        self._reward_running_mean += delta / self._reward_count

        delta2 = reward - self._reward_running_mean
        self._reward_running_var += delta * delta2

        # Calculate standard deviation (with minimum to prevent division by zero)
        if self._reward_count > 1:
            std = np.sqrt(self._reward_running_var / (self._reward_count - 1))
        else:
            std = 1.0
        std = max(std, 0.01)  # Prevent division by very small numbers

        # Normalize: zero mean, unit variance
        normalized_reward = (reward - self._reward_running_mean) / std

        # Clip normalized reward to prevent extreme values
        return np.clip(normalized_reward, -5.0, 5.0)

    def calculate_hold_time(self, snapshot: Snapshot) -> Dict[Tuple[str, str, str], float]:
        """Main decision function called by simulator"""
        stop_bus_hold_time = {}

        for (stop_id, route_id, bus_id) in snapshot.holder_snapshot.action_buses:
            if not snapshot.bus_snapshots[(route_id, bus_id)].is_need_to_hold:
                stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0
                continue

            # Extract state
            state, is_valid = self._extract_state(snapshot, route_id, bus_id, stop_id)

            if not is_valid:
                action, hold_time = 0.0, 0.0
                reward = None
            else:
                # Get action from policy
                action, hold_time = self.infer(state)

                # Calculate reward for training
                _, forward_spacing, _, backward_spacing = self.extract_local_info_from_snapshot(
                    bus_id, snapshot, ['spacing'])

                epsilon_arrival, epsilon_rtd = snapshot.get_bus_epsilon(route_id, bus_id, stop_id)
                last_epsilon_arrival, last_epsilon_rtd = snapshot.get_stop_epsilon(route_id, stop_id, bus_id)
                epsilon_curr_raw = epsilon_arrival if self._base_type == "arrival" else epsilon_rtd
                epsilon_last_raw = last_epsilon_arrival if self._base_type == "arrival" else last_epsilon_rtd


                # Get previous epsilon for this bus (for improvement-based reward)
                bus_key = (route_id, bus_id)
                prev_epsilon = self._bus_prev_epsilon.get(bus_key, None)

                # Calculate improvement-based reward
                reward = self._calculate_reward(
                    snapshot,
                    epsilon_curr_raw, epsilon_last_raw,
                    forward_spacing, backward_spacing,
                    stop_id, (route_id, bus_id),
                    action,
                    prev_epsilon=prev_epsilon
                )

                # Store current epsilon as previous for next time
                self._bus_prev_epsilon[bus_key] = epsilon_curr_raw

                # Apply reward normalization for stable training
                if self._is_train:
                    reward = self._normalize_reward(reward)

                # Record for logging (use unnormalized for interpretability)
                self._record_reward(reward)
                self._record_action(action)

            stop_bus_hold_time[(stop_id, route_id, bus_id)] = hold_time

            # Store transition for training
            if self.is_train:
                sar = SAR(state, action, reward)
                self._bus_stop_sar[(route_id, bus_id)].append((stop_id, sar))
                self._add_event_count += 1

                if self._add_event_count % self._batch_size == 0:
                    self._push_transitions_to_memory()

                self.learn()

            snapshot.record_holding_time(stop_bus_hold_time)

        return stop_bus_hold_time

    def infer(self, state: List[float]) -> Tuple[float, float]:
        """Get action from actor network"""
        state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)

        with torch.no_grad():
            action = self._actor_net(state_tensor)

            if self._is_train:
                noise = np.random.normal(0, self._noise_level)
                action = (action + noise).clamp(0, 1)

            action = float(action.item())

        hold_time = action * self._max_hold_time
        return action, hold_time

    def _push_transitions_to_memory(self):
        """Convert SAR sequences to SARS transitions and push to replay buffer"""
        for (route_id, bus_id), sar_list in self._bus_stop_sar.items():
            if len(sar_list) < 2:
                continue

            for (stop_id, sar), (next_stop_id, next_sar) in zip(sar_list[:-1], sar_list[1:]):
                # Verify consecutive stops
                node_type, prev_stop_id = self._blueprint.get_previous_node(route_id, next_stop_id)
                if node_type == 'terminal' or prev_stop_id != stop_id:
                    continue

                # Skip invalid transitions
                if sar.reward is None or next_sar.reward is None:
                    continue
                if any(s == 0.0 for s in sar.state) and all(s == 0.0 for s in sar.state):
                    continue

                sars = SARS(
                    state=sar.state,
                    action=sar.action,
                    # reward=next_sar.reward,  # Use next step's reward
                    reward=sar.reward,  # Use current step's reward
                    next_state=next_sar.state
                )
                self._memory.append(sars)

        self._bus_stop_sar.clear()

    def learn(self):
        """
        Update actor and critic networks using TD3 algorithm

        TD3 improvements over DDPG:
        1. Twin critics - use minimum Q to prevent overestimation
        2. Delayed policy updates - update actor every N critic updates
        3. Target policy smoothing - add noise to target actions
        """
        if self._add_event_count % self._update_cycle != 0:
            return
        if len(self._memory) < self._batch_size:
            return

        self._actor_net.train()
        self._critic_net_1.train()
        self._critic_net_2.train()
        self._learn_count += 1
        self._critic_update_count += 1

        # Sample batch
        samples = random.sample(self._memory, self._batch_size)

        states = torch.tensor([s.state for s in samples], dtype=torch.float32)
        actions = torch.tensor([s.action for s in samples], dtype=torch.float32).unsqueeze(1)
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor([s.next_state for s in samples], dtype=torch.float32)

        # ===== Update Critics (TD3: Twin Critics) =====
        self._critic_optim.zero_grad()

        # Current Q estimates from both critics
        state_action = torch.cat([states, actions], dim=1)
        current_q1 = self._critic_net_1(state_action)
        current_q2 = self._critic_net_2(state_action)

        # Target Q value with TD3 improvements
        with torch.no_grad():
            # TD3: Target policy smoothing - add clipped noise to target actions
            next_actions = self._target_actor_net(next_states)
            noise = torch.randn_like(next_actions) * self._target_noise
            noise = noise.clamp(-self._noise_clip, self._noise_clip)
            next_actions = (next_actions + noise).clamp(0, 1)

            # TD3: Use minimum of twin critics to prevent overestimation
            next_state_action = torch.cat([next_states, next_actions], dim=1)
            target_q1 = self._target_critic_net_1(next_state_action)
            target_q2 = self._target_critic_net_2(next_state_action)
            target_q = torch.min(target_q1, target_q2)

            target_value = rewards + self._gamma * target_q

        # Critic losses (both critics)
        td_error_1 = current_q1 - target_value
        td_error_2 = current_q2 - target_value
        critic_loss_1 = (td_error_1 ** 2).mean()
        critic_loss_2 = (td_error_2 ** 2).mean()
        critic_loss = critic_loss_1 + critic_loss_2

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._critic_net_1.parameters()) + list(self._critic_net_2.parameters()),
            self._grad_clip)
        self._critic_optim.step()

        # ===== Update Actor (TD3: Delayed Policy Updates) =====
        actor_loss = torch.tensor(0.0)
        if self._critic_update_count % self._policy_delay == 0:
            self._actor_optim.zero_grad()

            # Freeze critics for actor update
            for param in self._critic_net_1.parameters():
                param.requires_grad = False
            for param in self._critic_net_2.parameters():
                param.requires_grad = False

            # Actor loss: maximize Q value (use critic 1)
            predicted_actions = self._actor_net(states)
            state_predicted_action = torch.cat([states, predicted_actions], dim=1)
            actor_loss = -self._critic_net_1(state_predicted_action).mean()

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._actor_net.parameters(), self._grad_clip)
            self._actor_optim.step()

            # Unfreeze critics
            for param in self._critic_net_1.parameters():
                param.requires_grad = True
            for param in self._critic_net_2.parameters():
                param.requires_grad = True

            # ===== Update Target Networks (only when actor updates) =====
            with torch.no_grad():
                for p, p_target in zip(self._actor_net.parameters(),
                                        self._target_actor_net.parameters()):
                    p_target.data.mul_(self._polyak)
                    p_target.data.add_((1 - self._polyak) * p.data)

                for p, p_target in zip(self._critic_net_1.parameters(),
                                        self._target_critic_net_1.parameters()):
                    p_target.data.mul_(self._polyak)
                    p_target.data.add_((1 - self._polyak) * p.data)

                for p, p_target in zip(self._critic_net_2.parameters(),
                                        self._target_critic_net_2.parameters()):
                    p_target.data.mul_(self._polyak)
                    p_target.data.add_((1 - self._polyak) * p.data)

        # ===== Record Metrics =====
        self._record_metric('critic_loss', float(critic_loss.item()))
        self._record_metric('critic_loss_1', float(critic_loss_1.item()))
        self._record_metric('critic_loss_2', float(critic_loss_2.item()))
        self._record_metric('actor_loss', float(actor_loss.item()))
        self._record_metric('Q_value_1', float(current_q1.mean().item()))
        self._record_metric('Q_value_2', float(current_q2.mean().item()))
        self._record_metric('Q_value_min', float(torch.min(current_q1, current_q2).mean().item()))
        self._record_metric('target_Q', float(target_q.mean().item()))
        self._record_metric('td_error', float(td_error_1.abs().mean().item()))
        self._record_metric('batch_reward', float(rewards.mean().item()))

    def reset(self, episode: int):
        """Reset agent for new episode"""
        if self._is_train:
            # Decay noise level
            self._noise_level = max(
                self._min_noise,
                self._decay_rate ** episode * self._init_noise_level
            )
            self._learn_count = 0
            self._critic_update_count = 0

            # Clear previous epsilon tracking for new episode
            self._bus_prev_epsilon.clear()

            # Step learning rate schedulers
            if episode > 0 and episode % 10 == 0:
                self._actor_scheduler.step()
                self._critic_scheduler.step()
                current_actor_lr = self._actor_optim.param_groups[0]['lr']
                current_critic_lr = self._critic_optim.param_groups[0]['lr']
                print(f'Schedule_Aware_DDPG: LR decay - actor_lr={current_actor_lr:.6f}, critic_lr={current_critic_lr:.6f}')

            print(f'Schedule_Aware_DDPG: noise_level = {self._noise_level:.4f}')
            self._clear_episode_metrics()

    def _get_additional_metrics(self) -> Dict[str, float]:
        """Return agent-specific metrics for logging"""
        metrics = {}
        if self._is_train:
            metrics['buffer/size'] = len(self._memory)
            metrics['buffer/capacity'] = self._memory.maxlen
            metrics['buffer/utilization'] = len(self._memory) / self._memory.maxlen
            metrics['explore/noise_level'] = self._noise_level
            metrics['train/learn_count'] = self._learn_count
            metrics['train/critic_update_count'] = self._critic_update_count
            metrics['train/actor_lr'] = self._actor_optim.param_groups[0]['lr']
            metrics['train/critic_lr'] = self._critic_optim.param_groups[0]['lr']
            # Reward normalization stats
            if self._reward_count > 0:
                metrics['reward/running_mean'] = self._reward_running_mean
                std = np.sqrt(self._reward_running_var / max(1, self._reward_count - 1)) if self._reward_count > 1 else 1.0
                metrics['reward/running_std'] = std
        return metrics

    # ===== Save/Load Methods =====

    def save_net(self, path: str) -> None:
        """Simple save for backward compatibility"""
        torch.save(self._actor_net.state_dict(), path)

    def load_net(self, path: str) -> None:
        """Simple load for backward compatibility"""
        self._actor_net.load_state_dict(torch.load(path, map_location='cpu'))

    def _get_model_state_dict(self) -> Dict[str, Any]:
        """Get state dict for all models (TD3 twin critics)"""
        state_dict = {'actor_net': self._actor_net.state_dict()}
        if self._is_train:
            state_dict['critic_net_1'] = self._critic_net_1.state_dict()
            state_dict['critic_net_2'] = self._critic_net_2.state_dict()
        return state_dict

    def _get_optimizer_state_dict(self) -> Dict[str, Any]:
        """Get state dict for all optimizers and schedulers"""
        if not self._is_train:
            return {}
        return {
            'actor_optim': self._actor_optim.state_dict(),
            'critic_optim': self._critic_optim.state_dict(),
            'actor_scheduler': self._actor_scheduler.state_dict(),
            'critic_scheduler': self._critic_scheduler.state_dict(),
        }

    def _get_target_model_state_dict(self) -> Dict[str, Any]:
        """Get state dict for all target networks (TD3 twin critics)"""
        if not self._is_train:
            return {}
        return {
            'target_actor_net': self._target_actor_net.state_dict(),
            'target_critic_net_1': self._target_critic_net_1.state_dict(),
            'target_critic_net_2': self._target_critic_net_2.state_dict(),
        }

    def _load_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict for all models"""
        self._actor_net.load_state_dict(state_dict['actor_net'])
        if self._is_train:
            # Handle both old (single critic) and new (twin critics) checkpoints
            if 'critic_net_1' in state_dict:
                self._critic_net_1.load_state_dict(state_dict['critic_net_1'])
                self._critic_net_2.load_state_dict(state_dict['critic_net_2'])
            elif 'critic_net' in state_dict:
                # Backward compatibility: load old single critic into both
                self._critic_net_1.load_state_dict(state_dict['critic_net'])
                self._critic_net_2.load_state_dict(state_dict['critic_net'])

    def _load_optimizer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict for all optimizers and schedulers"""
        if not self._is_train:
            return
        if 'actor_optim' in state_dict:
            self._actor_optim.load_state_dict(state_dict['actor_optim'])
        if 'critic_optim' in state_dict:
            self._critic_optim.load_state_dict(state_dict['critic_optim'])
        if 'actor_scheduler' in state_dict:
            self._actor_scheduler.load_state_dict(state_dict['actor_scheduler'])
        if 'critic_scheduler' in state_dict:
            self._critic_scheduler.load_state_dict(state_dict['critic_scheduler'])

    def _load_target_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict for all target networks"""
        if not self._is_train:
            return
        if 'target_actor_net' in state_dict:
            self._target_actor_net.load_state_dict(state_dict['target_actor_net'])
        # Handle both old and new checkpoint formats
        if 'target_critic_net_1' in state_dict:
            self._target_critic_net_1.load_state_dict(state_dict['target_critic_net_1'])
            self._target_critic_net_2.load_state_dict(state_dict['target_critic_net_2'])
        elif 'target_critic_net' in state_dict:
            # Backward compatibility
            self._target_critic_net_1.load_state_dict(state_dict['target_critic_net'])
            self._target_critic_net_2.load_state_dict(state_dict['target_critic_net'])
