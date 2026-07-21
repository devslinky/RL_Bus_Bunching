"""
Rich_PPO: Proximal Policy Optimization agent for bus holding control

This agent uses PPO (Proximal Policy Optimization) for learning bus holding decisions.
PPO is an on-policy algorithm that provides stable training through clipped objective
and typically performs well with continuous action spaces.

Key Features:
- Actor-Critic architecture with separate policy and value networks
- Clipped surrogate objective for stable policy updates
- GAE (Generalized Advantage Estimation) for variance reduction
- Rich state representation including headway, epsilon, and passenger info

Reward Components:
- Headway deviation penalty: -|H - headway| / H
- Passenger wait time penalty (optional)
- Action penalty to discourage excessive holding
"""

from copy import deepcopy
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, List
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from simulator.snapshot import Snapshot
from setup.blueprint import Blueprint
from .rl_agent import RLAgent


@dataclass
class Transition:
    """Single transition for PPO training."""
    state: List[float]
    action: float
    log_prob: float
    reward: float
    value: float
    done: bool = False


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    states: List[List[float]] = field(default_factory=list)
    actions: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(self, state, action, log_prob, reward, value, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


class PPOActorNet(nn.Module):
    """
    Actor network for PPO that outputs a Gaussian distribution over actions.

    Outputs mean and log_std for the action distribution.
    Action is bounded to [0, 1] using tanh squashing.
    """

    def __init__(self, state_size: int, hidden_size: Tuple[int, ...] = (64, 64)):
        super(PPOActorNet, self).__init__()

        layers = []
        prev_dim = state_size
        for hidden_dim in hidden_size:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, 1)
        # Learnable log_std parameter
        self.log_std = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for action head
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: Mean of the action distribution
            std: Standard deviation of the action distribution
        """
        features = self.feature_net(state)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Sample an action from the policy.

        Args:
            state: Current state tensor
            deterministic: If True, return mean action (no sampling)

        Returns:
            action: Sampled action in [0, 1]
            log_prob: Log probability of the action
        """
        mean, std = self.forward(state)

        if deterministic:
            action = torch.sigmoid(mean)
            # Log prob for deterministic action (approximation)
            log_prob = torch.zeros_like(action)
        else:
            dist = Normal(mean, std)
            # Sample in unbounded space
            raw_action = dist.rsample()
            log_prob = dist.log_prob(raw_action)

            # Squash to [0, 1] using sigmoid
            action = torch.sigmoid(raw_action)

            # Adjust log_prob for the squashing (Jacobian correction)
            log_prob = log_prob - torch.log(action * (1 - action) + 1e-6)

        return action, log_prob.sum(dim=-1)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate log probability and entropy of actions.

        Args:
            states: Batch of states
            actions: Batch of actions (already in [0, 1])

        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of the action distribution
        """
        mean, std = self.forward(states)

        # Convert actions back to unbounded space
        # Clamp to avoid numerical issues at boundaries
        actions_clamped = actions.clamp(1e-6, 1 - 1e-6)
        raw_actions = torch.log(actions_clamped / (1 - actions_clamped))  # inverse sigmoid

        dist = Normal(mean, std)
        log_probs = dist.log_prob(raw_actions)

        # Jacobian correction for sigmoid squashing
        log_probs = log_probs - torch.log(actions_clamped * (1 - actions_clamped) + 1e-6)

        entropy = dist.entropy()

        return log_probs.sum(dim=-1), entropy.sum(dim=-1)


class PPOCriticNet(nn.Module):
    """
    Critic (Value) network for PPO.

    Estimates the value function V(s) for a given state.
    """

    def __init__(self, state_size: int, hidden_size: Tuple[int, ...] = (64, 64)):
        super(PPOCriticNet, self).__init__()

        layers = []
        prev_dim = state_size
        for hidden_dim in hidden_size:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            value: Estimated value of the state
        """
        features = self.feature_net(state)
        value = self.value_head(features)
        return value


class Rich_PPO(RLAgent):
    """
    PPO agent with rich state representation for bus holding control.

    State Features:
    - normalized_headway: headway / schedule_headway
    - epsilon_arrival: schedule deviation at arrival
    - pax_waiting: number of passengers waiting (optional)
    - forward_spacing: distance to forward bus (optional)
    - backward_spacing: distance to backward bus (optional)

    PPO Hyperparameters:
    - clip_ratio: PPO clipping parameter (default: 0.2)
    - value_coef: Value loss coefficient (default: 0.5)
    - entropy_coef: Entropy bonus coefficient (default: 0.01)
    - gae_lambda: GAE lambda for advantage estimation (default: 0.95)
    - n_epochs: Number of optimization epochs per update (default: 10)
    """

    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint,
                 run_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(agent_config, blueprint)

        self._state_size = agent_config['state_size']
        self._blueprint = blueprint
        self._max_hold_time = agent_config['max_hold_time']
        self._H = agent_config['schedule_headway']

        # Reward weights
        self._w_action = agent_config.get('w_action', 0.03)
        self._w_headway = agent_config.get('w_headway', 1.0)
        self._w_pax_wait = agent_config.get('w_pax_wait', 0.0)
        self._pax_wait_scale = agent_config.get('pax_wait_scale', 100.0)

        # PPO hyperparameters
        self._clip_ratio = agent_config.get('clip_ratio', 0.2)
        self._value_coef = agent_config.get('value_coef', 0.5)
        self._entropy_coef = agent_config.get('entropy_coef', 0.01)
        self._gae_lambda = agent_config.get('gae_lambda', 0.95)
        self._gamma = agent_config.get('gamma', 0.99)
        self._n_epochs = agent_config.get('n_epochs', 10)
        self._batch_size = agent_config.get('batch_size', 64)
        self._update_freq = agent_config.get('update_freq', 256)  # Update after N transitions
        self._max_grad_norm = agent_config.get('max_grad_norm', 0.5)

        # Networks
        hidden_size = tuple(agent_config.get('hidden_size', [64, 64]))
        self._actor_net = PPOActorNet(self._state_size, hidden_size)
        self._critic_net = PPOCriticNet(self._state_size, hidden_size)

        # Optimizers
        self._actor_optim = torch.optim.Adam(
            self._actor_net.parameters(), lr=agent_config.get('actor_lr', 3e-4))
        self._critic_optim = torch.optim.Adam(
            self._critic_net.parameters(), lr=agent_config.get('critic_lr', 3e-4))

        # Rollout buffer for on-policy data
        self._rollout_buffer = RolloutBuffer()

        # Track transitions for building (state, action, reward, next_state) pairs
        self._bus_stop_transitions: Dict[Tuple[str, str], List[Tuple[str, Dict]]] = defaultdict(list)
        self._pending_transitions: List[Dict] = []

        self._learn_count = 0
        self._total_transitions = 0

    def reset(self, episode: int):
        """Reset at the start of each episode."""
        if self._is_train:
            # Process any remaining transitions before clearing
            self._process_pending_transitions()
            # Perform final update if buffer has data
            if len(self._rollout_buffer) >= self._batch_size:
                self._update()
            self._rollout_buffer.clear()
            self._bus_stop_transitions.clear()
            self._pending_transitions.clear()
            self._learn_count = 0
            self._clear_episode_metrics()

    def _get_additional_metrics(self) -> Dict[str, float]:
        """Return PPO-specific metrics."""
        metrics = {}
        if self._is_train:
            metrics['buffer/size'] = len(self._rollout_buffer)
            metrics['train/learn_count'] = self._learn_count
            metrics['train/total_transitions'] = self._total_transitions
        return metrics

    def _transform_snapshot_to_state(self, snapshot: Snapshot,
                                      acting_bus: Tuple[str, str],
                                      stop_id: str) -> List[float]:
        """
        Transform snapshot to state vector.

        State: [normalized_headway, epsilon_arrival, ...]
        """
        stop_snapshots = snapshot.stop_snapshots
        route_id, bus_id = acting_bus

        # Get headway information
        current_stop_arrival_info = stop_snapshots[stop_id].route_arrival_time_seq[route_id]
        if len(current_stop_arrival_info) >= 2:
            previous_bus_arrival_time = current_stop_arrival_info[-2]
            current_bus_arrival_time = current_stop_arrival_info[-1]
            headway = current_bus_arrival_time - previous_bus_arrival_time
        else:
            headway = self._H

        normalized_headway = headway / self._H

        # Get schedule deviation
        epsilon_arrival, _ = snapshot.get_bus_epsilon(route_id, bus_id, stop_id)
        normalized_epsilon = epsilon_arrival / self._H

        state = [normalized_headway, normalized_epsilon]

        # Add additional features based on state_size
        if self._state_size > 2:
            # Passenger count at stop
            pax_waiting = stop_snapshots[stop_id].pax_num
            normalized_pax = pax_waiting / self._pax_wait_scale
            state.append(normalized_pax)

        if self._state_size > 3:
            # Forward and backward spacing
            _, forward_spacing, _, backward_spacing = self.extract_local_info_from_snapshot(
                bus_id, snapshot, ['spacing'])

            # Normalize spacing (use route length or large value)
            max_spacing = 10000.0  # meters
            norm_forward = min(forward_spacing, max_spacing) / max_spacing if forward_spacing != float('inf') else 1.0
            norm_backward = min(backward_spacing, max_spacing) / max_spacing if backward_spacing != float('inf') else 1.0
            state.extend([norm_forward, norm_backward])

        return state[:self._state_size]

    def _calculate_reward(self, snapshot: Snapshot, stop_id: str,
                          route_id: str, action: float) -> float:
        """
        Calculate reward for a holding action.

        Components:
        - Headway deviation penalty
        - Passenger wait penalty (optional)
        - Action penalty
        """
        stop_snapshots = snapshot.stop_snapshots

        # Get headway
        current_stop_arrival_info = stop_snapshots[stop_id].route_arrival_time_seq[route_id]
        if len(current_stop_arrival_info) >= 2:
            previous_bus_arrival_time = current_stop_arrival_info[-2]
            current_bus_arrival_time = current_stop_arrival_info[-1]
            headway = current_bus_arrival_time - previous_bus_arrival_time
        else:
            headway = self._H

        # Headway deviation reward (negative of absolute deviation)
        headway_reward = -abs((self._H - headway) / self._H)

        # Passenger wait penalty
        pax_penalty = 0.0
        if self._w_pax_wait > 0:
            pax_waiting = stop_snapshots[stop_id].pax_num
            pax_penalty = pax_waiting * (headway / self._H) / self._pax_wait_scale

        # Action penalty
        action_penalty = self._w_action * action

        # Combined reward
        reward = (
            self._w_headway * headway_reward
            - self._w_pax_wait * pax_penalty
            - action_penalty
        )

        return reward

    def calculate_hold_time(self, snapshot: Snapshot) -> Dict[Tuple[str, str, str], float]:
        """
        Main method called by simulator to get holding decisions.
        """
        stop_bus_hold_time = {}

        for (stop_id, route_id, bus_id) in snapshot.holder_snapshot.action_buses:
            # Skip buses that don't need holding
            if not snapshot.bus_snapshots[(route_id, bus_id)].is_need_to_hold:
                stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0
                continue

            # Check for valid spacing
            _, forward_spacing, _, backward_spacing = self.extract_local_info_from_snapshot(
                bus_id, snapshot, ['spacing'])

            if forward_spacing == float('inf') or backward_spacing == float('inf'):
                # Not enough buses to compute spacing - skip learning
                stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0
                continue

            # Get state
            state = self._transform_snapshot_to_state(snapshot, (route_id, bus_id), stop_id)

            # Get action from policy
            action, log_prob, value, hold_time = self.infer(state)

            # Calculate reward
            reward = self._calculate_reward(snapshot, stop_id, route_id, action)

            stop_bus_hold_time[(stop_id, route_id, bus_id)] = hold_time

            # Store transition for training
            if self._is_train:
                self._record_reward(reward)
                self._record_action(action)

                # Add to rollout buffer
                self._rollout_buffer.add(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    done=False
                )
                self._total_transitions += 1

                # Update policy when buffer is full
                if len(self._rollout_buffer) >= self._update_freq:
                    self._update()
                    self._rollout_buffer.clear()

            snapshot.record_holding_time(stop_bus_hold_time)

        return stop_bus_hold_time

    def infer(self, state: List[float]) -> Tuple[float, float, float, float]:
        """
        Infer action from state.

        Returns:
            action: Action in [0, 1]
            log_prob: Log probability of action
            value: Estimated state value
            hold_time: Actual hold time in seconds
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        self._actor_net.eval()
        self._critic_net.eval()

        with torch.no_grad():
            # Get action
            if self._is_train:
                action, log_prob = self._actor_net.get_action(state_tensor, deterministic=False)
            else:
                action, log_prob = self._actor_net.get_action(state_tensor, deterministic=True)

            # Get value estimate
            value = self._critic_net(state_tensor)

        action = float(action.squeeze())
        log_prob = float(log_prob)
        value = float(value.squeeze())
        hold_time = action * self._max_hold_time

        return action, log_prob, value, hold_time

    def _process_pending_transitions(self):
        """Process any pending transitions before episode end."""
        pass  # PPO uses rollout buffer directly, no pending transitions

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                     dones: torch.Tensor, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Tensor of rewards
            values: Tensor of value estimates
            dones: Tensor of done flags
            last_value: Value estimate for the last state

        Returns:
            advantages: GAE advantages
            returns: Discounted returns (advantages + values)
        """
        batch_size = len(rewards)
        advantages = torch.zeros(batch_size)
        last_gae = 0.0

        # Append last value for bootstrapping
        values_extended = torch.cat([values, torch.tensor([last_value])])

        for t in reversed(range(batch_size)):
            if dones[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = values_extended[t + 1]

            delta = rewards[t] + self._gamma * next_value - values[t]
            last_gae = delta + self._gamma * self._gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def _update(self):
        """
        Perform PPO update using collected rollout data.
        """
        if len(self._rollout_buffer) < self._batch_size:
            return

        self._actor_net.train()
        self._critic_net.train()
        self._learn_count += 1

        # Convert buffer to tensors
        states = torch.tensor(self._rollout_buffer.states, dtype=torch.float32)
        actions = torch.tensor(self._rollout_buffer.actions, dtype=torch.float32).unsqueeze(-1)
        old_log_probs = torch.tensor(self._rollout_buffer.log_probs, dtype=torch.float32)
        rewards = torch.tensor(self._rollout_buffer.rewards, dtype=torch.float32)
        old_values = torch.tensor(self._rollout_buffer.values, dtype=torch.float32)
        dones = torch.tensor(self._rollout_buffer.dones, dtype=torch.float32)

        # Compute advantages using GAE
        with torch.no_grad():
            # Get value of last state for bootstrapping
            last_state = torch.tensor(self._rollout_buffer.states[-1], dtype=torch.float32).unsqueeze(0)
            last_value = float(self._critic_net(last_state).squeeze())

        advantages, returns = self._compute_gae(rewards, old_values, dones, last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        batch_size = len(states)
        indices = np.arange(batch_size)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self._n_epochs):
            # Shuffle data
            np.random.shuffle(indices)

            for start in range(0, batch_size, self._batch_size):
                end = start + self._batch_size
                if end > batch_size:
                    continue

                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions under current policy
                new_log_probs, entropy = self._actor_net.evaluate_actions(batch_states, batch_actions)
                new_values = self._critic_net(batch_states).squeeze()

                # Compute probability ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    actor_loss
                    + self._value_coef * value_loss
                    + self._entropy_coef * entropy_loss
                )

                # Update networks
                self._actor_optim.zero_grad()
                self._critic_optim.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self._actor_net.parameters(), self._max_grad_norm)
                nn.utils.clip_grad_norm_(self._critic_net.parameters(), self._max_grad_norm)

                self._actor_optim.step()
                self._critic_optim.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Record metrics
        if n_updates > 0:
            self._record_metric('actor_loss', total_actor_loss / n_updates)
            self._record_metric('critic_loss', total_critic_loss / n_updates)
            self._record_metric('entropy', total_entropy / n_updates)
            self._record_metric('advantage_mean', float(advantages.mean()))
            self._record_metric('advantage_std', float(advantages.std()))
            self._record_metric('return_mean', float(returns.mean()))

    def save_net(self, path: str) -> None:
        """Save actor network (backward compatible)."""
        torch.save(self._actor_net.state_dict(), path)

    def load_net(self, path: str) -> None:
        """Load actor network (backward compatible)."""
        self._actor_net.load_state_dict(torch.load(path, map_location='cpu'))

    def _get_model_state_dict(self) -> Dict[str, Any]:
        """Get all model state dicts for checkpoint."""
        state_dict = {
            'actor_net': self._actor_net.state_dict(),
        }
        if self._is_train:
            state_dict['critic_net'] = self._critic_net.state_dict()
        return state_dict

    def _get_optimizer_state_dict(self) -> Dict[str, Any]:
        """Get all optimizer state dicts for checkpoint."""
        if not self._is_train:
            return {}
        return {
            'actor_optim': self._actor_optim.state_dict(),
            'critic_optim': self._critic_optim.state_dict(),
        }

    def _get_target_model_state_dict(self) -> Dict[str, Any]:
        """PPO doesn't use target networks."""
        return {}

    def _load_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load model state dicts from checkpoint."""
        self._actor_net.load_state_dict(state_dict['actor_net'])
        if self._is_train and 'critic_net' in state_dict:
            self._critic_net.load_state_dict(state_dict['critic_net'])

    def _load_optimizer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dicts from checkpoint."""
        if not self._is_train:
            return
        if 'actor_optim' in state_dict:
            self._actor_optim.load_state_dict(state_dict['actor_optim'])
        if 'critic_optim' in state_dict:
            self._critic_optim.load_state_dict(state_dict['critic_optim'])

    def _load_target_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """PPO doesn't use target networks."""
        pass
