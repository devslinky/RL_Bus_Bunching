"""
Pax_Wait_Time_DDPG: DDPG that optimizes for actual passenger wait time

This agent extends the DDPG framework by computing passenger wait time penalties
from actual individual passenger arrival times rather than headway-based estimates.

For each passenger waiting at a stop, the wait time is: current_time - arrival_time.
The total wait time penalty is the sum across all waiting passengers, normalized by a scale factor.

Reward Components:
- Headway deviation penalty (weighted by w_headway)
- Passenger wait time penalty (weighted by w_pax_wait): sum of actual per-passenger wait times
- Action penalty (weighted by w)
"""

from copy import deepcopy
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import random
import numpy as np
import torch
import torch.nn as nn
import joblib

from simulator.snapshot import Snapshot
from setup.blueprint import Blueprint
from .rl_agent import RLAgent
from .net import Actor_Net, Critic_Net


@dataclass(frozen=True)
class SAR:
    state: List[float]
    action: float
    reward: Optional[float]


@dataclass(frozen=True)
class SARS:
    state: List[float]
    action: float
    reward: Optional[float]
    next_state: List[float]


class Pax_Wait_Time_DDPG_Pax_Penal(RLAgent):
    """
    DDPG agent with Behavior Cloning initialization that optimizes for passenger wait time.

    Key Differences from Naive_DDPG_BC:
    1. Reward includes passenger wait time penalty based on:
       - Number of passengers waiting at stop
       - Headway deviation (longer gaps = more waiting passengers)
       - Expected passenger accumulation rate

    2. Additional reward weights:
       - w_pax_wait: Weight for passenger wait penalty
       - w_headway: Weight for headway deviation penalty
    """

    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint, run_config: Dict[str, Any]) -> None:
        super().__init__(agent_config, blueprint)

        self._state_size = agent_config['state_size']
        self._blueprint = blueprint
        self._actor_net = Actor_Net(
            state_size=agent_config['state_size'], hidde_size=tuple(agent_config['hidden_size']))

        self._max_hold_time = agent_config['max_hold_time']
        self._H = agent_config['schedule_headway']

        # Original reward weights
        self._w = agent_config['w']  # Action penalty weight

        # NEW: Passenger wait time reward weights
        self._w_pax_wait = agent_config.get('w_pax_wait', 0.1)  # Passenger wait penalty weight
        self._w_headway = agent_config.get('w_headway', 0.9)    # Headway deviation weight
        self._pax_wait_scale = agent_config.get('pax_wait_scale', 100.0)  # Normalization scale

        # Training networks
        self._critic_net = Critic_Net(
            state_size=agent_config['state_size'], hidde_size=tuple(agent_config['hidden_size']))
        self._target_actor_net = deepcopy(self._actor_net)
        self._target_critic_net = deepcopy(self._critic_net)

        for param in self._target_actor_net.parameters():
            param.requires_grad = False
        for param in self._target_critic_net.parameters():
            param.requires_grad = False
        self._actor_optim = torch.optim.Adam(
            self._actor_net.parameters(), lr=agent_config['actor_lr'])
        self._critic_optim = torch.optim.Adam(
            self._critic_net.parameters(), lr=agent_config['critic_lr'])

        self._gamma = agent_config['gamma']
        self._polya = agent_config['polya']
        self._memory = deque(maxlen=agent_config['memory_size'])

        self._bus_stop_sar: Dict[Tuple[str, str], List[Tuple[str, SAR]]] = defaultdict(list)
        self._add_event_count = 0
        self._update_cycle = agent_config['update_cycle']
        self._batch_size = agent_config['batch_size']
        self._init_noise_level = agent_config['init_noise_level']
        self._decay_rate = agent_config['decay_rate']
        self._noise_level = self._init_noise_level
        self._learn_count = 0
        # ── NEW: Asymmetric headway penalty params (ratio x = headway / H) ──
        # Base:  -(w_neg * relu(-log x)^2 + w_pos * relu(log x)^2)
        self._hw_w_neg = agent_config.get('hw_w_neg', 3.0)   # bunching side weight
        self._hw_w_pos = agent_config.get('hw_w_pos', 1.0)   # gapping side weight

        # Barrier: -lam * softplus((beta - x)/tau)  (activates when x < beta)
        self._hw_beta = agent_config.get('hw_beta', 0.5)     # bunching threshold ratio
        self._hw_lam = agent_config.get('hw_lam', 3.0)       # barrier magnitude
        self._hw_tau = agent_config.get('hw_tau', 0.05)      # barrier smoothness

        # Numerical safety + optional clipping
        self._hw_eps = agent_config.get('hw_eps', 1e-6)
        self._hw_clip_min = agent_config.get('hw_clip_min', None)  # e.g. -5.0 or None

    @staticmethod
    def _softplus(z: float) -> float:
        # numerically stable softplus
        # softplus(z) = log(1 + exp(z))
        if z > 50:
            return z
        if z < -50:
            return 0.0
        return float(np.log1p(np.exp(z)))

    def _headway_penalty(self, headway: float) -> float:
        """
        Asymmetric, smooth headway penalty (negative; 0 is best at headway==H).
        Uses ratio x = headway/H:
          base = -(w_neg*relu(-log x)^2 + w_pos*relu(log x)^2)
          barrier = -lam * softplus((beta - x)/tau)
        """
        x = max(headway / self._H, self._hw_eps)   # ratio
        e = float(np.log(x))                       # log-ratio error

        neg = max(-e, 0.0)  # bunching side (x<1)
        pos = max(e, 0.0)   # gapping side  (x>1)

        base = -(self._hw_w_neg * (neg ** 2) + self._hw_w_pos * (pos ** 2))

        # Smooth "stay out of bunching" barrier when x < beta
        barrier = -self._hw_lam * self._softplus((self._hw_beta - x) / self._hw_tau)

        pen = base + barrier

        # Optional: clip only the headway term lower bound (not required)
        if self._hw_clip_min is not None:
            pen = max(pen, float(self._hw_clip_min))

        return pen

    def reset(self, episode: int):
        if self._is_train:
            self._push_transitions_to_memory()
            self._noise_level = self._decay_rate ** episode * self._init_noise_level
            self._learn_count = 0
            print('noise level:', self._noise_level)
            self._clear_episode_metrics()

    def _get_additional_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self._is_train:
            metrics['buffer/size'] = len(self._memory)
            metrics['buffer/capacity'] = self._memory.maxlen
            metrics['buffer/utilization'] = len(self._memory) / self._memory.maxlen
            metrics['explore/noise_level'] = self._noise_level
            metrics['train/learn_count'] = self._learn_count
        return metrics

    def _calculate_pax_wait_penalty(self, snapshot: Snapshot, stop_id: str) -> float:
        """
        Calculate passenger wait time penalty based on actual individual wait times.

        For each passenger waiting at the stop, the wait time is computed as:
            wait_time = current_time - passenger_arrival_time

        The penalty is the total wait time across all passengers, normalized
        by the scale factor.

        Args:
            snapshot: Current system snapshot
            stop_id: Current stop ID

        Returns:
            Normalized passenger wait penalty (negative, lower = worse)
        """
        t = snapshot.t
        pax_arrival_times = snapshot.stop_snapshots[stop_id].pax_arrival_times

        if len(pax_arrival_times) == 0:
            return 0.0

        # Total wait time: sum of (current_time - arrival_time) for each passenger
        total_wait_time = sum(t - arr_t for arr_t in pax_arrival_times) / self._H 

        # Normalize
        normalized_penalty = -total_wait_time / self._pax_wait_scale

        return normalized_penalty

    def _transform_snapshot_to_SR(self, snapshot: Snapshot, acting_bus: Tuple[str, str],
                                   stop_id: str) -> Tuple[List[float], float]:
        """
        Transform snapshot to state and reward with passenger wait time penalty.

        State: [normalized_headway]
        Reward: w_headway * headway_reward + w_pax_wait * pax_wait_penalty
               where pax_wait_penalty = -sum(t - arrival_time) / scale
        """
        stop_snapshots = snapshot.stop_snapshots
        route_id, bus_id = acting_bus

        current_stop_arrival_info = stop_snapshots[stop_id].route_arrival_time_seq[acting_bus[0]]
        pervious_bus_arrival_time = current_stop_arrival_info[-2]
        current_bus_arrival_time = current_stop_arrival_info[-1]
        headway = current_bus_arrival_time - pervious_bus_arrival_time
        normalized_headway = headway / self._H

        H = self._H

        # Original headway-based reward
        # headway_reward = -abs((H - headway) / H)
        headway_reward = self._headway_penalty(headway)

        # Passenger wait time penalty based on actual pax wait times
        pax_wait_penalty = self._calculate_pax_wait_penalty(snapshot, stop_id)
        
        # print(pax_wait_penalty)
        

        # print('Headway:', headway, 'Pax Wait Penalty:', pax_wait_penalty)
        # xxx

        # Combined reward
        reward = (
            self._w_headway * headway_reward +
            self._w_pax_wait * pax_wait_penalty
        )

        state = [normalized_headway]
        # scalered_state = SCALER_OBSERVATIONS.transform([state])[0].tolist()
        return state, reward

    def _push_transitions_to_memory(self):
        for (route_id, bus_id), sar_list in self._bus_stop_sar.items():
            if len(sar_list) > 1:
                for (stop_id, sar), (next_stop_id, next_sar) in zip(sar_list[0:-1], sar_list[1:]):
                    node_type, found_prev_stop_id = self._blueprint.get_previous_node(
                        route_id, next_stop_id)
                    if node_type == 'terminal':
                        continue
                    if found_prev_stop_id == stop_id:
                        state = sar.state
                        action = sar.action
                        reward = next_sar.reward
                        next_state = next_sar.state

                        if any(var is None for var in [state, action, reward, next_state]):
                            continue
                        else:
                            reward -= self._w * action

                        sars = SARS(state, action, reward, next_state)
                        self._memory.append(sars)
        self._bus_stop_sar.clear()

    def calculate_hold_time(self, snapshot: Snapshot):
        stop_bus_hold_time = {}
        for (stop_id, route_id, bus_id) in snapshot.holder_snapshot.action_buses:
            if not snapshot.bus_snapshots[(route_id, bus_id)].is_need_to_hold:
                stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0
                continue

            _, forward_spacing, _, backward_spacing = self.extract_local_info_from_snapshot(
                bus_id, snapshot, ['spacing'])

            state, reward = self._transform_snapshot_to_SR(
                snapshot, (route_id, bus_id), stop_id)
            action = 0.0
            if forward_spacing == float('inf') or backward_spacing == float('inf'):
                action, hold_time = 0.0, 0.0
                reward = None
            else:
                action, hold_time = self.infer(state)
                self._record_reward(reward)
                self._record_action(action)

            stop_bus_hold_time[(stop_id, route_id, bus_id)] = hold_time

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
        state_ = torch.tensor(state, dtype=torch.float32).reshape(-1, self._state_size)
        with torch.no_grad():
            action = self._actor_net(state_)
            if self._is_train:
                noise = np.random.normal(0, self._noise_level)
                action = (action + noise).clip(0, 1)
            action = float(action)
        hold_time = action * self._max_hold_time
        return action, hold_time

    def learn(self):
        if self._add_event_count % self._update_cycle != 0 or len(self._memory) < self._batch_size:
            return

        self._actor_net.train()
        self._learn_count += 1
        samples = random.sample(self._memory, self._batch_size)
        stats, actis, rewas, next_stats = [], [], [], []
        for sample in samples:
            stats.append(sample.state)
            actis.append(sample.action)
            rewas.append(sample.reward)
            next_stats.append(sample.next_state)

        s = torch.tensor(stats, dtype=torch.float32).reshape(-1, self._state_size)
        a = torch.tensor(actis, dtype=torch.float32)
        r = torch.tensor(rewas, dtype=torch.float32)
        n_s = torch.tensor(next_stats, dtype=torch.float32).reshape(-1, self._state_size)

        # Update critic
        self._critic_optim.zero_grad()
        s_a = torch.concat((s, a.unsqueeze(dim=1)), dim=1)
        for param in self._critic_net.parameters():
            param.requires_grad = True
        Q = self._critic_net(s_a)

        targe_imagi_a = self._target_actor_net(n_s)
        s_targe_imagi_a = torch.concat((n_s, targe_imagi_a), dim=1)
        with torch.no_grad():
            q_polic_targe = self._target_critic_net(s_targe_imagi_a)
            back_up = r.unsqueeze(1) + self._gamma * q_polic_targe

        td = Q - back_up
        criti_loss = (td**2).mean()
        criti_loss.backward()
        self._critic_optim.step()

        # Update actor
        self._actor_optim.zero_grad()
        imagi_a = self._actor_net(s)
        s_imagi_a = torch.concat((s, imagi_a), dim=1)
        for param in self._critic_net.parameters():
            param.requires_grad = False
        Q_for_actor = self._critic_net(s_imagi_a)
        actor_loss = -Q_for_actor.mean()
        actor_loss.backward()
        self._actor_optim.step()

        # Record metrics
        self._record_metric('critic_loss', float(criti_loss.item()))
        self._record_metric('actor_loss', float(actor_loss.item()))
        self._record_metric('Q_value', float(Q.mean().item()))
        self._record_metric('Q_std', float(Q.std().item()))
        self._record_metric('target_Q', float(q_polic_targe.mean().item()))
        self._record_metric('td_error', float(td.abs().mean().item()))
        self._record_metric('batch_reward', float(r.mean().item()))

        # Update target networks
        with torch.no_grad():
            for p, p_targ in zip(self._actor_net.parameters(), self._target_actor_net.parameters()):
                p_targ.data.mul_(self._polya)
                p_targ.data.add_((1 - self._polya) * p.data)
            for p, p_targ in zip(self._critic_net.parameters(), self._target_critic_net.parameters()):
                p_targ.data.mul_(self._polya)
                p_targ.data.add_((1 - self._polya) * p.data)

    def save_net(self, path: str) -> None:
        torch.save(self._actor_net.state_dict(), path)

    def load_net(self, path: str) -> None:
        self._actor_net.load_state_dict(torch.load(path, map_location='cpu'))

    def _get_model_state_dict(self) -> Dict[str, Any]:
        state_dict = {'actor_net': self._actor_net.state_dict()}
        if self._is_train:
            state_dict['critic_net'] = self._critic_net.state_dict()
        return state_dict

    def _get_optimizer_state_dict(self) -> Dict[str, Any]:
        if not self._is_train:
            return {}
        return {
            'actor_optim': self._actor_optim.state_dict(),
            'critic_optim': self._critic_optim.state_dict(),
        }

    def _get_target_model_state_dict(self) -> Dict[str, Any]:
        if not self._is_train:
            return {}
        return {
            'target_actor_net': self._target_actor_net.state_dict(),
            'target_critic_net': self._target_critic_net.state_dict(),
        }

    def _load_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._actor_net.load_state_dict(state_dict['actor_net'])
        if self._is_train and 'critic_net' in state_dict:
            self._critic_net.load_state_dict(state_dict['critic_net'])

    def _load_optimizer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not self._is_train:
            return
        if 'actor_optim' in state_dict:
            self._actor_optim.load_state_dict(state_dict['actor_optim'])
        if 'critic_optim' in state_dict:
            self._critic_optim.load_state_dict(state_dict['critic_optim'])

    def _load_target_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not self._is_train:
            return
        if 'target_actor_net' in state_dict:
            self._target_actor_net.load_state_dict(state_dict['target_actor_net'])
        if 'target_critic_net' in state_dict:
            self._target_critic_net.load_state_dict(state_dict['target_critic_net'])
