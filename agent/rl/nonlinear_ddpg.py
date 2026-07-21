from copy import deepcopy
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import random
import numpy as np
import torch

from simulator.snapshot import Snapshot
from setup.blueprint import Blueprint
from simulator.virtual_bus import VirtualBus
from simulator.simulator import Simulator

from .rl_agent import RLAgent
from ..single_line_agent import AgentByLine
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


class Nonlinear_DDPG(RLAgent):
    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint, run_config: Dict[str, Any]) -> None:
        super().__init__(agent_config, blueprint)

        self._state_size = agent_config['state_size']
        self._blueprint = blueprint
        self._actor_net = Actor_Net(
            state_size=agent_config['state_size'], hidde_size=tuple(agent_config['hidden_size']))
        self._max_hold_time = agent_config['max_hold_time']
        # self._H = 300 if agent_config['env_name'] == 'homogeneous_one_route' else 170
        self._H = agent_config['schedule_headway']
        self._w = agent_config['w']
        self._f0 = 0.2
        self._f1 = 0
        self._episode_num_for_stabilize_average_hold = agent_config[
            'episode_num_for_stabilize_average_hold']
        self._episode_duration_for_stabilize_average_hold = agent_config[
            'episode_duration_for_stabilize_average_hold']
        self._slack = agent_config['slack']
        self._run_config = run_config
        self._base_type = agent_config['base_type']


        # training mode
        self._critic_net = Critic_Net(
            state_size=agent_config['state_size'], hidde_size=tuple(agent_config['hidden_size']))
        self._target_actor_net = deepcopy(self._actor_net)
        self._target_critic_net = deepcopy(self._critic_net)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
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

        # {{route_id, bus_id}: [(stop_id, SAR)]}
        self._bus_stop_sar: Dict[Tuple[str, str],
                                    List[Tuple[str, SAR]]] = defaultdict(list)
        self._add_event_count = 0
        self._update_cycle = agent_config['update_cycle']
        self._batch_size = agent_config['batch_size']
        self._init_noise_level = agent_config['init_noise_level']
        self._decay_rate = agent_config['decay_rate']
        self._noise_level = self._init_noise_level
        self._learn_count = 0
        self._route_stop_arrival_rate = self._calculate_total_arrival_rate()
        self._route_schedule = self._set_schedule_headway()
        self._generate_virtual_bus()

    def _set_schedule_headway(self) -> Dict[str, float]:
        route_schedule = {}
        for route_id, route in self._blueprint.route_schema.route_details_by_id.items():
            route_schedule[route_id] = route.schedule_headway
        return route_schedule
    
    def calculate_hold_time_simple(self, snapshot: Snapshot) -> Dict[Tuple[str, str, str], float]:
        ''' Implement the nonlinear control algorithm.

        Args:
            snapshot: Snapshot

        Returns:
            stop_bus_hold_time: a dictionary {(stop_id, route_id, bus_id) -> hold_time}

        '''
        stop_bus_hold_time = {}
        action_buses = snapshot.holder_snapshot.action_buses

        if len(action_buses) == 0:
            return stop_bus_hold_time

        for (stop_id, route_id, bus_id) in action_buses:
            if not snapshot.bus_snapshots[(route_id, bus_id)].is_need_to_hold:
                stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0
                continue

            _, forward_spacing, _, backward_spacing = self.extract_local_info_from_snapshot(
                bus_id, snapshot, ['spacing'])

            stop_boarding_rate = self._blueprint.route_schema.route_details_by_id[
                route_id].boarding_rate
            arrival_rate = self._route_stop_arrival_rate[route_id][stop_id]
            beta = arrival_rate / stop_boarding_rate[stop_id]
            H = self._route_schedule[route_id]

            last_rtd_time = snapshot.get_last_rtd_time(route_id, stop_id)
            current_time = snapshot.t
            h = current_time - last_rtd_time

            # get the current bus's `epsilon_arrival` and `epsilon_rtd` at the current stop
            epsilon_arrival_curr_stop, epsilon_rtd_curr_stop = snapshot.get_bus_epsilon(
                route_id, bus_id, stop_id)

            # get the last bus's epsilon_arrival and epsilon_rtd at the current stop
            last_bus_epsilon_arrival_curr_stop, last_bus_epsilon_rtd_curr_stop = snapshot.get_stop_epsilon(
                route_id, stop_id, bus_id)

            # verify if the values are calculated correctly
            numerical_diff = abs(
                (h-H) - (epsilon_rtd_curr_stop - last_bus_epsilon_rtd_curr_stop))
            if numerical_diff > 1:
                print('A mismatch between h-H and epsilon difference...')

            hold_time = 0
            if self._base_type == 'arrival':
                hold_time = -epsilon_arrival_curr_stop + self._f0 * epsilon_arrival_curr_stop

                # hold_time = self._f0*epsilon_arrival_curr_stop + \
                #     self._f1*last_bus_epsilon_arrival_curr_stop

                hold_time += beta * \
                    (last_bus_epsilon_arrival_curr_stop - epsilon_arrival_curr_stop)
                hold_time += self._slack
            elif self._base_type == 'rtd':
                assert self._f1 == 0, 'f0 must be 0 for rtd base type'
                hold_time = -epsilon_rtd_curr_stop + self._f0 * epsilon_arrival_curr_stop

                # hold_time = self._f0*epsilon_rtd_curr_stop + \
                #     self._f1*last_bus_epsilon_rtd_curr_stop
                hold_time += self._slack

            # if forward_spacing == float('inf') or backward_spacing == float('inf'):
            #     hold_time = 0

            hold_time = max(0, hold_time)
            stop_bus_hold_time[(stop_id, route_id, bus_id)] = hold_time

        return stop_bus_hold_time

    def _calculate_total_arrival_rate(self) -> Dict[str, Dict[str, float]]:
        ''' Calculate the total arrival rate at each stop for each route by summing up the OD table by row.
        '''
        route_total_arrival_rate = defaultdict(dict)
        for route_id, route in self._blueprint.route_schema.route_details_by_id.items():
            for origin_stop_id, destination_rate in route.od_rate_table.items():
                total_origin_demand = sum(destination_rate.values())
                route_total_arrival_rate[route_id][origin_stop_id] = total_origin_demand

            last_stop_id = route.visit_seq_stops[-1]

            # # case 1. the last stop's arrival demand rate is 0, i.e., no one will get on the bus at the last stop
            route_total_arrival_rate[route_id][last_stop_id] = 0.0

            # case 2. the last stop's arrival demand rate equals the last but one stop's arrival demand rate
            # last_but_one_stop_id = route.visit_seq_stops[-2]
            # route_total_arrival_rate[route_id][last_stop_id] = route_total_arrival_rate[route_id][last_but_one_stop_id]

        return dict(route_total_arrival_rate)

    def _generate_virtual_bus(self):
        ''' Generate the virtual bus.

        For nonlinear version, the average holding time at each stop need to be dynamically updated
        by running the simulation until convergence. The average holding time is initialized to be the slack.
        The episode number and duration for stabilizing the average holding time are specified in the configuration.
        '''
        # the virtual bus's average holding time is initialized to be the slack
        self._virtual_bus = VirtualBus(self._blueprint)
        self._virtual_bus.initialize_with_perfect_schedule(
            self._route_stop_arrival_rate, self._slack)

        if self._episode_num_for_stabilize_average_hold == 0:
            print('Do not stabilize the average hold time for the virtual bus ......')
            return

        route_stop_average_hold_time: Dict[str, Dict[str, float]] = {}
        for _ in range(self._episode_num_for_stabilize_average_hold):
            simulator = Simulator(self._blueprint, self, self._run_config)
            stop_bus_hold_action: Dict[Tuple[str, str, str], float] = {}
            for t in range(self._episode_duration_for_stabilize_average_hold):
                snapshot = simulator.step(t, stop_bus_hold_action)
                stop_bus_hold_action = self.calculate_hold_time_simple(snapshot)
                snapshot.record_holding_time(stop_bus_hold_action)

            route_stop_average_hold_time = simulator.get_stop_average_hold_time()
            self._virtual_bus.update_trajectory(route_stop_average_hold_time)

    def reset(self, episode: int):
        self._generate_virtual_bus()
        if self._is_train:
            self._noise_level = self._decay_rate ** episode * self._init_noise_level
            self._learn_count = 0
            print('noise level:', self._noise_level)
            # 清空本episode的训练指标
            self._clear_episode_metrics()

    def _get_additional_metrics(self) -> Dict[str, float]:
        """返回Naive_DDPG特有的额外指标"""
        metrics = {}
        if self._is_train:
            metrics['buffer/size'] = len(self._memory)
            metrics['buffer/capacity'] = self._memory.maxlen
            metrics['buffer/utilization'] = len(self._memory) / self._memory.maxlen
            metrics['explore/noise_level'] = self._noise_level
            metrics['train/learn_count'] = self._learn_count
        return metrics

    def _transform_snapshot_to_SR(self, snapshot: Snapshot, acting_bus: Tuple[str, str], stop_id: str) -> Tuple[List[float], float]:
        ''' Transform the snapshot to state, reward.

        Args:
            snapshot: the snapshot of the current time step
            acting_bus: the bus that is acting: (route_id, bus_id)

        '''
        stop_snapshots = snapshot.stop_snapshots
        route_id, bus_id = acting_bus

        # all the buses' arrival time at this stop
        current_stop_arrival_info = stop_snapshots[stop_id].route_arrival_time_seq[acting_bus[0]]
        # current_stop_departure_info = holder_snapshots.route_stop_departure_time_seq[acting_bus[0]][stop_id]
        # the pervious bus's arrival time at this stop
        pervious_bus_arrival_time = current_stop_arrival_info[-2]
        # the current bus's arrival time at this stop
        current_bus_arrival_time = current_stop_arrival_info[-1]
        headway = current_bus_arrival_time - pervious_bus_arrival_time
        normalized_headway = headway / self._H

        stop_boarding_rate = self._blueprint.route_schema.route_details_by_id[
            route_id].boarding_rate
        arrival_rate = self._route_stop_arrival_rate[route_id][stop_id]
        beta = arrival_rate / stop_boarding_rate[stop_id]
        H = self._route_schedule[route_id]

        last_rtd_time = snapshot.get_last_rtd_time(route_id, stop_id)
        current_time = snapshot.t
        h = current_time - last_rtd_time

        last_bus_epsilon_arrival_curr_stop, last_bus_epsilon_rtd_curr_stop = snapshot.get_stop_epsilon(
            route_id, stop_id, bus_id)

        reward = -abs((H - headway) / H)
        # return [normalized_headway, beta, h, H, last_bus_epsilon_arrival_curr_stop, last_bus_epsilon_rtd_curr_stop], reward
        return [normalized_headway], reward

    def _push_transitions_to_memory(self):
        for (route_id, bus_id), sar_list in self._bus_stop_sar.items():
            if len(sar_list) > 1:
                for (stop_id, sar), (next_stop_id, next_sar) in zip(sar_list[0:-1], sar_list[1:]):
                    node_type, found_prev_stop_id = self._blueprint.get_previous_node(
                        route_id, next_stop_id)
                    assert node_type != 'terminal', 'The previous node cannot be a terminal'

                    if found_prev_stop_id == stop_id:
                        # if int(next_stop_id) - int(stop_id) == 1:
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
                # 记录奖励和动作用于统计
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
            # when training, add noise
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
        stats = []
        actis = []
        rewas = []
        next_stats = []
        for sample in samples:
            stats.append(sample.state)
            actis.append(sample.action)
            rewas.append(sample.reward)
            next_stats.append(sample.next_state)

        s = torch.tensor(stats, dtype=torch.float32).reshape(-1, self._state_size)
        # LongTensor for idx selection
        a = torch.tensor(actis, dtype=torch.float32)
        r = torch.tensor(rewas, dtype=torch.float32)
        n_s = torch.tensor(next_stats, dtype=torch.float32).reshape(-1, self._state_size)
        # update critic network
        # self.__criti_net.zero_grad()
        self._critic_optim.zero_grad()
        # current estimate
        s_a = torch.concat((s, a.unsqueeze(dim=1)), dim=1)
        for param in self._critic_net.parameters():
            param.requires_grad = True
        Q = self._critic_net(s_a)

        # Bellman backup for Q function
        targe_imagi_a = self._target_actor_net(n_s)  # (batch_size, 1)
        s_targe_imagi_a = torch.concat((n_s, targe_imagi_a), dim=1)
        with torch.no_grad():
            q_polic_targe = self._target_critic_net(s_targe_imagi_a)
            # r is (batch_size, ), need to align with output from NN
            back_up = r.unsqueeze(1) + self._gamma * q_polic_targe
        # MSE loss against Bellman backup
        # Unfreeze Q-network so as to optimize it
        td = Q - back_up
        criti_loss = (td**2).mean()
        # update critic parameters
        criti_loss.backward()
        self._critic_optim.step()

        # update actor network
        self._actor_optim.zero_grad()
        imagi_a = self._actor_net(s)
        s_imagi_a = torch.concat((s, imagi_a), dim=1)
        # Freeze Q-network to save computational efforts
        for param in self._critic_net.parameters():
            param.requires_grad = False
        Q_for_actor = self._critic_net(s_imagi_a)
        actor_loss = -Q_for_actor.mean()
        actor_loss.backward()
        self._actor_optim.step()

        # 记录训练指标
        self._record_metric('critic_loss', float(criti_loss.item()))
        self._record_metric('actor_loss', float(actor_loss.item()))
        self._record_metric('Q_value', float(Q.mean().item()))
        self._record_metric('Q_std', float(Q.std().item()))
        self._record_metric('target_Q', float(q_polic_targe.mean().item()))
        self._record_metric('td_error', float(td.abs().mean().item()))
        self._record_metric('batch_reward', float(r.mean().item()))

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self._actor_net.parameters(), self._target_actor_net.parameters()):
                p_targ.data.mul_(self._polya)
                p_targ.data.add_((1 - self._polya) * p.data)
            for p, p_targ in zip(self._critic_net.parameters(), self._target_critic_net.parameters()):
                p_targ.data.mul_(self._polya)
                p_targ.data.add_((1 - self._polya) * p.data)

    def save_net(self, path: str) -> None:
        """简单保存（向后兼容）"""
        torch.save(self._actor_net.state_dict(), path)

    def load_net(self, path: str) -> None:
        """简单加载（向后兼容）"""
        self._actor_net.load_state_dict(torch.load(path, map_location='cpu'))

    # ============ Checkpoint 方法实现 ============

    def _get_model_state_dict(self) -> Dict[str, Any]:
        """获取所有模型的状态字典"""
        state_dict = {'actor_net': self._actor_net.state_dict()}
        if self._is_train:
            state_dict['critic_net'] = self._critic_net.state_dict()
        return state_dict

    def _get_optimizer_state_dict(self) -> Dict[str, Any]:
        """获取所有优化器的状态字典"""
        if not self._is_train:
            return {}
        return {
            'actor_optim': self._actor_optim.state_dict(),
            'critic_optim': self._critic_optim.state_dict(),
        }

    def _get_target_model_state_dict(self) -> Dict[str, Any]:
        """获取所有目标网络的状态字典"""
        if not self._is_train:
            return {}
        return {
            'target_actor_net': self._target_actor_net.state_dict(),
            'target_critic_net': self._target_critic_net.state_dict(),
        }

    def _load_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载所有模型的状态字典"""
        self._actor_net.load_state_dict(state_dict['actor_net'])
        if self._is_train and 'critic_net' in state_dict:
            self._critic_net.load_state_dict(state_dict['critic_net'])

    def _load_optimizer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载所有优化器的状态字典"""
        if not self._is_train:
            return
        if 'actor_optim' in state_dict:
            self._actor_optim.load_state_dict(state_dict['actor_optim'])
        if 'critic_optim' in state_dict:
            self._critic_optim.load_state_dict(state_dict['critic_optim'])

    def _load_target_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载所有目标网络的状态字典"""
        if not self._is_train:
            return
        if 'target_actor_net' in state_dict:
            self._target_actor_net.load_state_dict(state_dict['target_actor_net'])
        if 'target_critic_net' in state_dict:
            self._target_critic_net.load_state_dict(state_dict['target_critic_net'])
