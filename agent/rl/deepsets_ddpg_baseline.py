"""
DeepSets_DDPG_Baseline — DDPG with DeepSets (mean-pool) encoder using the
SAME state features as the Transformer agent (SetDDPG_BusHoldingAgent_Or).

Purpose: Isolate the attention mechanism's contribution by comparing against
a permutation-invariant aggregation (mean pooling) that does NOT use attention.

Architecture:
    ego → BusTokenEmbedding → ego_token
    leaders → BusTokenEmbedding → phi(·) → masked mean pool → leader_repr
    concat(ego_token, leader_repr) → rho(·) → output (token_dim)

This follows the DeepSets framework (Zaheer et al., 2017):
    f(X) = rho( sum/mean_x phi(x) )

Everything else (state extraction, reward, DDPG training) is identical
to SetDDPG_BusHoldingAgent_Or.
"""

from copy import deepcopy
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from simulator.snapshot import Snapshot
from setup.blueprint import Blueprint
from .rl_agent import RLAgent


# ──────────────────────────────────────────────
# Replay structures
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SAR:
    state: Dict[str, Any]
    action: float
    reward: Optional[float]


@dataclass(frozen=True)
class SARS:
    state: Dict[str, Any]
    action: float
    reward: float
    next_state: Dict[str, Any]
    done: bool


# ──────────────────────────────────────────────
# Embedding (same as transformer)
# ──────────────────────────────────────────────

class BusTokenEmbedding(nn.Module):
    def __init__(self, num_stops, stop_emb_dim, continuous_dim, token_dim, padding_idx=0):
        super().__init__()
        self.stop_embed = nn.Embedding(num_stops, stop_emb_dim, padding_idx=padding_idx)
        self.cont_mlp = nn.Sequential(
            nn.Linear(continuous_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, stop_emb_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(stop_emb_dim * 2, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, stop_id, cont):
        s = self.stop_embed(stop_id)
        c = self.cont_mlp(cont)
        return self.fuse(torch.cat([s, c], dim=-1))


# ──────────────────────────────────────────────
# DeepSets Encoder (replaces SetLeaderEncoder)
# ──────────────────────────────────────────────

class DeepSetsLeaderEncoder(nn.Module):
    """
    DeepSets-based encoder (permutation-invariant, NO attention):
        ego → BusTokenEmbedding → ego_token
        leaders → BusTokenEmbedding → phi(·) per element → masked mean pool
        concat(ego_token, pooled_leaders) → rho(·) → output (token_dim)

    When no leaders are present, uses a learned fallback token.
    """

    def __init__(self, num_stops, ego_cont_dim, leader_cont_dim,
                 stop_emb_dim=32, token_dim=64,
                 max_leaders=20, padding_idx=0, dropout=0.0):
        super().__init__()
        self.token_dim = token_dim
        self.max_leaders = max_leaders

        self.ego_embed = BusTokenEmbedding(num_stops, stop_emb_dim, ego_cont_dim, token_dim, padding_idx)
        self.leader_embed = BusTokenEmbedding(num_stops, stop_emb_dim, leader_cont_dim, token_dim, padding_idx)

        # phi: per-element transformation (applied to each leader independently)
        self.phi = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.ReLU(),
        )

        self.no_leader_token = nn.Parameter(torch.randn(1, token_dim) * 0.02)

        # rho: post-aggregation transformation
        self.rho = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )

    def forward(self, ego_features, leader_features=None, num_leading=None):
        B = ego_features["stop_id"].shape[0]
        device = ego_features["stop_id"].device

        ego_tok = self.ego_embed(
            ego_features["stop_id"], ego_features["continuous"]
        )  # [B, token_dim]

        if leader_features is None or num_leading is None or int(num_leading.max().item()) == 0:
            leader_repr = self.no_leader_token.expand(B, -1)
        else:
            K = leader_features["stop_id"].shape[1]
            flat_stop = leader_features["stop_id"].reshape(-1)
            flat_cont = leader_features["continuous"].reshape(-1, leader_features["continuous"].shape[-1])
            leaders_tok = self.leader_embed(flat_stop, flat_cont).reshape(B, K, -1)

            # Apply phi to each leader token independently
            leaders_tok = self.phi(leaders_tok)  # [B, K, token_dim]

            # Masked mean pooling (zero out padded positions)
            pad_mask = torch.arange(K, device=device).unsqueeze(0) >= num_leading.unsqueeze(1)
            leaders_tok = leaders_tok.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            counts = num_leading.float().clamp(min=1.0).unsqueeze(-1)
            leader_repr = leaders_tok.sum(dim=1) / counts  # [B, token_dim]

        return self.rho(torch.cat([ego_tok, leader_repr], dim=-1))


# ──────────────────────────────────────────────
# Actor / Critic
# ──────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, encoder, hidden=(128, 64)):
        super().__init__()
        self.encoder = encoder
        layers = []
        d = encoder.token_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, ego_f, leader_f=None, num_l=None):
        z = self.encoder(ego_f, leader_f, num_l)
        return (torch.tanh(self.head(z)) + 1.0) / 2.0

    def forward_raw(self, ego_f, leader_f=None, num_l=None):
        z = self.encoder(ego_f, leader_f, num_l)
        return self.head(z)


class Critic(nn.Module):
    def __init__(self, encoder, hidden=(128, 64)):
        super().__init__()
        self.encoder = encoder
        layers = []
        d = encoder.token_dim + 1
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.q = nn.Sequential(*layers)

    def forward(self, ego_f, a, leader_f=None, num_l=None):
        z = self.encoder(ego_f, leader_f, num_l)
        return self.q(torch.cat([z, a], dim=-1))


# ──────────────────────────────────────────────
# Main Agent — DeepSets_DDPG_Baseline
# ──────────────────────────────────────────────

class DeepSets_DDPG_Baseline(RLAgent):

    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint,
                 run_config: Dict[str, Any]) -> None:
        super().__init__(agent_config, blueprint)
        self._blueprint = blueprint
        self._max_hold_time = agent_config["max_hold_time"]
        self._H = agent_config["schedule_headway"]

        use_gpu = agent_config.get("use_gpu", True)
        self._device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"DeepSets_DDPG_Baseline: Using {self._device}")

        # Stop ID mapping (0 = PAD)
        all_stops = set()
        for _, details in blueprint.route_schema.route_details_by_id.items():
            all_stops.update(details.visit_seq_stops)
        all_stops = sorted(list(all_stops))
        self._PAD_STOP = 0
        self._stop_to_idx = {s: (i + 1) for i, s in enumerate(all_stops)}
        num_stops = len(all_stops) + 1

        self._route_stop_to_pos = {
            rid: {sid: i for i, sid in enumerate(details.visit_seq_stops)}
            for rid, details in blueprint.route_schema.route_details_by_id.items()
        }

        # Dimensions
        ego_cont_dim = 4
        leader_cont_dim = 5

        stop_emb_dim = agent_config.get("stop_emb_dim", 32)
        token_dim = agent_config.get("token_dim", 64)
        max_leaders = agent_config.get("max_buses", 20)
        hidden = tuple(agent_config.get("hidden_size", (128, 64)))
        dropout = agent_config.get("dropout", 0.0)

        # DDPG hyperparams
        self._gamma = agent_config["gamma"]
        self._polya = agent_config["polya"]
        self._batch_size = agent_config["batch_size"]
        self._update_cycle = agent_config["update_cycle"]
        self._grad_clip = agent_config.get("grad_clip", 1.0)

        # Exploration
        self._init_noise_level = agent_config["init_noise_level"]
        self._decay_rate = agent_config["decay_rate"]
        self._noise_level = self._init_noise_level
        self._learn_count = 0
        self._add_event_count = 0

        # Reward weights (same as transformer_ddpg_imp)
        self._w = agent_config["w"]  # action penalty weight
        self._w_headway = agent_config.get("w_headway", 0.9)
        self._w_pax_wait = agent_config.get("w_pax_wait", 0.1)
        self._pax_wait_scale = agent_config.get("pax_wait_scale", 100.0)
        self._w_pax_onbus = agent_config.get("w_pax_onbus", 0.05)
        self._w_hold = agent_config.get("w_hold", 0.05)
        self._max_cum_hw_norm = agent_config.get("max_cum_hw_norm", 10.0)

        # Networks — DeepSets encoder
        def make_encoder():
            return DeepSetsLeaderEncoder(
                num_stops, ego_cont_dim, leader_cont_dim,
                stop_emb_dim, token_dim,
                max_leaders, self._PAD_STOP, dropout
            )

        self._actor = Actor(make_encoder(), hidden).to(self._device)
        self._critic = Critic(make_encoder(), hidden).to(self._device)

        self._t_actor = deepcopy(self._actor).to(self._device)
        self._t_critic = deepcopy(self._critic).to(self._device)
        for m in [self._t_actor, self._t_critic]:
            for p in m.parameters():
                p.requires_grad_(False)

        self._actor_optim = torch.optim.Adam(self._actor.parameters(), lr=agent_config["actor_lr"])
        self._critic_optim = torch.optim.Adam(self._critic.parameters(), lr=agent_config["critic_lr"])

        total = sum(p.numel() for p in self._actor.parameters())
        print(f"DeepSets_DDPG_Baseline: Actor params: {total:,}")

        # Replay
        self._memory = deque(maxlen=agent_config["memory_size"])
        self._bus_stop_sar: Dict[Tuple[str, str], List[Tuple[str, SAR]]] = defaultdict(list)

    # ──────────────────────────────────────────
    # State extraction (same as transformer)
    # ──────────────────────────────────────────

    def _extract_state_dict(self, snapshot: Snapshot, route_id: str,
                            bus_id: str, stop_id: str) -> Dict[str, Any]:
        stop_snaps = snapshot.stop_snapshots
        bus_snaps = snapshot.bus_snapshots

        arrival_info = stop_snaps[stop_id].route_arrival_time_seq[route_id]
        headway = (arrival_info[-1] - arrival_info[-2]) if len(arrival_info) >= 2 else self._H
        normalized_headway = headway / self._H

        pax_waiting_raw = stop_snaps[stop_id].pax_num
        pax_num_raw = bus_snaps[(route_id, bus_id)].pax_num

        visit_seq = self._blueprint.route_schema.route_details_by_id[route_id].visit_seq_stops
        pos = self._route_stop_to_pos[route_id].get(stop_id, -1)
        next_stop_id = visit_seq[pos + 1] if 0 <= pos < len(visit_seq) - 1 else None
        pax_next_raw = stop_snaps[next_stop_id].pax_num if next_stop_id else 0

        ego = {
            "stop_idx": self._stop_to_idx.get(stop_id, self._PAD_STOP),
            "continuous": [normalized_headway, pax_waiting_raw / 80.0,
                           pax_next_raw / 80.0, pax_num_raw / 80.0],
            "pax_num_raw": pax_num_raw,
        }

        arrival_bus_seq = stop_snaps[stop_id].route_arrival_bus_id_seq[route_id]
        if bus_id not in arrival_bus_seq:
            return {"ego": ego, "leaders": [], "cumulative_headways": []}

        curr = arrival_bus_seq.index(bus_id)
        leaders, cumulative_headways = [], []
        cum_hw = 0.0

        for i in range(curr - 1, -1, -1):
            lb = arrival_bus_seq[i]
            key = (route_id, lb)
            if key not in bus_snaps:
                continue
            ls = bus_snaps[key]
            if getattr(ls, "status", "") == "finished":
                continue
            visited = getattr(ls, "visited_stops", [])
            if not visited:
                continue
            leader_stop = visited[-1]

            step_hw = abs(arrival_info[i + 1] - arrival_info[i]) if i < len(arrival_info) - 1 else self._H
            cum_hw += step_hw

            leader_pax_raw = ls.pax_num
            leader_stop_snap = stop_snaps.get(leader_stop, None)
            leader_pax_at_stop = leader_stop_snap.pax_num if leader_stop_snap else 0

            leader_pos = self._route_stop_to_pos[route_id].get(leader_stop, -1)
            if 0 <= leader_pos < len(visit_seq) - 1:
                lns = visit_seq[leader_pos + 1]
                leader_pax_next = stop_snaps[lns].pax_num if lns in stop_snaps else 0
            else:
                leader_pax_next = 0

            cum_hw_norm = min(cum_hw / self._H, self._max_cum_hw_norm)

            leaders.append({
                "stop_idx": self._stop_to_idx.get(leader_stop, self._PAD_STOP),
                "continuous": [step_hw / self._H, leader_pax_at_stop / 80.0,
                               leader_pax_next / 80.0, leader_pax_raw / 80.0,
                               cum_hw_norm],
            })
            cumulative_headways.append(cum_hw_norm)

        return {"ego": ego, "leaders": leaders, "cumulative_headways": cumulative_headways}

    # ──────────────────────────────────────────
    # Tensorization (same as transformer)
    # ──────────────────────────────────────────

    def _tensorize_single(self, state_dict, device):
        ego = state_dict["ego"]
        ego_f = {
            "stop_id": torch.tensor([ego["stop_idx"]], dtype=torch.long, device=device),
            "continuous": torch.tensor([ego["continuous"]], dtype=torch.float32, device=device),
        }
        leaders = state_dict["leaders"]
        num_l = torch.tensor([len(leaders)], dtype=torch.long, device=device)
        if len(leaders) > 0:
            leader_f = {
                "stop_id": torch.tensor([[l["stop_idx"] for l in leaders]],
                                        dtype=torch.long, device=device),
                "continuous": torch.tensor([[l["continuous"] for l in leaders]],
                                           dtype=torch.float32, device=device),
            }
        else:
            leader_f = None
        return ego_f, leader_f, num_l

    def _tensorize_batch(self, state_dicts, device):
        B = len(state_dicts)
        max_k = max((len(sd["leaders"]) for sd in state_dicts), default=0)

        ego_f = {
            "stop_id": torch.tensor([sd["ego"]["stop_idx"] for sd in state_dicts],
                                    dtype=torch.long, device=device),
            "continuous": torch.tensor([sd["ego"]["continuous"] for sd in state_dicts],
                                       dtype=torch.float32, device=device),
        }
        num_l = torch.tensor([len(sd["leaders"]) for sd in state_dicts],
                             dtype=torch.long, device=device)

        if max_k == 0:
            return ego_f, None, num_l

        padded_stop = torch.full((B, max_k), self._PAD_STOP, dtype=torch.long, device=device)
        padded_cont = torch.zeros((B, max_k, 5), dtype=torch.float32, device=device)

        for i, sd in enumerate(state_dicts):
            k = len(sd["leaders"])
            if k > 0:
                padded_stop[i, :k] = torch.tensor(
                    [l["stop_idx"] for l in sd["leaders"]], dtype=torch.long, device=device)
                padded_cont[i, :k] = torch.tensor(
                    [l["continuous"] for l in sd["leaders"]], dtype=torch.float32, device=device)

        leader_f = {"stop_id": padded_stop, "continuous": padded_cont}
        return ego_f, leader_f, num_l

    # ──────────────────────────────────────────
    # Reward (same as transformer_ddpg_imp)
    # ──────────────────────────────────────────

    def _compute_base_reward(self, snapshot, route_id, stop_id):
        stop_snaps = snapshot.stop_snapshots
        arrival_info = stop_snaps[stop_id].route_arrival_time_seq[route_id]
        headway = (arrival_info[-1] - arrival_info[-2]) if len(arrival_info) >= 2 else self._H

        headway_reward = -abs((self._H - headway) / self._H)

        t = snapshot.t
        pax_arrival_times = stop_snaps[stop_id].pax_arrival_times
        if len(pax_arrival_times) > 0:
            total_wait_time = sum(t - arr_t for arr_t in pax_arrival_times) / self._H
        else:
            total_wait_time = 0.0
        pax_penalty = -(total_wait_time / self._pax_wait_scale)

        return self._w_headway * headway_reward + self._w_pax_wait * pax_penalty

    # ──────────────────────────────────────────
    # Interaction
    # ──────────────────────────────────────────

    def reset(self, episode):
        if self._is_train:
            self._push_transitions_to_memory()
            self._noise_level = (self._decay_rate ** episode) * self._init_noise_level
            self._learn_count = 0
            self._clear_episode_metrics()

    def infer(self, state_dict):
        self._actor.eval()
        ego_f, leader_f, num_l = self._tensorize_single(state_dict, self._device)
        with torch.no_grad():
            if self._is_train:
                raw = self._actor.forward_raw(ego_f, leader_f, num_l)
                noise = torch.randn_like(raw) * self._noise_level
                a = (torch.tanh(raw + noise) + 1.0) / 2.0
            else:
                a = self._actor(ego_f, leader_f, num_l)
            a = float(a.cpu().item())
        return a, a * self._max_hold_time

    def calculate_hold_time(self, snapshot):
        stop_bus_hold_time = {}
        for (stop_id, route_id, bus_id) in snapshot.holder_snapshot.action_buses:
            if not snapshot.bus_snapshots[(route_id, bus_id)].is_need_to_hold:
                stop_bus_hold_time[(stop_id, route_id, bus_id)] = 0.0
                continue

            _, forward_spacing, _, backward_spacing = self.extract_local_info_from_snapshot(
                bus_id, snapshot, ["spacing"]
            )

            s = self._extract_state_dict(snapshot, route_id, bus_id, stop_id)
            base_r = self._compute_base_reward(snapshot, route_id, stop_id)

            if forward_spacing == float("inf") or backward_spacing == float("inf"):
                a, hold = 0.0, 0.0
                base_r = None
            else:
                a, hold = self.infer(s)
                self._record_reward(base_r)
                self._record_action(a)

            stop_bus_hold_time[(stop_id, route_id, bus_id)] = hold

            if self.is_train:
                self._bus_stop_sar[(route_id, bus_id)].append((stop_id, SAR(s, a, base_r)))
                self._add_event_count += 1
                if self._add_event_count % self._batch_size == 0:
                    self._push_transitions_to_memory()
                self.learn()

        snapshot.record_holding_time(stop_bus_hold_time)
        return stop_bus_hold_time

    # ──────────────────────────────────────────
    # Build transitions
    # ──────────────────────────────────────────

    def _push_transitions_to_memory(self):
        for (route_id, bus_id), sar_list in self._bus_stop_sar.items():
            if len(sar_list) <= 1:
                continue
            for (stop_id, sar), (next_stop_id, next_sar) in zip(sar_list[:-1], sar_list[1:]):
                node_type, found_prev = self._blueprint.get_previous_node(route_id, next_stop_id)
                if found_prev != stop_id:
                    continue
                done = (node_type == "terminal")
                if any(v is None for v in [sar.state, sar.action, next_sar.reward, next_sar.state]):
                    continue

                r = float(next_sar.reward)
                a = float(sar.action)

                self._memory.append(SARS(sar.state, a, r, next_sar.state, done))
        self._bus_stop_sar.clear()

    # ──────────────────────────────────────────
    # DDPG learning
    # ──────────────────────────────────────────

    def learn(self):
        if (self._add_event_count % self._update_cycle != 0) or \
           (len(self._memory) < self._batch_size):
            return

        self._actor.train()
        self._critic.train()
        self._learn_count += 1

        batch = random.sample(self._memory, self._batch_size)

        a = torch.tensor([x.action for x in batch],
                         dtype=torch.float32, device=self._device).unsqueeze(1)
        r = torch.tensor([x.reward for x in batch],
                         dtype=torch.float32, device=self._device).unsqueeze(1)
        d = torch.tensor([1.0 if x.done else 0.0 for x in batch],
                         dtype=torch.float32, device=self._device).unsqueeze(1)

        s_ego, s_lead, s_num = self._tensorize_batch([x.state for x in batch], self._device)
        ns_ego, ns_lead, ns_num = self._tensorize_batch([x.next_state for x in batch], self._device)

        # Critic update
        self._critic_optim.zero_grad(set_to_none=True)
        q = self._critic(s_ego, a, s_lead, s_num)

        with torch.no_grad():
            na = self._t_actor(ns_ego, ns_lead, ns_num)
            tq = self._t_critic(ns_ego, na, ns_lead, ns_num)
            y = r + self._gamma * (1.0 - d) * tq

        critic_loss = F.mse_loss(q, y)

        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            self._critic_optim.zero_grad(set_to_none=True)
            return

        critic_loss.backward()
        if self._grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._grad_clip)
        self._critic_optim.step()

        # Actor update
        self._actor_optim.zero_grad(set_to_none=True)
        pa = self._actor(s_ego, s_lead, s_num)
        actor_loss = -self._critic(s_ego, pa, s_lead, s_num).mean()

        actor_loss_val = None
        if not (torch.isnan(actor_loss) or torch.isinf(actor_loss)):
            actor_loss.backward()
            if self._grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._grad_clip)
            self._actor_optim.step()
            actor_loss_val = float(actor_loss.item())

        # Polyak update
        with torch.no_grad():
            for p, pt in zip(self._actor.parameters(), self._t_actor.parameters()):
                pt.data.mul_(self._polya).add_((1 - self._polya) * p.data)
            for p, pt in zip(self._critic.parameters(), self._t_critic.parameters()):
                pt.data.mul_(self._polya).add_((1 - self._polya) * p.data)

        self._record_metric("critic_loss", float(critic_loss.item()))
        self._record_metric("q", float(q.mean().item()))
        self._record_metric("target_q", float(y.mean().item()))
        self._record_metric("batch_reward", float(r.mean().item()))
        self._record_metric("batch_done_rate", float(d.mean().item()))
        if actor_loss_val is not None:
            self._record_metric("actor_loss", actor_loss_val)

    # ──────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────

    def _get_model_state_dict(self):
        d = {"actor": self._actor.state_dict()}
        if self._is_train:
            d["critic"] = self._critic.state_dict()
        return d

    def _get_optimizer_state_dict(self):
        if not self._is_train:
            return {}
        return {
            "actor_optim": self._actor_optim.state_dict(),
            "critic_optim": self._critic_optim.state_dict(),
        }

    def _get_target_model_state_dict(self):
        if not self._is_train:
            return {}
        return {
            "t_actor": self._t_actor.state_dict(),
            "t_critic": self._t_critic.state_dict(),
        }

    def _load_model_state_dict(self, sd):
        self._actor.load_state_dict(sd["actor"])
        self._actor.to(self._device)
        if self._is_train:
            if "critic" in sd:
                self._critic.load_state_dict(sd["critic"])
            self._critic.to(self._device)

    def _load_optimizer_state_dict(self, sd):
        if not self._is_train:
            return
        if "actor_optim" in sd:
            self._actor_optim.load_state_dict(sd["actor_optim"])
        if "critic_optim" in sd:
            self._critic_optim.load_state_dict(sd["critic_optim"])

    def _load_target_model_state_dict(self, sd):
        if not self._is_train:
            return
        if "t_actor" in sd:
            self._t_actor.load_state_dict(sd["t_actor"])
        if "t_critic" in sd:
            self._t_critic.load_state_dict(sd["t_critic"])
        self._t_actor.to(self._device)
        self._t_critic.to(self._device)

    def _get_additional_metrics(self):
        if not self._is_train:
            return {}
        return {
            "buffer/size": float(len(self._memory)),
            "buffer/capacity": float(self._memory.maxlen),
            "explore/noise_level": float(self._noise_level),
            "train/learn_count": float(self._learn_count),
        }

    def save_net(self, path):
        torch.save(self._actor.state_dict(), path)

    def load_net(self, path):
        self._actor.load_state_dict(torch.load(path, map_location=self._device))
        self._actor.to(self._device)
