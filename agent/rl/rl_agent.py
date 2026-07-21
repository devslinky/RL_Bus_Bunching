from typing import Dict, Any, List, Optional
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import os
import numpy as np
import torch
from setup.blueprint import Blueprint

from agent.agent import Agent


class RLAgent(Agent):
    """强化学习Agent基类，提供训练指标收集和模型检查点管理功能"""

    def __init__(self, agent_config: Dict[str, Any], blueprint: Blueprint) -> None:
        super().__init__(agent_config)
        self._blueprint = blueprint
        self._is_train: bool = agent_config['is_train']
        self._agent_config = agent_config  # 保存配置用于checkpoint

        # 训练指标收集器
        self._training_metrics: Dict[str, List[float]] = defaultdict(list)
        self._episode_rewards: List[float] = []
        self._episode_actions: List[float] = []

        # 训练历史记录（用于保存到checkpoint）
        self._training_history: Dict[str, List[float]] = defaultdict(list)
        self._best_metric: float = float('inf')  # 用于保存最佳模型
        self._current_episode: int = 0

    @property
    def is_train(self) -> bool:
        return self._is_train

    @is_train.setter
    def is_train(self, is_train: bool) -> None:
        self._is_train = is_train

    def _record_metric(self, name: str, value: float) -> None:
        """记录单个训练指标"""
        if value is not None and not np.isnan(value) and not np.isinf(value):
            self._training_metrics[name].append(value)

    def _record_reward(self, reward: float) -> None:
        """记录奖励"""
        if reward is not None and not np.isnan(reward) and not np.isinf(reward):
            self._episode_rewards.append(reward)

    def _record_action(self, action: float) -> None:
        """记录动作"""
        if action is not None and not np.isnan(action) and not np.isinf(action):
            self._episode_actions.append(action)

    def get_training_metrics(self) -> Dict[str, float]:
        """
        获取本episode的训练指标汇总

        Returns:
            Dict包含以下指标（如果有数据）:
            - train/critic_loss: Critic网络损失
            - train/actor_loss: Actor网络损失
            - train/Q_mean: Q值均值
            - train/Q_std: Q值标准差
            - train/td_error: TD误差均值
            - train/reward_mean: 奖励均值
            - train/reward_std: 奖励标准差
            - train/action_mean: 动作均值
            - train/action_std: 动作标准差
            - train/learn_steps: 本episode学习次数
            - buffer/size: 经验回放缓冲区大小
            - explore/noise_level: 当前噪声水平
        """
        metrics = {}

        # 汇总训练指标
        for name, values in self._training_metrics.items():
            if values:
                metrics[f'train/{name}_mean'] = float(np.mean(values))
                if len(values) > 1:
                    metrics[f'train/{name}_std'] = float(np.std(values))

        # 奖励统计
        if self._episode_rewards:
            metrics['train/reward_mean'] = float(np.mean(self._episode_rewards))
            metrics['train/reward_std'] = float(np.std(self._episode_rewards))
            metrics['train/reward_min'] = float(np.min(self._episode_rewards))
            metrics['train/reward_max'] = float(np.max(self._episode_rewards))
            metrics['train/reward_count'] = len(self._episode_rewards)

        # 动作统计
        if self._episode_actions:
            metrics['train/action_mean'] = float(np.mean(self._episode_actions))
            metrics['train/action_std'] = float(np.std(self._episode_actions))

        # 子类可以添加额外的指标（如buffer_size, noise_level）
        metrics.update(self._get_additional_metrics())

        return metrics

    def _get_additional_metrics(self) -> Dict[str, float]:
        """子类可重写以添加额外指标"""
        return {}

    def _clear_episode_metrics(self) -> None:
        """清空本episode的指标，在reset时调用"""
        self._training_metrics.clear()
        self._episode_rewards.clear()
        self._episode_actions.clear()

    def record_episode_metrics(self, episode: int, env_metrics: Dict[str, float]) -> None:
        """记录每个episode的指标到训练历史"""
        self._current_episode = episode
        for name, value in env_metrics.items():
            self._training_history[f'env/{name}'].append(value)

        # 记录训练指标
        training_metrics = self.get_training_metrics()
        for name, value in training_metrics.items():
            self._training_history[name].append(value)

    def save_checkpoint(
        self,
        save_dir: str,
        episode: int,
        env_metrics: Dict[str, float],
        env_name: str,
        run_config: Dict[str, Any],
        is_best: bool = False,
        custom_name: Optional[str] = None
    ) -> str:
        """
        保存完整的模型检查点，包含所有训练信息

        Args:
            save_dir: 保存根目录
            episode: 当前episode数
            env_metrics: 环境指标（如headway_std）
            env_name: 环境名称
            run_config: 运行配置
            is_best: 是否为最佳模型
            custom_name: 自定义文件名（可选）

        Returns:
            保存的检查点路径

        目录结构:
            checkpoints/
            └── {agent_name}_{env_name}_{timestamp}/
                ├── best.pt
                ├── best_meta.json
                ├── ep10.pt
                ├── ep10_meta.json
                ├── final.pt
                └── final_meta.json
        """
        # 生成时间戳和agent名称
        agent_name = self.__class__.__name__

        # 创建子文件夹: {agent_name}_{env_name}_{start_timestamp}
        # 使用实例属性来保持同一次训练使用同一个文件夹
        if not hasattr(self, '_checkpoint_subdir') or self._checkpoint_subdir is None:
            start_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._checkpoint_subdir = f'{agent_name}_{env_name}_{start_timestamp}'

        # 创建完整保存路径
        save_path = Path(save_dir) / self._checkpoint_subdir
        save_path.mkdir(parents=True, exist_ok=True)

        # 生成文件名（简化，因为已经在子文件夹中）
        if custom_name:
            filename = custom_name
        elif is_best:
            filename = 'best'
        else:
            filename = f'ep{episode}'

        checkpoint_path = save_path / f'{filename}.pt'
        metadata_path = save_path / f'{filename}_meta.json'

        # 构建检查点数据
        checkpoint = {
            # 模型权重（由子类实现）
            'model_state_dict': self._get_model_state_dict(),
            'optimizer_state_dict': self._get_optimizer_state_dict(),
            'target_model_state_dict': self._get_target_model_state_dict(),

            # 训练状态
            'episode': episode,
            'noise_level': getattr(self, '_noise_level', None),
            'learn_count': getattr(self, '_learn_count', 0),

            # 配置信息
            'agent_config': self._agent_config,
            'agent_name': agent_name,
            'env_name': env_name,

            # 最终指标
            'final_metrics': env_metrics,
        }

        # 保存PyTorch检查点
        torch.save(checkpoint, checkpoint_path)

        # 构建可读的元数据
        metadata = {
            'checkpoint_file': str(checkpoint_path),
            'checkpoint_dir': str(save_path),
            'filename': filename,
            'created_at': datetime.now().isoformat(),
            'agent_name': agent_name,
            'env_name': env_name,
            'episode': episode,
            'is_best': is_best,

            # 训练配置
            'training_config': {
                'state_size': self._agent_config.get('state_size'),
                'hidden_size': self._agent_config.get('hidden_size'),
                'actor_lr': self._agent_config.get('actor_lr'),
                'critic_lr': self._agent_config.get('critic_lr'),
                'gamma': self._agent_config.get('gamma'),
                'batch_size': self._agent_config.get('batch_size'),
                'memory_size': self._agent_config.get('memory_size'),
                'max_hold_time': self._agent_config.get('max_hold_time'),
            },

            # 运行配置
            'run_config': {
                'episode_num': run_config.get('episode_num'),
                'episode_duration': run_config.get('episode_duration'),
                'hold_start_time': run_config.get('hold_start_time'),
                'hold_end_time': run_config.get('hold_end_time'),
            },

            # 最终性能指标
            'final_metrics': env_metrics,

            # 训练历史摘要
            'training_summary': self._get_training_summary(),
        }

        # 保存JSON元数据
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 同时保存/更新训练信息文件（在子文件夹根目录）
        training_info_path = save_path / 'training_info.json'
        training_info = {
            'agent_name': agent_name,
            'env_name': env_name,
            'training_start': self._checkpoint_subdir.split('_')[-2] + '_' + self._checkpoint_subdir.split('_')[-1],
            'last_updated': datetime.now().isoformat(),
            'total_episodes': episode + 1,
            'training_config': metadata['training_config'],
            'run_config': metadata['run_config'],
            'checkpoints': self._get_checkpoint_list(save_path),
        }
        with open(training_info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)

        print(f'Checkpoint saved: {checkpoint_path}')
        print(f'Metadata saved: {metadata_path}')

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        加载模型检查点

        Args:
            checkpoint_path: 检查点文件路径

        Returns:
            包含检查点信息的字典
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载模型权重
        self._load_model_state_dict(checkpoint['model_state_dict'])

        if 'target_model_state_dict' in checkpoint and checkpoint['target_model_state_dict']:
            self._load_target_model_state_dict(checkpoint['target_model_state_dict'])

        # 恢复训练状态（如果继续训练）
        if self._is_train and 'optimizer_state_dict' in checkpoint:
            self._load_optimizer_state_dict(checkpoint['optimizer_state_dict'])

        if 'noise_level' in checkpoint and checkpoint['noise_level'] is not None:
            self._noise_level = checkpoint['noise_level']

        print(f'Checkpoint loaded from: {checkpoint_path}')
        print(f'  Agent: {checkpoint.get("agent_name")}')
        print(f'  Environment: {checkpoint.get("env_name")}')
        print(f'  Episode: {checkpoint.get("episode")}')
        print(f'  Final metrics: {checkpoint.get("final_metrics")}')

        return checkpoint

    def _get_training_summary(self) -> Dict[str, Any]:
        """获取训练历史摘要"""
        summary = {}
        for name, values in self._training_history.items():
            if values:
                summary[name] = {
                    'final': values[-1],
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        return summary

    def _get_checkpoint_list(self, save_path: Path) -> List[Dict[str, Any]]:
        """获取当前保存目录中的所有检查点列表"""
        checkpoints = []
        for pt_file in save_path.glob('*.pt'):
            meta_file = pt_file.with_suffix('.pt').with_name(pt_file.stem + '_meta.json')
            ckpt_info = {
                'filename': pt_file.name,
                'path': str(pt_file),
            }
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                        ckpt_info['episode'] = meta.get('episode')
                        ckpt_info['is_best'] = meta.get('is_best', False)
                        ckpt_info['created_at'] = meta.get('created_at')
                        ckpt_info['headway_std'] = meta.get('final_metrics', {}).get('headway_std')
                except Exception:
                    pass
            checkpoints.append(ckpt_info)
        return sorted(checkpoints, key=lambda x: x.get('episode', 0))

    # 以下方法由子类实现
    def _get_model_state_dict(self) -> Dict[str, Any]:
        """获取模型状态字典（子类实现）"""
        return {}

    def _get_optimizer_state_dict(self) -> Dict[str, Any]:
        """获取优化器状态字典（子类实现）"""
        return {}

    def _get_target_model_state_dict(self) -> Dict[str, Any]:
        """获取目标网络状态字典（子类实现）"""
        return {}

    def _load_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载模型状态字典（子类实现）"""
        pass

    def _load_optimizer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载优化器状态字典（子类实现）"""
        pass

    def _load_target_model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载目标网络状态字典（子类实现）"""
        pass

    @abstractmethod
    def save_net(self, path: str) -> None:
        """简单保存（向后兼容）"""
        ...

    @abstractmethod
    def load_net(self, path: str) -> None:
        """简单加载（向后兼容）"""
        ...

    # def _calculate_total_arrival_rate(self) -> Dict[str, Dict[str, float]]:
    #     ''' Calculate the total arrival rate at each stop for each route by summing up the OD table by row.
    #     '''
    #     route_total_arrival_rate = defaultdict(dict)
    #     for route_id, route in self._blueprint.route_info.route_infos.items():
    #         for origin_stop_id, destination_rate in route.od_rate_table.items():
    #             total_origin_demand = sum(destination_rate.values())
    #             route_total_arrival_rate[route_id][origin_stop_id] = total_origin_demand

    #         last_stop_id = route.visit_seq_stops[-1]

    #         # # case 1. the last stop's arrival demand rate is 0, i.e., no one will get on the bus at the last stop
    #         # route_total_arrival_rate[route_id][last_stop_id] = 0.0

    #         # case 2. the last stop's arrival demand rate equals the last but one stop's arrival demand rate
    #         last_but_one_stop_id = route.visit_seq_stops[-2]
    #         route_total_arrival_rate[route_id][last_stop_id] = route_total_arrival_rate[route_id][last_but_one_stop_id]

    #     return dict(route_total_arrival_rate)
