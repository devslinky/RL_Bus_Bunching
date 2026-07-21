import random
from typing import Tuple, Dict

import numpy as np
import torch
import yaml

from agent.agent import Agent
from agent.do_nothing import DoNothing
from agent.model_based.simple_control_nonlinear import SimpleControlNonlinear
from agent.model_based.forward_headway_control import ForwardHeadwayControl
from agent.rl.naive_ddpg import Naive_DDPG
from agent.rl.event_ddpg import Event_DDPG
from agent.rl.local_spacing_ddpg import Local_Spacing_DDPG
from agent.rl.nonlinear_ddpg import Nonlinear_DDPG
from agent.rl.schedule_aware_ddpg import Schedule_Aware_DDPG
from agent.rl.pax_wait_ddpg_bc import Pax_Wait_DDPG_BC
from agent.rl.pax_wait_time_ddpg import Pax_Wait_Time_DDPG
from agent.rl.rich_ddpg import Rich_DDPG
from agent.rl.transformer_ddpg import Transformer_DDPG
from agent.rl.transformer_ddpg_reward import Transformer_DDPG_Reward
from agent.rl.rich_ppo import Rich_PPO
from agent.rl.transformer_ddpg_imp import Transformer_DDPG_Imp
from agent.rl.transformer_ddpg_claude import Transformer_DDPG_Claude
from agent.rl.transformer_ddpg_openai import SetTD3_BusHoldingAgent
from agent.rl.transformer_ddpg_claude_imp import Transformer_DDPG_Claude_Imp
from agent.rl.transformer_ddpg_openai_imp import SetTD3_BusHoldingAgent_Imp
from agent.rl.pax_wait_time_ddpg_pax_penal import Pax_Wait_Time_DDPG_Pax_Penal
from agent.rl.transformer_ddpg_claude_or import Transformer_DDPG_Claude_Or
from agent.rl.transformer_ddpg_openai_or import SetDDPG_BusHoldingAgent_Or
from agent.rl.mlp_ddpg_baseline import MLP_DDPG_Baseline
from agent.rl.lstm_ddpg_baseline import LSTM_DDPG_Baseline
from agent.rl.deepsets_ddpg_baseline import DeepSets_DDPG_Baseline

from setup.blueprint import Blueprint


def build_simulation_elements(config_path=None, eval_mode=False) -> Tuple[Blueprint, Agent, Dict, Dict]:
    ''' Build simulation elements as per config.yaml file

    Returns:
        blueprint: a Blueprint object that provide network and route schema as a whole
        agent: a specific Agent object
        run_config: specify episode number, episode duration, hold start time and hold end time
        record_config: configuration for recording in wandb, return an empty dict if not recording

    '''
    if config_path is not None:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            sanity_check(config)
    else:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            sanity_check(config)

    record_config = {}
    if config['wandb_config']['is_record_wandb']:
        running_agent_name = config['running_agent']
        record_config.update(config['agent_config'][running_agent_name])
        record_config['env_name'] = config['env_name']
        record_config['seed'] = config['seed']
        record_config['episode_num'] = config['episode_num']
        record_config['wandb_config'] = config['wandb_config']
        record_config['env_name'] = config['env_name']
        # record_config['lambda'] = 4
        # record_config['sigma'] = 5
        record_config['is_train'] = config['is_train']
    else:
        record_config = {}

    # set seed
    if 'seed' in config:
        seed = config['seed']
        np.random.seed(seed)
        random.seed(seed)
        torch.random.manual_seed(seed)

    # set running config
    run_config = {}
    run_config['episode_num'] = config['episode_num']
    run_config['episode_duration'] = config['episode_duration']
    run_config['fleet_size'] = config.get('fleet_size', None)
    run_config['hold_start_time'] = config['hold_start_time']
    run_config['hold_end_time'] = config['hold_end_time']
    run_config['has_schedule'] = config['has_schedule']
    run_config['metric_names'] = config['metric_names']
    run_config['warm_up'] = config.get('warm_up', False)
    run_config['env_name'] = config['env_name']  # 用于checkpoint命名

    # build blueprint for the environment
    env_name = config['env_name']
    blueprint = Blueprint(env_name)

    # create agent
    running_agent = config['running_agent']
    agent_config = config['agent_config'][running_agent]

    if eval_mode:
        return blueprint, None, run_config, record_config
    
    if agent_config['agent_name'] == 'Simple_Control_Nonlinear':
        agent = SimpleControlNonlinear(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Naive_DDPG':
        agent = Naive_DDPG(agent_config, blueprint)
    elif agent_config['agent_name'] == 'Do_Nothing':
        agent = DoNothing(agent_config, blueprint)
    elif agent_config['agent_name'] == 'Event_DDPG':
        agent = Event_DDPG(agent_config, blueprint)
    elif agent_config['agent_name'] == 'Local_Spacing_DDPG':
        agent = Local_Spacing_DDPG(agent_config, blueprint)
    elif agent_config['agent_name'] == 'Nonlinear_DDPG':
        agent = Nonlinear_DDPG(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Schedule_Aware_DDPG':
        agent = Schedule_Aware_DDPG(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Forward_Headway_Control':
        agent = ForwardHeadwayControl(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Pax_Wait_DDPG_BC':
        agent = Pax_Wait_DDPG_BC(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Pax_Wait_Time_DDPG':
        agent = Pax_Wait_Time_DDPG(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Rich_DDPG':
        agent = Rich_DDPG(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Transformer_DDPG':
        agent = Transformer_DDPG(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Transformer_DDPG_Reward':
        agent = Transformer_DDPG_Reward(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Rich_PPO':
        agent = Rich_PPO(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Transformer_DDPG_Imp':
        agent = Transformer_DDPG_Imp(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Transformer_DDPG_Claude':
        agent = Transformer_DDPG_Claude(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'SetTD3_BusHoldingAgent':
        agent = SetTD3_BusHoldingAgent(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Transformer_DDPG_Claude_Imp':
        agent = Transformer_DDPG_Claude_Imp(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'SetTD3_BusHoldingAgent_Imp':
        agent = SetTD3_BusHoldingAgent_Imp(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Pax_Wait_Time_DDPG_Pax_Penal':
        agent = Pax_Wait_Time_DDPG_Pax_Penal(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'Transformer_DDPG_Claude_Or':
        agent = Transformer_DDPG_Claude_Or(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'SetDDPG_BusHoldingAgent_Or':
        agent = SetDDPG_BusHoldingAgent_Or(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'MLP_DDPG_Baseline':
        agent = MLP_DDPG_Baseline(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'LSTM_DDPG_Baseline':
        agent = LSTM_DDPG_Baseline(agent_config, blueprint, run_config)
    elif agent_config['agent_name'] == 'DeepSets_DDPG_Baseline':
        agent = DeepSets_DDPG_Baseline(agent_config, blueprint, run_config)
    return blueprint, agent, run_config, record_config


def sanity_check(config: Dict):
    ''' Check if neccessary parameters are specified in the `config.yaml' file 

    '''
    assert 'episode_num' in config, 'episode_num must be specified in the config.yaml file'
    assert 'hold_start_time' in config, 'hold_start_time must be specified in the config.yaml file'
    assert 'hold_end_time' in config, 'hold_end_time must be specified in the config.yaml file'
    assert 'episode_duration' in config, 'episode_duration must be specified in the config.yaml file'
    assert 'env_name' in config, 'env_name must be specified in the config.yaml file'

    running_agent = config['running_agent']
    agent_config = config['agent_config'][running_agent]
    if agent_config['agent_name'] == 'Simple_Control_Nonlinear':
        assert 'fs' in agent_config, 'fs must be specified in the config.yaml file'
        assert 'slack' in agent_config, 'slack must be specified in the config.yaml file'
        assert 'base_type' in agent_config, 'base_type must be specified in the config.yaml file'

    # check conflicts between metric_names and has_schedule
    if config['has_schedule'] is False:
        assert 'schedule_deviation' not in config[
            'metric_names'], 'schedule_deviation cannot be calculated if has_schedule is False in the `config.yaml`'

    if agent_config['agent_name'] == 'Simple_Control_Nonlinear':
        assert config['has_schedule'] is True, 'has_schedule must be True if the agent is Simple_Control_Nonlinear in the `config.yaml`'

    # check the headway_std is always in the metric_names
    assert 'headway_std' in config['metric_names'], 'headway_std must be specified in the metric_names in the `config.yaml`'
